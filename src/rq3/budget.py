"""Estimate PBS walltime from development measurements, without GPU claims."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from .protocol import build_jobs, load_protocol
from .run import atomic_json


def estimate(config, rows, *, num_shards=4, seeds=None, timing_repeats=0,
             setup_minutes=30, safety_factor=1.3):
    """Use the slowest observed per-condition charged cost, plus warmup/setup.

    Charging thinking to B/D/E/F overcounts replay amortization deliberately.
    Region sizes differ on unseen examples; this is a scheduling estimate only.
    """
    from .run import shard_jobs
    if type(num_shards) is not int or not 1 <= num_shards <= 40:
        raise ValueError("num_shards must be an integer in 1..40")
    if type(timing_repeats) is not int or timing_repeats != 0 and timing_repeats < 3:
        raise ValueError("timing_repeats must be zero or at least three")
    if not math.isfinite(setup_minutes) or not math.isfinite(safety_factor) or setup_minutes < 0 or safety_factor < 1:
        raise ValueError("Require finite nonnegative setup and safety_factor >= 1")
    observed = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("phase") != "pilot" or row["example_uid"] not in config["_splits"]["k_dev"]:
            raise ValueError("Use development pilot timings, never test outcomes, to plan walltime")
        if row.get("protocol_hash") != config["protocol_hash"]:
            raise ValueError("Pilot budget input belongs to a different protocol")
        seconds = row.get("timings", {}).get("charged_total_seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Missing or invalid pilot charged_total_seconds")
        observed[row["condition"]].append(seconds)
    if not set("ABCDEF") <= set(observed):
        raise ValueError("Measure all six conditions in the development pilot first")
    costs = {condition: max(observed[condition]) for condition in "ABCDEF"}
    jobs = build_jobs(config, phase="all", seeds=seeds)
    estimates = []
    for index in range(num_shards):
        selected = shard_jobs(jobs, index, num_shards)
        warmup = sum(costs[c] for c in {job["condition"] for job in selected})
        workload = sum(costs[job["condition"]] for job in selected) * (1 + timing_repeats)
        seconds = (workload + warmup) * safety_factor + setup_minutes * 60
        hours = max(1, math.ceil(seconds / 3600))
        estimates.append({"shard_index": index, "examples": len({j["example_uid"] for j in selected}),
                          "quality_generations": len(selected), "timing_generations": len(selected) * timing_repeats,
                          "estimated_seconds_with_margin": seconds, "suggested_walltime": f"{hours:02d}:00:00"})
    longest = max(item["estimated_seconds_with_margin"] for item in estimates)
    return {"protocol_hash": config["protocol_hash"], "num_shards": num_shards,
            "timing_repeats": timing_repeats, "per_condition_slowest_pilot_seconds": costs,
            "setup_allowance_minutes": setup_minutes, "safety_factor": safety_factor,
            "array_walltime": f"{max(1, math.ceil(longest / 3600)):02d}:00:00", "shards": estimates,
            "warning": "Estimate only: unseen edit regions, filesystem load, model loading and JIT compilation can change runtime. Resume unfinished shards; retain fixed membership."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rq3.yaml")
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--timing-repeats", type=int, default=0)
    parser.add_argument("--setup-minutes", type=float, default=30)
    parser.add_argument("--safety-factor", type=float, default=1.3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.pilot).read_text().splitlines() if line.strip()]
    result = estimate(load_protocol(args.config), rows, num_shards=args.num_shards, seeds=args.seeds,
                      timing_repeats=args.timing_repeats, setup_minutes=args.setup_minutes, safety_factor=args.safety_factor)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

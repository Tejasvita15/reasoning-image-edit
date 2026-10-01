# src/rq2/check_2_1.py

"""
Validate RQ2.1 generation logs before handing outputs to the judge.

    python -m src.rq2.check_2_1 --log_dir logs/rq2 \
        --thought_cache outputs/rq2/thought_cache/k_dev_seed42.jsonl \
        --expected_n 90

Checks:
    - completeness: every (thinking, cfg) grid point has expected_n rows
    - flags: 50 steps, reflection off, one attempt, cfg as named
    - replay: thinking-off rows used the raw instruction; cached rows used
      the cached rewrite, identical across every cfg value
    - noise: one diffusion seed per example across all conditions
    - consistency: one offload mode, deterministic mode on, images exist
    - labelled repeats (--rep_label) match the main run (S7 tolerance)
    - live-thinking rows reproduce the cached rewrite and image
      (replay equivalence)

Prints a per-condition summary and exits 1 if any check fails.
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.rq2.check_determinism import MAD_TOLERANCE, mean_abs_diff


def load_rows(log_dir: Path, split: str, seed: int):
    """
    Latest successful row per (condition, uid), plus failure counts.
    """

    rows = defaultdict(dict)
    failures = Counter()

    for path in sorted(log_dir.glob(f"*_{split}_seed{seed}.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record.get("status") != "success":
                    failures[record.get("condition")] += 1
                    continue

                rows[record["condition"]][record["example_uid"]] = record

    return rows, failures


def median_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def main():

    parser = argparse.ArgumentParser(
        description="Check RQ2.1 logs."
    )

    parser.add_argument("--log_dir", required=True, type=Path)
    parser.add_argument("--split", default="k_dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", default="rq2_step1x")
    parser.add_argument("--thinking", nargs="+", default=["off", "cached"])
    parser.add_argument("--cfgs", nargs="+", type=float, default=[2, 6, 10])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--expected_n", type=int, required=True)
    parser.add_argument("--thought_cache", type=Path, default=None)
    parser.add_argument("--rep_label", default=None)

    args = parser.parse_args()

    from src.rq2.common import condition_name

    rows, failures = load_rows(args.log_dir, args.split, args.seed)

    cache = {}

    if args.thought_cache:
        with open(args.thought_cache, "r", encoding="utf-8") as f:
            cache = {
                r["example_uid"]: r
                for r in map(json.loads, filter(str.strip, f))
            }

    problems = []

    def fail(message):
        problems.append(message)
        print(f"FAIL: {message}")

    # =========================================================
    # Completeness
    # =========================================================

    expected = {
        (thinking, cfg): condition_name(
            args.experiment_name, thinking, cfg, args.steps
        )
        for thinking in args.thinking
        for cfg in args.cfgs
    }

    for name in expected.values():
        n = len(rows.get(name, {}))

        if n != args.expected_n:
            fail(f"{name}: {n}/{args.expected_n} successful rows")

    # =========================================================
    # Per-row invariants
    # =========================================================

    all_rows = [
        row
        for by_uid in rows.values()
        for row in by_uid.values()
    ]

    for row in all_rows:
        where = f"{row['condition']} {row['example_uid']}"
        thinking = row.get("thinking_condition")

        if row["num_inference_steps"] != args.steps:
            fail(f"{where}: steps {row['num_inference_steps']}")

        if row["reflection_enabled"] or row["max_try_cnt"] != 1:
            fail(f"{where}: reflection must be off with max_try_cnt 1")

        if row["attempt_count"] != 1:
            fail(f"{where}: attempt_count {row['attempt_count']}")

        expected_calls = args.steps * (2 if row["true_cfg_scale"] > 1 else 1)

        if row.get("generator_calls") != expected_calls:
            fail(f"{where}: generator_calls {row.get('generator_calls')}")

        if not Path(row["output_path"]).exists():
            fail(f"{where}: image missing {row['output_path']}")

        determinism = row["full_config"]["rq2"]["determinism"]

        if not determinism.get("use_deterministic_algorithms"):
            fail(f"{where}: deterministic mode off")

        if thinking == "off":
            if row["prompt_used"] != row["instruction"]:
                fail(f"{where}: off row did not use the raw instruction")

            if row["reformat_prompt"] is not None:
                fail(f"{where}: off row has a reformat_prompt")

        if thinking == "cached":
            if row["prompt_used"] != row["reformat_prompt"]:
                fail(f"{where}: prompt_used != reformat_prompt")

            if row["pipeline_enable_thinking_mode"]:
                fail(f"{where}: replay must keep pipeline thinking off")

            cached = cache.get(row["example_uid"])

            if cache and cached is None:
                fail(f"{where}: example not in the thought cache")

            if cached is not None:
                if row["prompt_used"] != cached["rewrite"]:
                    fail(f"{where}: prompt differs from the cached rewrite")

                if row["latency_thinking_sec"] != cached["latency_thinking_sec"]:
                    fail(f"{where}: thinking latency not charged from cache")

    # =========================================================
    # Cross-row invariants
    # =========================================================

    seeds_by_uid = defaultdict(set)
    cached_prompts_by_uid = defaultdict(set)

    for row in all_rows:
        seeds_by_uid[row["example_uid"]].add(
            row["stage_seeds"]["diffusion_0"]
        )

        if row.get("thinking_condition") == "cached":
            cached_prompts_by_uid[row["example_uid"]].add(
                row["prompt_used"]
            )

    for uid, seeds in seeds_by_uid.items():
        if len(seeds) != 1:
            fail(f"{uid}: {len(seeds)} different diffusion seeds")

    for uid, prompts in cached_prompts_by_uid.items():
        if len(prompts) != 1:
            fail(f"{uid}: cached prompt differs across conditions")

    offload_modes = {row.get("offload_mode") for row in all_rows}

    if len(offload_modes) > 1:
        fail(f"mixed offload modes: {sorted(map(str, offload_modes))}")

    commits = {row["env"]["repo_commit"] for row in all_rows if "env" in row}
    dirty = any(row["env"].get("repo_dirty") for row in all_rows if "env" in row)

    if len(commits) > 1 or dirty:
        print(
            f"WARN: rows come from {len(commits)} commit(s), "
            f"dirty tree={dirty}; check nothing changed mid-experiment"
        )

    # =========================================================
    # Pairwise image checks
    # =========================================================

    def image_of(thinking, cfg, label=None):
        name = condition_name(
            args.experiment_name, thinking, cfg, args.steps, label
        )
        return rows.get(name, {})

    print("\nPairwise checks")

    for thinking in args.thinking:
        low = image_of(thinking, min(args.cfgs))
        high = image_of(thinking, max(args.cfgs))
        shared = sorted(set(low) & set(high))

        if not shared:
            continue

        mads = [
            mean_abs_diff(low[uid]["output_path"], high[uid]["output_path"])
            for uid in shared
        ]

        # Sanity only: the dial should change the image.
        print(
            f"  guidance effect {thinking} cfg{min(args.cfgs):g} vs "
            f"cfg{max(args.cfgs):g}: median MAD {statistics.median(mads):.3f}, "
            f"{sum(m == 0.0 for m in mads)}/{len(mads)} unchanged"
        )

    if args.rep_label:
        for (thinking, cfg), name in expected.items():
            main_rows = rows.get(name, {})
            rep_rows = image_of(thinking, cfg, args.rep_label)

            for uid in sorted(set(main_rows) & set(rep_rows)):
                mad = mean_abs_diff(
                    main_rows[uid]["output_path"],
                    rep_rows[uid]["output_path"],
                )

                print(f"  S7 repeat {name} {uid}: MAD {mad:.4f}")

                if mad >= MAD_TOLERANCE:
                    fail(f"S7 repeat {name} {uid}: MAD {mad:.4f}")

    for cfg in args.cfgs:
        live_rows = image_of("live", cfg)
        cached_rows = image_of("cached", cfg)

        for uid in sorted(set(live_rows) & set(cached_rows)):
            live = live_rows[uid]
            same_rewrite = live["reformat_prompt"] == cached_rows[uid]["prompt_used"]

            mad = mean_abs_diff(
                live["output_path"],
                cached_rows[uid]["output_path"],
            )

            print(
                f"  live vs replay cfg{cfg:g} {uid}: "
                f"rewrite identical={same_rewrite} MAD {mad:.4f}"
            )

            if not same_rewrite:
                fail(
                    f"{uid}: live rewrite differs from the cache "
                    "(thinking not reproducible across processes)"
                )

            elif mad >= MAD_TOLERANCE:
                fail(f"{uid}: replay image differs from live thinking")

    # =========================================================
    # Summary
    # =========================================================

    print(
        "\ncondition | ok | failed attempts | pipeline s | generate s "
        "| think s | reconstructed s | peak VRAM GB | tokens"
    )

    for name in sorted(rows):
        by_uid = rows[name].values()

        print(
            f"{name} | {len(rows[name])} | {failures.get(name, 0)} | "
            f"{fmt(median_or_none(r['latency_pipeline_median_sec'] for r in by_uid))} | "
            f"{fmt(median_or_none(r['latency_generate_sec'] for r in by_uid))} | "
            f"{fmt(median_or_none(r['latency_thinking_sec'] for r in by_uid))} | "
            f"{fmt(median_or_none(r['latency_reconstructed_total_sec'] for r in by_uid))} | "
            f"{fmt(median_or_none(r['peak_vram_gb'] for r in by_uid), 1)} | "
            f"{fmt(median_or_none(r['thinking_token_count'] for r in by_uid), 0)}"
        )

    print(f"\nOffload mode: {sorted(map(str, offload_modes))}")
    print("RQ2.1 check:", "PASS" if not problems else f"FAIL ({len(problems)})")

    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()

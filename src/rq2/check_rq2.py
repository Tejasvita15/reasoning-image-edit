"""
Validate RQ2.1 / RQ2.2 generation logs before handing outputs to the judge.

    python -m src.rq2.check_rq2 --log_dir logs/rq2 \
        --thought_cache outputs/rq2/thought_cache/k_dev_seed42.jsonl \
        --expected_n 90

    # RQ2.2: guidance 6, 20 and 28 steps
    python -m src.rq2.check_rq2 --log_dir logs/rq2_2 --cfgs 6 --steps 20 28 \
        --thought_cache outputs/rq2/thought_cache/k_dev_seed42.jsonl \
        --expected_n 90

Checks:
    - completeness: every (thinking, cfg, steps) grid point has expected_n rows
    - flags: steps as named, reflection off, one attempt, cfg as named
    - replay: thinking-off rows used the raw instruction; cached rows used
      the cached rewrite, identical across every cfg value
    - noise: one diffusion seed per example across all conditions
    - consistency: one offload mode, deterministic mode on, images exist
    - labelled repeats (--rep_label) match the main run (S7 tolerance)
    - live-thinking rows reproduce the cached rewrite and image
      (replay equivalence)
    - optional: rewrites match a reference thought cache, and images match
      the same (condition, example) in reference log directories

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
        description="Check RQ2 generation logs (RQ2.1, RQ2.2)."
    )

    parser.add_argument("--log_dir", required=True, type=Path)
    parser.add_argument("--split", default="k_dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", default="rq2_step1x")
    parser.add_argument("--thinking_is_appended", action="store_true",
                        help="UniReason: the thought is appended after the instruction, "
                             "so the prompt stays the raw instruction.")
    parser.add_argument("--thinking", nargs="+", default=["off", "cached"])
    parser.add_argument("--cfgs", nargs="+", type=float, default=[2, 6, 10])
    parser.add_argument("--steps", nargs="+", type=int, default=[50])
    parser.add_argument("--expected_n", type=int, required=True)
    parser.add_argument("--thought_cache", type=Path, default=None)
    parser.add_argument("--rep_label", default=None)
    parser.add_argument("--reference_cache", type=Path, default=None,
                        help="Rewrites must equal this cache for shared examples.")
    parser.add_argument("--reference_log_dir", type=Path, nargs="+", default=[],
                        help="Images must equal the same condition/example here.")

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
        (thinking, cfg, steps): condition_name(
            args.experiment_name, thinking, cfg, steps
        )
        for thinking in args.thinking
        for cfg in args.cfgs
        for steps in args.steps
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

        if row["num_inference_steps"] not in args.steps:
            fail(f"{where}: steps {row['num_inference_steps']}")

        if row["reflection_enabled"] or row["max_try_cnt"] != 1:
            fail(f"{where}: reflection must be off with max_try_cnt 1")

        if row["attempt_count"] != 1:
            fail(f"{where}: attempt_count {row['attempt_count']}")

        expected_calls = row["num_inference_steps"] * (2 if row["true_cfg_scale"] > 1 else 1)

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
            if args.thinking_is_appended:
                if row["prompt_used"] != row["instruction"]:
                    fail(f"{where}: appended replay must use the raw instruction")
                thought = row["reformat_prompt"]
            else:
                if row["prompt_used"] != row["reformat_prompt"]:
                    fail(f"{where}: prompt_used != reformat_prompt")
                thought = row["prompt_used"]

            if row["pipeline_enable_thinking_mode"]:
                fail(f"{where}: replay must keep pipeline thinking off")

            cached = cache.get(row["example_uid"])

            if cache and cached is None:
                fail(f"{where}: example not in the thought cache")

            if cached is not None:
                if thought != cached["rewrite"]:
                    fail(f"{where}: thought differs from the cached rewrite")

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
                row["reformat_prompt"] if args.thinking_is_appended else row["prompt_used"]
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

    def image_of(thinking, cfg, steps, label=None):
        name = condition_name(
            args.experiment_name, thinking, cfg, steps, label
        )
        return rows.get(name, {})

    print("\nPairwise checks")

    def dial_effect(description, low, high):
        shared = sorted(set(low) & set(high))

        if not shared:
            return

        mads = [
            mean_abs_diff(low[uid]["output_path"], high[uid]["output_path"])
            for uid in shared
        ]

        # Sanity only: the dial should change the image.
        print(
            f"  {description}: median MAD {statistics.median(mads):.3f}, "
            f"{sum(m == 0.0 for m in mads)}/{len(mads)} unchanged"
        )

    for thinking in args.thinking:
        for steps in args.steps:
            if len(args.cfgs) > 1:
                dial_effect(
                    f"guidance effect {thinking} steps{steps} cfg{min(args.cfgs):g} vs cfg{max(args.cfgs):g}",
                    image_of(thinking, min(args.cfgs), steps),
                    image_of(thinking, max(args.cfgs), steps),
                )

        for cfg in args.cfgs:
            if len(args.steps) > 1:
                dial_effect(
                    f"steps effect {thinking} cfg{cfg:g} steps{min(args.steps)} vs steps{max(args.steps)}",
                    image_of(thinking, cfg, min(args.steps)),
                    image_of(thinking, cfg, max(args.steps)),
                )

    if args.rep_label:
        for (thinking, cfg, steps), name in expected.items():
            main_rows = rows.get(name, {})
            rep_rows = image_of(thinking, cfg, steps, args.rep_label)

            for uid in sorted(set(main_rows) & set(rep_rows)):
                mad = mean_abs_diff(
                    main_rows[uid]["output_path"],
                    rep_rows[uid]["output_path"],
                )

                print(f"  S7 repeat {name} {uid}: MAD {mad:.4f}")

                if mad >= MAD_TOLERANCE:
                    fail(f"S7 repeat {name} {uid}: MAD {mad:.4f}")

    for cfg, steps in ((c, s) for c in args.cfgs for s in args.steps):
        live_rows = image_of("live", cfg, steps)
        cached_rows = image_of("cached", cfg, steps)

        for uid in sorted(set(live_rows) & set(cached_rows)):
            live = live_rows[uid]
            cached_thought = (cached_rows[uid]["reformat_prompt"] if args.thinking_is_appended
                              else cached_rows[uid]["prompt_used"])
            same_rewrite = live["reformat_prompt"] == cached_thought

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
    # Reference checks (optional)
    # =========================================================

    if args.reference_cache:
        with open(args.reference_cache, "r", encoding="utf-8") as f:
            reference = {
                r["example_uid"]: r["rewrite"]
                for r in map(json.loads, filter(str.strip, f))
            }

        shared = sorted(set(cache) & set(reference))

        if not shared:
            fail("no examples shared with the reference cache")

        differing = [uid for uid in shared if cache[uid]["rewrite"] != reference[uid]]

        print(f"  rewrites vs reference cache: {len(shared) - len(differing)}/{len(shared)} identical")

        for uid in differing:
            fail(f"{uid}: rewrite differs from the reference cache")

    for reference_dir in args.reference_log_dir:
        reference_rows, _ = load_rows(reference_dir, args.split, args.seed)
        compared = identical = 0

        for name, by_uid in rows.items():
            for uid, row in by_uid.items():
                other = reference_rows.get(name, {}).get(uid)

                if other is None:
                    continue

                compared += 1
                mad = mean_abs_diff(row["output_path"], other["output_path"])
                identical += mad == 0.0

                if mad >= MAD_TOLERANCE:
                    fail(f"{name} {uid}: image differs from {reference_dir} (MAD {mad:.4f})")

        print(f"  images vs {reference_dir}: {identical}/{compared} pixel-identical")

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
    print("RQ2 check:", "PASS" if not problems else f"FAIL ({len(problems)})")

    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
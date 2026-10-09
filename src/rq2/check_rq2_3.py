# src/rq2/check_rq2_3.py

"""
Validate the RQ2.3 generation logs (and the selector logs, if given) before
judging.

    python -m src.rq2.check_rq2_3 --log_dir logs/rq2_3 --seeds 42 1234 7 \
        --expected_n 40 --cfg 6 --steps 28 --selector_dir logs/rq2_3/selector

Checks, per seed and example:
    - every expected example logged once as a success, nothing extra;
    - same configuration on every row; prompt = the cached rewrite;
    - candidate images exist and match their logged sha256;
    - both arms share candidate 1 (pixel-identical) and the same noise seed;
    - the feedback run retried exactly when its reflection did not accept
      candidate 1, and native selection matches the returned final image;
    - the resampling redraw exists exactly when the feedback run retried,
      with the logged derived seeds, and differs from candidate 1;
    - optionally, candidate 1 equals the image of the same configuration in
      another run (e.g. the seed-42 K-test frozen run);
    - selector rows (if given) cover every example, score the logged images,
      and each arm keeps its higher-scoring candidate.

Prints a summary (retry rate, retry types, picks, latency) and
"RQ2.3 check: PASS" or "RQ2.3 check: FAIL".
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from src.rq2.common import (
    condition_name,
    file_sha256,
    stratified_subset,
)


EXPERIMENT = "rq2_3_step1x"


def parse_args():

    parser = argparse.ArgumentParser(description="Validate RQ2.3 logs.")

    parser.add_argument("--log_dir", default="logs/rq2_3")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--split", default="k_small")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--expected_n", type=int, required=True)
    parser.add_argument("--cfg", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--selector_dir", default=None)

    parser.add_argument(
        "--reference_log_dir",
        default=None,
        help="Run of the same configuration (reflection off) whose images "
             "candidate 1 must equal, for the seeds it covers.",
    )

    parser.add_argument(
        "--allow_forced_redraw",
        action="store_true",
        help="Smoke runs redraw even when the reflection accepted.",
    )

    return parser.parse_args()


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def same_pixels(path_a, path_b):
    import numpy as np
    from PIL import Image

    with Image.open(path_a) as a, Image.open(path_b) as b:
        return np.array_equal(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB")))


def main():

    args = parse_args()

    from src.seeding import derive_seed

    errors = []
    warnings = []

    def check(ok, message):
        if not ok:
            errors.append(message)

        return ok

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    split_uids = list(splits[args.split])

    expected_uids = set(
        split_uids
        if args.expected_n == len(split_uids)
        else stratified_subset(split_uids, args.expected_n)
    )

    check(len(expected_uids) == args.expected_n,
          f"expected_n {args.expected_n} does not match the split")

    condition = condition_name(EXPERIMENT, "cached", args.cfg, args.steps)
    summary = []

    for seed in args.seeds:

        log_path = Path(args.log_dir) / f"{condition}_{args.split}_seed{seed}.jsonl"

        if not check(log_path.exists(), f"missing {log_path}"):
            continue

        latest = {}

        for row in read_jsonl(log_path):
            latest[row["example_uid"]] = row

        failed = sorted(uid for uid, row in latest.items() if row.get("status") != "success")
        check(not failed, f"seed {seed}: unresolved failures {failed[:5]}")

        rows = {uid: row for uid, row in latest.items() if row.get("status") == "success"}

        check(set(rows) | set(failed) == expected_uids,
              f"seed {seed}: examples missing={len(expected_uids - set(latest))} "
              f"extra={len(set(latest) - expected_uids)}")

        caches = {}

        for uid, row in sorted(rows.items()):

            where = f"seed {seed} {uid}"
            fb = row["feedback"]
            rs = row["resample"]
            paths = row["candidate_paths"]

            # ---------------------------------------------------
            # Configuration and prompt
            # ---------------------------------------------------

            check(row["condition"] == condition and row["base_seed"] == seed
                  and row["split_name"] == args.split, f"{where}: identity")
            check(row["true_cfg_scale"] == args.cfg and row["num_inference_steps"] == args.steps,
                  f"{where}: guidance/steps")

            gens = row["generation_configs"]
            check(gens["feedback"]["enable_reflection_mode"] is True
                  and gens["feedback"]["max_try_cnt"] == 2
                  and gens["plain"]["enable_reflection_mode"] is False
                  and gens["plain"]["max_try_cnt"] == 1
                  and not gens["feedback"]["enable_thinking_mode"]
                  and not gens["plain"]["enable_thinking_mode"],
                  f"{where}: run flags")

            cache_path = row["thought_cache_path"]

            if cache_path not in caches:
                caches[cache_path] = (
                    file_sha256(cache_path),
                    {r["example_uid"]: r for r in read_jsonl(cache_path)},
                )

            cache_sha, cache = caches[cache_path]
            check(row["thought_cache_sha256"] == cache_sha, f"{where}: thought cache changed")
            check(cache[uid]["base_seed"] == seed, f"{where}: cache seed")
            check(row["prompt_used"] == cache[uid]["rewrite"], f"{where}: prompt is not the cached rewrite")

            # ---------------------------------------------------
            # Files
            # ---------------------------------------------------

            for name, path in paths.items():
                if check(Path(path).exists(), f"{where}: missing {path}"):
                    check(file_sha256(path) == row["candidate_sha256"][name],
                          f"{where}: {name} differs from its logged sha256")

            # ---------------------------------------------------
            # Shared candidate 1
            # ---------------------------------------------------

            check(row["cand1_identical_across_arms"] is True,
                  f"{where}: arms do not share candidate 1 "
                  f"(max abs diff {row['cand1_max_abs_diff']})")

            noise = derive_seed(seed, uid, "diffusion_0")
            check(fb["generator_seed"] == noise and rs["cand1_generator_seed"] == noise,
                  f"{where}: candidate 1 noise seed")

            # ---------------------------------------------------
            # Feedback run
            # ---------------------------------------------------

            retried = fb["attempt_count"] == 2
            check(fb["attempt_count"] in (1, 2), f"{where}: attempt_count {fb['attempt_count']}")
            check(len(fb["think_info"]) == fb["attempt_count"] == len(fb["best_info"]),
                  f"{where}: reflections per candidate")
            check(retried == (fb["reflection_decisions"][0] != "accept"),
                  f"{where}: retry does not follow the reflection")
            check(retried == ("feedback_cand2" in paths), f"{where}: feedback candidate 2 file")
            check(fb["native_index_matches_final_image"] is True,
                  f"{where}: native selection index does not match the final image")

            # ---------------------------------------------------
            # Resampling redraw
            # ---------------------------------------------------

            if rs["redraw"] and not retried:
                (warnings if args.allow_forced_redraw else errors).append(
                    f"{where}: forced redraw (feedback accepted candidate 1)"
                )

            check(rs["redraw"] == ("resample_cand2" in paths), f"{where}: redraw file")
            check(rs["redraw"] or not retried, f"{where}: feedback retried but no redraw")

            if rs["redraw"]:
                redraw_seed = derive_seed(seed, uid, "resample_1")
                check(rs["redraw_base_seed"] == redraw_seed
                      and rs["redraw_generator_seed"] == derive_seed(redraw_seed, uid, "diffusion_0"),
                      f"{where}: redraw seeds")
                check(row["candidate_sha256"]["resample_cand2"] != row["candidate_sha256"]["cand1"],
                      f"{where}: redraw identical to candidate 1")

            # ---------------------------------------------------
            # Reference run (same configuration, reflection off)
            # ---------------------------------------------------

            if args.reference_log_dir:
                reference_name = condition_name("rq2_step1x", "cached", args.cfg, args.steps)
                reference = sorted(Path(args.reference_log_dir).glob(f"{reference_name}_*_seed{seed}.jsonl"))

                for reference_log in reference:
                    for ref in read_jsonl(reference_log):
                        if ref.get("example_uid") == uid and ref.get("status") == "success":
                            check(ref["prompt_used"] == row["prompt_used"],
                                  f"{where}: prompt differs from {reference_log.name}")
                            check(same_pixels(ref["output_path"], paths["cand1"]),
                                  f"{where}: candidate 1 differs from {reference_log.name}")

        # -------------------------------------------------------
        # Selector
        # -------------------------------------------------------

        selected = None

        if args.selector_dir:
            selector_path = Path(args.selector_dir) / f"rq2_3_selector_{args.split}_seed{seed}.jsonl"

            if check(selector_path.exists(), f"missing {selector_path}"):
                selections = {r["example_uid"]: r for r in read_jsonl(selector_path)
                              if r.get("status") == "success"}
                check(set(selections) == set(rows),
                      f"seed {seed}: selector rows missing={len(set(rows) - set(selections))} "
                      f"extra={len(set(selections) - set(rows))}")
                check(len({json.dumps({k: v for k, v in s["selector"].items() if k != "gpu"}, sort_keys=True)
                           for s in selections.values()}) <= 1,
                      f"seed {seed}: selector settings differ between rows")

                selected = Counter()

                for uid, sel in selections.items():
                    if uid not in rows:
                        continue

                    where = f"seed {seed} {uid} selector"
                    check(sel["candidate_sha256"] == rows[uid]["candidate_sha256"],
                          f"{where}: scored images differ from the generation log")
                    check(sel["selector"]["instruction_source"] == "original_kris_instruction"
                          and sel["instruction"] == rows[uid]["instruction"],
                          f"{where}: not the original instruction")

                    for arm, second in (("feedback", "feedback_cand2"), ("resample", "resample_cand2")):
                        choice = sel["selection"][arm]
                        has_second = second in rows[uid]["candidate_paths"]
                        check((len(choice["candidates"]) == 2) == has_second,
                              f"{where}: {arm} candidates")

                        if has_second:
                            first, other = sel["scores"]["cand1"]["score"], sel["scores"][second]["score"]
                            check(choice["selected"] == (second if other > first else "cand1"),
                                  f"{where}: {arm} did not keep its higher-scoring candidate")
                            selected[(arm, choice["selected"] != "cand1")] += 1

        # -------------------------------------------------------
        # Summary
        # -------------------------------------------------------

        if rows:
            fbs = [row["feedback"] for row in rows.values()]
            retried_rows = [fb for fb in fbs if fb["attempt_count"] == 2]
            line = (
                f"seed {seed}: {len(rows)} examples | feedback retried {len(retried_rows)} "
                f"({dict(Counter(fb['retry_type'] for fb in retried_rows))}) | "
                f"native kept candidate 2 in {sum(fb['native_selected_index'] == 1 for fb in retried_rows)} | "
                f"median s: feedback run "
                f"{statistics.median(fb['latency_pipeline_sec'] for fb in fbs):.1f}, plain "
                f"{statistics.median(row['resample']['cand1_latency_pipeline_sec'] for row in rows.values()):.1f}"
            )

            if selected is not None:
                line += (
                    f" | selector kept candidate 2: feedback {selected[('feedback', True)]}"
                    f"/{selected[('feedback', True)] + selected[('feedback', False)]}, "
                    f"resample {selected[('resample', True)]}"
                    f"/{selected[('resample', True)] + selected[('resample', False)]}"
                )

            summary.append(line)

            if not retried_rows:
                warnings.append(f"seed {seed}: no feedback retries, so the candidate-2 path was not exercised")

    # ===========================================================
    # Report
    # ===========================================================

    print("=" * 60)

    for line in summary:
        print(line)

    for message in warnings:
        print(f"WARN: {message}")

    for message in errors[:50]:
        print(f"ERROR: {message}")

    if len(errors) > 50:
        print(f"... {len(errors) - 50} more errors")

    print("=" * 60)
    print(f"RQ2.3 check: {'PASS' if not errors else 'FAIL'}")

    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()

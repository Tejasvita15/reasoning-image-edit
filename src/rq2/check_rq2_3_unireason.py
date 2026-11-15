# src/rq2/check_rq2_3_unireason.py
"""
Validate UniReason RQ2.3 outputs for one seed (k_small examples).

    python -m src.rq2.check_rq2_3_unireason --seed 42

Checks:
  - generation log: one success row per example, correct condition, cfg and steps
  - images: first, feedback, resample exist and match the logged sha256
  - candidate 1 is shared across arms
  - feedback and resample candidate 2 differ from candidate 1 (warning otherwise)
  - timing: one row per example with the latency fields
  - thought cache: one row per example with a thinking latency
  - selector: one row per example, same images and instruction, each arm
    keeps its higher-scoring candidate
  - config: explicit flags in configs/rq2/unireason.yaml

Not checked from logs: derived seeds and prompts (not recorded there).
"""
import argparse
import json
from pathlib import Path

import yaml

from src.rq2.common import file_sha256


CONDITION = "rq2_3_unireason_think-cached_cfg4_steps50"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Validate UniReason RQ2.3 outputs.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--split", default="k_small")
    parser.add_argument("--gen_log", default=None)
    parser.add_argument("--timing", default=None)
    parser.add_argument("--selector", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--config", default="configs/rq2/unireason.yaml")
    args = parser.parse_args()

    seed = args.seed
    gen_log = Path(args.gen_log or f"logs/rq2_3_unireason/gen/{CONDITION}_{args.split}_seed{seed}.jsonl")
    timing_path = Path(args.timing or f"outputs/rq2_3_unireason_timed/timing_seed_{seed}.jsonl")
    selector_path = Path(args.selector or f"logs/rq2_3_unireason/selector/rq2_3_selector_{args.split}_seed{seed}.jsonl")
    cache_path = Path(args.cache or f"outputs/rq2_3_unireason/thought_cache/{args.split}_seed{seed}.jsonl")

    errors, warnings = [], []

    def check(ok, message):
        if not ok:
            errors.append(message)
        return ok

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    expected = set(splits[args.split])

    # Config flags
    generation = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["generation"]
    check(generation.get("num_inference_steps") == 50, "config: num_inference_steps must be 50")
    check(generation.get("true_cfg_scale") == 4.0, "config: true_cfg_scale must be 4.0")
    check(generation.get("enable_reflection_mode") is False, "config: enable_reflection_mode must be false")
    check(generation.get("max_try_cnt") == 1, "config: max_try_cnt must be 1")

    # Generation log
    rows = {}
    if check(gen_log.exists(), f"missing {gen_log}"):
        for row in read_jsonl(gen_log):
            if row.get("status") == "success":
                rows[row["example_uid"]] = row

    check(set(rows) == expected,
          f"gen log: missing={len(expected - set(rows))} extra={len(set(rows) - expected)}")

    for uid, row in rows.items():
        check(row["condition"] == CONDITION and row["base_seed"] == seed and row["split_name"] == args.split,
              f"{uid}: identity")
        check(row["true_cfg_scale"] == 4.0 and row["num_inference_steps"] == 50, f"{uid}: cfg or steps")
        check(row["cand1_identical_across_arms"] is True, f"{uid}: candidate 1 not shared")

        for name, path in row["candidate_paths"].items():
            if check(Path(path).exists(), f"{uid}: missing {path}"):
                check(file_sha256(path) == row["candidate_sha256"][name],
                      f"{uid}: {name} sha256 differs from the log")

        sha = row["candidate_sha256"]
        if sha["feedback_cand2"] == sha["cand1"]:
            warnings.append(f"{uid}: feedback candidate equals candidate 1")
        if sha["resample_cand2"] == sha["cand1"]:
            warnings.append(f"{uid}: resample candidate equals candidate 1")

    # Timing
    timing = {}
    if check(timing_path.exists(), f"missing {timing_path}"):
        timing = {r["example_uid"]: r for r in read_jsonl(timing_path)}
        check(set(timing) == expected,
              f"timing: missing={len(expected - set(timing))} extra={len(set(timing) - expected)}")
        for uid, t in timing.items():
            check(t.get("seed") == seed, f"{uid}: timing seed")
            for key in ("cached_think_sec", "first_gen_sec", "feedback_gen_sec", "resample_gen_sec"):
                check(isinstance(t.get(key), (int, float)) and t[key] > 0, f"{uid}: timing {key}")

    # Thought cache
    if check(cache_path.exists(), f"missing {cache_path}"):
        cache = {r["example_uid"]: r for r in read_jsonl(cache_path)}
        check(expected <= set(cache), f"thought cache missing {len(expected - set(cache))}")
        for uid in expected & set(cache):
            check((cache[uid].get("latency_thinking_sec") or 0) > 0, f"{uid}: no thinking latency in cache")

    # Selector
    selections = {}
    if check(selector_path.exists(), f"missing {selector_path}"):
        selections = {r["example_uid"]: r for r in read_jsonl(selector_path) if r.get("status") == "success"}
        check(set(selections) == expected,
              f"selector: missing={len(expected - set(selections))} extra={len(set(selections) - expected)}")

    for uid, sel in selections.items():
        if uid not in rows:
            continue
        gen = rows[uid]
        check(sel["candidate_sha256"] == gen["candidate_sha256"], f"{uid}: selector scored different images")
        check(sel["instruction"] == gen["instruction"], f"{uid}: selector instruction differs")
        check(sel["selector"]["instruction_source"] == "original_kris_instruction",
              f"{uid}: selector not on the original instruction")

        first = sel["scores"]["cand1"]["score"]
        for arm, second in (("feedback", "feedback_cand2"), ("resample", "resample_cand2")):
            other = sel["scores"][second]["score"]
            expected_pick = second if other > first else "cand1"
            check(sel["selection"][arm]["selected"] == expected_pick,
                  f"{uid}: {arm} did not keep its higher-scoring candidate")

    # Report
    print("=" * 60)
    print(f"seed {seed}: gen rows {len(rows)} | timing rows {len(timing)} | selector rows {len(selections)}")
    for message in warnings[:20]:
        print(f"WARN: {message}")
    for message in errors[:50]:
        print(f"ERROR: {message}")
    if len(errors) > 50:
        print(f"... {len(errors) - 50} more errors")
    print("=" * 60)
    print(f"RQ2.3 UniReason check: {'PASS' if not errors else 'FAIL'}")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()

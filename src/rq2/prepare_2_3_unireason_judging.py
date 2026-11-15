# src/rq2/prepare_2_3_judging.py

"""
Turn the RQ2.3 generation logs into judge-ready logs for src.rq2_manual.

    python -m src.rq2.prepare_2_3_judging

Every distinct RQ2.3 image is judged once (blind to arm and selector), so
both arms, the native-selection view and "did candidate 2 beat candidate 1"
come from the same scores. The unchanged export tool needs complete grids
(every condition for every split example), so per seed there are two bundles:

    cand1   40 shared first candidates           splits: data/splits.json
    cand2   feedback + resampling candidate 2,   splits: data/splits.json with
            on the examples where feedback        k_small restricted to
            retried                               those examples

Writes (refuses to overwrite):
    logs/rq2_3/judge/cand1/<condition>_k_small_seed<S>.jsonl
    logs/rq2_3/judge/cand2/<condition>_k_small_seed<S>.jsonl   (two per seed)
    outputs/rq2_3/judge/splits_cand2_seed<S>.json
    outputs/rq2_3/judge/judge_prep.json                         provenance
"""

import argparse
import hashlib
import json
from pathlib import Path

from src.rq2.common import file_sha256


CONDITION = "rq2_3_unireason_think-cached_cfg4_steps50"

# Judge condition suffix -> generation candidate name.
CANDIDATES = {
    "cand1": "cand1",
    "feedback-cand2": "feedback_cand2",
    "resample-cand2": "resample_cand2",
}


def parse_args():

    parser = argparse.ArgumentParser(description="Prepare RQ2.3 judging logs.")

    parser.add_argument("--log_dir", default="logs/rq2_3")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--split", default="k_small")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 7])
    parser.add_argument("--judge_log_dir", default="logs/rq2_3/judge")
    parser.add_argument("--out_dir", default="outputs/rq2_3/judge")

    return parser.parse_args()


def row_hash(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():

    args = parse_args()

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    judge_root = Path(args.judge_log_dir)
    out_dir = Path(args.out_dir)

    if judge_root.exists():
        raise SystemExit(f"{judge_root} exists; judging logs are written once.")

    provenance = {"generation_logs": {}, "retried": {}, "outputs": {}}

    for seed in args.seeds:

        log_path = Path(args.log_dir) / f"{CONDITION}_{args.split}_seed{seed}.jsonl"

        latest = {}

        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["example_uid"]] = row

        if set(latest) != set(splits[args.split]) or any(
            row["status"] != "success" for row in latest.values()
        ):
            raise SystemExit(f"{log_path}: incomplete; run check_rq2_3 first.")

        provenance["generation_logs"][str(log_path)] = file_sha256(str(log_path))

        rows = {suffix: [] for suffix in CANDIDATES}

        for uid in sorted(latest):

            source = latest[uid]

            if not source["cand1_identical_across_arms"]:
                raise SystemExit(f"seed {seed} {uid}: arms do not share candidate 1.")

            for suffix, candidate in CANDIDATES.items():

                if candidate not in source["candidate_paths"]:
                    continue

                path = source["candidate_paths"][candidate]

                if file_sha256(path) != source["candidate_sha256"][candidate]:
                    raise SystemExit(f"seed {seed} {uid}: {path} changed since generation.")

                # Fields read by src.rq2_manual export, plus provenance.
                rows[suffix].append({
                    "status": "success",
                    "example_uid": uid,
                    "condition": f"{CONDITION}_{suffix}",
                    "base_seed": seed,
                    "split_name": args.split,
                    "knowledge_type": source["knowledge_type"],
                    "thinking_condition": source["thinking_condition"],
                    "true_cfg_scale": source["true_cfg_scale"],
                    "num_inference_steps": source["num_inference_steps"],
                    "instruction": source["instruction"],
                    "output_path": path,
                    "rq2_3_candidate": candidate,
                    "source_generation_log": str(log_path),
                    "source_generation_row_sha256": row_hash(source),
                })

        retried = sorted(row["example_uid"] for row in rows["feedback-cand2"])

        if retried != sorted(row["example_uid"] for row in rows["resample-cand2"]):
            raise SystemExit(f"seed {seed}: redraws do not match feedback retries.")

        for suffix, bundle in (("cand1", "cand1"), ("feedback-cand2", "cand2"), ("resample-cand2", "cand2")):
            path = judge_root / bundle / f"{CONDITION}_{suffix}_{args.split}_seed{seed}.jsonl"
            write_jsonl(path, rows[suffix])
            provenance["outputs"][str(path)] = len(rows[suffix])

        # data/splits.json with only the split under judgement restricted.
        cand2_splits = dict(splits)
        cand2_splits[args.split] = retried

        splits_path = out_dir / f"splits_cand2_seed{seed}.json"
        splits_path.parent.mkdir(parents=True, exist_ok=True)

        with open(splits_path, "x", encoding="utf-8") as f:
            json.dump(cand2_splits, f, indent=2)
            f.write("\n")

        provenance["retried"][str(seed)] = retried
        provenance["outputs"][str(splits_path)] = len(retried)

        print(f"seed {seed}: cand1 {len(rows['cand1'])} | retried {len(retried)} "
              f"-> cand2 outputs {2 * len(retried)}")

    provenance["splits"] = {args.splits: file_sha256(args.splits)}

    with open(out_dir / "judge_prep.json", "x", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")

    print(f"Wrote {judge_root} and {out_dir / 'judge_prep.json'}")


if __name__ == "__main__":
    main()

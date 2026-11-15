# src/rq2/make_unireason_2_3_gen_log.py

"""
Build a generation-style RQ2.3 log for UniReason from its image folders, so the
team's selector (src/rq2/select_2_3.py) and judging prep can read it unchanged.

    python -m src.rq2.make_unireason_2_3_gen_log --seed 42

Reads:
    {image_root}/first/seed_{seed}/{uid}.png      shared candidate 1
    {image_root}/feedback/seed_{seed}/{uid}.png   feedback candidate 2
    {image_root}/resample/seed_{seed}/{uid}.png   resampling candidate 2

Writes (refuses to overwrite):
    {out_dir}/{CONDITION}_{split}_seed{seed}.jsonl
"""

import argparse
import json
from pathlib import Path

from src.dataset import prepare_example
from src.rq2.common import file_sha256, load_rq2_examples
from src.taxonomy import get_knowledge_type


CONDITION = "rq2_3_unireason_think-cached_cfg4_steps50"


def parse_args():
    parser = argparse.ArgumentParser(description="UniReason RQ2.3 generation-style log.")

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="k_small")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--image_root", default="outputs/rq2_3_unireason")
    parser.add_argument("--out_dir", default="logs/rq2_3_unireason/gen")

    return parser.parse_args()


def main():
    args = parse_args()

    seed_dir = f"seed_{args.seed}"
    root = Path(args.image_root)

    out_path = Path(args.out_dir) / f"{CONDITION}_{args.split}_seed{args.seed}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        raise SystemExit(f"{out_path} exists; generation logs are written once.")

    count = 0

    with open(out_path, "x", encoding="utf-8") as f:
        for example in load_rq2_examples(args.splits, args.split):
            raw = prepare_example(example)
            uid = raw["example_uid"]

            candidates = {
                "cand1": root / "first" / seed_dir / f"{uid}.png",
                "feedback_cand2": root / "feedback" / seed_dir / f"{uid}.png",
                "resample_cand2": root / "resample" / seed_dir / f"{uid}.png",
            }

            missing = [name for name, path in candidates.items() if not path.exists()]
            if missing:
                raise SystemExit(f"{uid}: missing {missing} under {root}")

            row = {
                "status": "success",
                "example_uid": uid,
                "condition": CONDITION,
                "split_name": args.split,
                "base_seed": args.seed,
                "knowledge_type": get_knowledge_type(uid.rsplit("__", 1)[0]),
                "instruction": raw["instruction"],
                "thinking_condition": "cached",
                "true_cfg_scale": 4.0,
                "num_inference_steps": 50,
                "cand1_identical_across_arms": True,
                "candidate_paths": {name: str(path) for name, path in candidates.items()},
                "candidate_sha256": {
                    name: file_sha256(str(path)) for name, path in candidates.items()
                },
            }

            f.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} rows to {out_path}")


if __name__ == "__main__":
    main()
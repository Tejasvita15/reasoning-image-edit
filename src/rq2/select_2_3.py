# src/rq2/select_2_3.py

"""
RQ2.3 selector: EditReward-MiMo-VL-7B picks the better candidate in each arm.

    python -m src.rq2.select_2_3 --gen_log <RQ2.3 generation log for one seed>

Runs in its own environment (the EditReward package pins transformers 4.57.0;
Step1X needs 4.55.0), after src/rq2/run_rq2_3.py. The selector is a fixed
reward model from a different family than the judge, used identically for
both arms, and never sees judge scores (RQ2_plan.md, hard rule).

Per example, every distinct candidate (shared candidate 1, the feedback
candidate 2, the resampling candidate 2) is scored once, one image per call,
against the KRIS source image and the ORIGINAL KRIS instruction (not the
thinking rewrite), using EditReward's documented settings
(reward_dim="overall_detail", rm_head_type="ranknet_multi_head"; the score is
the first output, the predicted reward mean). Each arm keeps the
higher-scoring of its candidates; a tie keeps candidate 1. An arm with one
candidate needs no selection and is charged no selector time. Each scoring
call is timed once, CUDA-synchronised, after an unlogged warm-up.

Log: {out_dir}/rq2_3_selector_{split}_seed{seed}.jsonl, one row per example
(resume skips logged examples).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from src.rq2.common import (
    enable_determinism,
    file_sha256,
    load_rq2_examples,
)


CHECKPOINT_REPO = "TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508"
CHECKPOINT_REVISION = "4a702b4c1d6b45e597ced81beddcfaea3741f675"
CONFIG_PATH = "config/EditReward-MiMo-VL-7B-SFT-2508.yaml"
REWARD_DIM = "overall_detail"
RM_HEAD_TYPE = "ranknet_multi_head"

ARMS = {
    "feedback": "feedback_cand2",
    "resample": "resample_cand2",
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="RQ2.3 EditReward selection."
    )

    parser.add_argument("--gen_log", required=True)
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--out_dir", default="logs/rq2_3/selector")
    parser.add_argument("--checkpoint_repo", default=CHECKPOINT_REPO)
    parser.add_argument("--checkpoint_revision", default=CHECKPOINT_REVISION)

    return parser.parse_args()


def git_commit(path):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_generation_rows(path):
    """
    Latest successful row per example of one RQ2.3 generation log.
    """

    rows = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("status") == "success":
                rows[row["example_uid"]] = row

    if not rows:
        raise SystemExit(f"No successful RQ2.3 generations in {path}")

    if len({(row["split_name"], row["base_seed"]) for row in rows.values()}) != 1:
        raise SystemExit(f"{path} mixes splits or seeds.")

    return rows


def load_done(path):
    done = set()

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)

                if row.get("status") == "success":
                    done.add(row["example_uid"])

    return done


def main():

    args = parse_args()

    enable_determinism()

    import torch
    import transformers
    from huggingface_hub import snapshot_download

    import EditReward
    from EditReward import EditRewardInferencer

    from src.dataset import (
        get_decoded_source_images,
        prepare_example,
    )

    rows = load_generation_rows(args.gen_log)
    first_row = next(iter(rows.values()))
    split, seed = first_row["split_name"], first_row["base_seed"]

    out_path = Path(args.out_dir) / f"rq2_3_selector_{split}_seed{seed}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_path)
    pending = [uid for uid in rows if uid not in done]

    print("=" * 60)
    print(f"RQ2.3 selector | {args.gen_log} -> {out_path}")
    print(f"Examples: {len(rows)} | pending {len(pending)}")
    print("=" * 60)

    if not pending:
        print("Nothing to do; every example is scored.")
        return

    # =========================================================
    # Source images (KRIS, same loader as generation)
    # =========================================================

    examples = load_rq2_examples(args.splits, split)
    sources = {}

    for idx in range(len(examples)):
        uid = prepare_example(examples[idx])["example_uid"]

        if uid in rows:
            images = get_decoded_source_images(examples[idx])

            if len(images) != 1:
                raise SystemExit(f"{uid}: expected one source image")

            sources[uid] = images[0]

    missing = sorted(set(rows) - set(sources))

    if missing:
        raise SystemExit(f"Source images not found: {missing[:5]}")

    # =========================================================
    # Selector model
    # =========================================================

    checkpoint_path = snapshot_download(
        args.checkpoint_repo,
        revision=args.checkpoint_revision,
    )

    inferencer = EditRewardInferencer(
        config_path=CONFIG_PATH,
        checkpoint_path=checkpoint_path,
        device="cuda",
        reward_dim=REWARD_DIM,
        rm_head_type=RM_HEAD_TYPE,
    )

    selector = {
        "model": args.checkpoint_repo,
        "checkpoint_revision": args.checkpoint_revision,
        "checkpoint_path": checkpoint_path,
        "base_model": inferencer.model.config._name_or_path,
        "base_model_commit": getattr(inferencer.model.config, "_commit_hash", None),
        "config_path": CONFIG_PATH,
        "reward_dim": REWARD_DIM,
        "rm_head_type": RM_HEAD_TYPE,
        "instruction_source": "original_kris_instruction",
        "editreward_commit": git_commit(Path(EditReward.__file__).resolve().parent),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    def score(uid, path):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            reward = inferencer.reward(
                prompts=[rows[uid]["instruction"]],
                image_src=[sources[uid]],
                image_paths=[str(path)],
            )

        torch.cuda.synchronize()
        latency = time.perf_counter() - start

        return {
            "score": float(reward[0][0].item()),
            "raw": [float(value) for value in reward[0].tolist()],
            "latency_sec": latency,
        }

    # Warm-up (unlogged).
    warm_uid = pending[0]
    score(warm_uid, rows[warm_uid]["candidate_paths"]["cand1"])

    # =========================================================
    # Score and select
    # =========================================================

    with open(out_path, "a", encoding="utf-8") as log:

        for position, uid in enumerate(pending, 1):

            row = rows[uid]
            paths = row["candidate_paths"]

            # Score exactly the images the generation log recorded.
            for name, path in paths.items():
                if file_sha256(path) != row["candidate_sha256"][name]:
                    raise SystemExit(f"{uid}: {path} differs from the generation log")

            scores = {
                name: score(uid, paths[name])
                for name in ("cand1", *ARMS.values())
                if name in paths
            }

            selection = {}

            for arm, second in ARMS.items():

                if second not in scores:
                    selection[arm] = {
                        "candidates": ["cand1"],
                        "selected": "cand1",
                        "selected_index": 0,
                        "latency_selector_sec": 0.0,
                    }
                    continue

                first_score = scores["cand1"]["score"]
                second_score = scores[second]["score"]
                pick_second = second_score > first_score

                selection[arm] = {
                    "candidates": ["cand1", second],
                    "selected": second if pick_second else "cand1",
                    "selected_index": 1 if pick_second else 0,
                    "tie": second_score == first_score,
                    "latency_selector_sec":
                        scores["cand1"]["latency_sec"] + scores[second]["latency_sec"],
                }

            record = {
                "status": "success",
                "example_uid": uid,
                "condition": row["condition"],
                "split_name": split,
                "base_seed": seed,
                "knowledge_type": row["knowledge_type"],
                "instruction": row["instruction"],
                "generation_log": args.gen_log,
                "candidate_paths": paths,
                "candidate_sha256": row["candidate_sha256"],
                "scores": scores,
                "selection": selection,
                "selector": selector,
            }

            log.write(json.dumps(record) + "\n")
            log.flush()

            print(
                f"[{position}/{len(pending)}] {uid} | "
                + " | ".join(f"{name} {value['score']:.3f}" for name, value in scores.items())
                + " | picks "
                + ", ".join(f"{arm} {sel['selected']}" for arm, sel in selection.items())
            )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

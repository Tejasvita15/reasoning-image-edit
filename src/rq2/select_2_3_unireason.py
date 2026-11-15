# src/rq2/select_2_3_unireason.py

"""
RQ2.3 selector for UniReason: EditReward-MiMo-VL-7B picks the better candidate in each arm.

    python -m src.rq2.select_2_3_unireason --seed 42

Reads the RQ2.3 image folders written by src/rq2/run_rq2_3_unireason.py:
    {out_root}/first/seed_{seed}/{uid}.png     shared candidate 1
    {out_root}/feedback/seed_{seed}/{uid}.png  feedback candidate 2
    {out_root}/resample/seed_{seed}/{uid}.png  resampling candidate 2

Each candidate is scored once per example against the KRIS source image and the
ORIGINAL KRIS instruction, using EditReward's documented settings. Each arm keeps
the higher-scoring of (candidate 1, its candidate 2). A tie keeps candidate 1.
The selector never reads judge scores.

Timing: each scoring call is timed with CUDA synchronisation, after one unlogged
warm-up call. Rows are appended to a JSON Lines log; reruns skip logged examples.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.dataset import get_decoded_source_images, prepare_example
from src.rq2.common import load_rq2_examples


CONFIG_PATH = "/home/users/ntu/kirubhah/EditReward/EditReward/config/EditReward-MiMo-VL-7B-SFT-2508.yaml"
CHECKPOINT_PATH = "/scratch/users/ntu/kirubhah/models/EditReward-MiMo-VL-7B-SFT-2508"
REWARD_DIM = "overall_detail"
RM_HEAD_TYPE = "ranknet_multi_head"
ARMS = ("feedback", "resample")


def parse_args():
    parser = argparse.ArgumentParser(description="RQ2.3 UniReason EditReward selection.")

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--out_root", default="outputs/rq2_3_unireason")
    parser.add_argument("--out_dir", default="logs/rq2_3_unireason/selector")
    parser.add_argument("--config_path", default=CONFIG_PATH)
    parser.add_argument("--checkpoint_path", default=CHECKPOINT_PATH)

    return parser.parse_args()


def load_done(path):
    done = set()

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)

                if row.get("status") == "success":
                    done.add(row["example_uid"])

    return done


def example_uid(raw):
    uid = raw.get("example_uid") or raw.get("uid")

    if uid is None:
        raise SystemExit("Prepared example has no example_uid/uid field; check prepare_example output keys.")

    return uid


def timed_score(inferencer, prompt, src_path, img_path):
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        reward = inferencer.reward(
            prompts=[prompt],
            image_src=[str(src_path)],
            image_paths=[str(img_path)],
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return float(reward[0].item()), elapsed


def main():
    args = parse_args()

    # Imported here so the module can be read without the EditReward environment.
    from EditReward import EditRewardInferencer

    out_root = Path(args.out_root)
    seed_dir = f"seed_{args.seed}"

    first_dir = out_root / "first" / seed_dir
    arm_dirs = {arm: out_root / arm / seed_dir for arm in ARMS}

    if not first_dir.exists():
        raise SystemExit(f"Missing candidate folder: {first_dir}")

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / f"rq2_3_selector_k_small_seed{args.seed}.jsonl"
    src_tmp = out_path / "source_images"
    src_tmp.mkdir(parents=True, exist_ok=True)

    done = load_done(log_path)

    # Prepared examples, keyed by uid, for the source image and original instruction.
    prepared = {}
    for raw in (prepare_example(e) for e in load_rq2_examples(args.splits, "k_small")):
        prepared[example_uid(raw)] = raw

    uids = sorted(p.stem for p in first_dir.glob("*.png"))
    if not uids:
        raise SystemExit(f"No PNGs in {first_dir}")

    inferencer = EditRewardInferencer(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device="cuda",
        reward_dim=REWARD_DIM,
        rm_head_type=RM_HEAD_TYPE,
    )

    # One unlogged warm-up call, so the first timed call does not include model setup.
    warm_uid = uids[0]
    warm_raw = prepared[warm_uid]
    warm_src = src_tmp / f"{warm_uid}.png"
    images = get_decoded_source_images(warm_raw)
    assert len(images) == 1, f"{warm_uid}: expected one source image, got {len(images)}"
    images[0].convert("RGB").save(warm_src)
    timed_score(inferencer, warm_raw["instruction"], warm_src, first_dir / f"{warm_uid}.png")
    print("warm-up done", flush=True)

    with open(log_path, "a", encoding="utf-8") as log:
        for uid in uids:
            if uid in done:
                continue

            if uid not in prepared:
                print(f"skip {uid}: not in k_small", flush=True)
                continue

            paths = {"first": first_dir / f"{uid}.png"}
            for arm in ARMS:
                paths[arm] = arm_dirs[arm] / f"{uid}.png"

            missing = [name for name, p in paths.items() if not p.exists()]
            if missing:
                print(f"skip {uid}: missing {missing}", flush=True)
                continue

            raw = prepared[uid]
            instruction = raw["instruction"]

            src_path = src_tmp / f"{uid}.png"
            images = get_decoded_source_images(raw)
            assert len(images) == 1, f"{uid}: expected one source image, got {len(images)}"
            images[0].convert("RGB").save(src_path)

            # Score the shared candidate 1 once, then each arm's candidate 2.
            scores, times = {}, {}
            for name in ("first",) + ARMS:
                scores[name], times[name] = timed_score(
                    inferencer, instruction, src_path, paths[name]
                )

            row = {
                "example_uid": uid,
                "seed": args.seed,
                "split_name": "k_small",
                "status": "success",
                "instruction": instruction,
                "score_first": scores["first"],
                "time_first_sec": times["first"],
            }

            for arm in ARMS:
                # Keep candidate 2 only if strictly higher; a tie keeps candidate 1.
                chosen = arm if scores[arm] > scores["first"] else "first"
                row[f"score_{arm}"] = scores[arm]
                row[f"time_{arm}_sec"] = times[arm]
                row[f"selected_{arm}"] = chosen
                row[f"selected_{arm}_path"] = str(paths[chosen])

            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                f"{uid} | first {scores['first']:.3f} | feedback {scores['feedback']:.3f} "
                f"-> {row['selected_feedback']} | resample {scores['resample']:.3f} "
                f"-> {row['selected_resample']}",
                flush=True,
            )

    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()

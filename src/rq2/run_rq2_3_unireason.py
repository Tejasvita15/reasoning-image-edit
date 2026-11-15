import argparse
import json
import time
from pathlib import Path

import torch

from src.config import load_config
from src.dataset import get_decoded_source_images, prepare_example
from src.rq2.common import load_rq2_examples, load_thought_cache
from src.seeding import derive_seed, set_global_seed
from src.unireason_runner import UniReasonRunner
from src.rq2.refine_unireason import refine_with_cot

# Refine prompt copied from the released UniReason KRIS refine script.
REFINE_PROMPT = (
    "Strictly apply all thinking_text: {think} to refine and correct the previous "
    "edit image, and ensure the final edited image is a perfect, faithful, and "
    "unambiguous to edit the source image follwing the instruction: {prompt} "
    "better than the previous edit image."
)


def timed(fn):
    # Synchronize before and after so the timer covers the GPU work.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - t0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--thought_cache", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base_config = load_config(args.config)
    gen = base_config["generation"]
    steps = int(gen["num_inference_steps"])
    cfg_text = float(gen["true_cfg_scale"])
    cfg_img = float(gen.get("cfg_img_scale", 2.0))

    set_global_seed(args.seed)

    examples = [prepare_example(e) for e in load_rq2_examples(args.splits, "k_small")]
    if args.limit:
        examples = examples[: args.limit]

    cache, _ = load_thought_cache(
        args.thought_cache,
        expected_uids=[e["example_uid"] for e in examples],
        base_seed=args.seed,
        model_id=base_config["model"]["model_id"],
    )

    runner = UniReasonRunner(base_config)

    out = Path(args.output_dir)
    dirs = {name: out / name / f"seed_{args.seed}" for name in ("first", "resample", "feedback")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    timing_path = out / f"timing_seed_{args.seed}.jsonl"
    with open(timing_path, "w") as timing_f:
        for raw in examples:
            uid = raw["example_uid"]
            instruction = raw["instruction"]
            plan = cache[uid]["rewrite"]
            images = get_decoded_source_images(raw)

            # Shared first candidate, from the cached plan.
            first_out, t_first = timed(lambda: runner.run(
                images=images,
                instruction=instruction,
                example_uid=uid,
                base_seed=args.seed,
                cached_thought=plan,
            ))
            first = first_out["final_image"]
            first.save(dirs["first"] / f"{uid}.png")

            # Resampling arm: same plan, new seed, no look at the first image.
            resampled, t_resample = timed(lambda: runner._generate(
                images=images,
                instruction=instruction,
                think_output=plan,
                diffusion_seed=derive_seed(args.seed, uid, "resample_1"),
                num_timesteps=steps,
                cfg_text_scale=cfg_text,
                cfg_img_scale=cfg_img,
            ))
            resampled.save(dirs["resample"] / f"{uid}.png")

            # Feedback arm: released refine step, conditioned on source and first image.
            refined, t_feedback = timed(lambda: refine_with_cot(
                runner,
                src_image=images[0],
                first_image=first,
                enhance_prompt=REFINE_PROMPT.format(think=plan, prompt=instruction),
                think_text=plan,
                seed=derive_seed(args.seed, uid, "feedback_1"),
                num_timesteps=steps,
                cfg_text_scale=cfg_text,
                cfg_img_scale=cfg_img,
            ))
            refined.save(dirs["feedback"] / f"{uid}.png")

            record = {
                "example_uid": uid,
                "seed": args.seed,
                "cached_think_sec": float(cache[uid]["latency_thinking_sec"]),
                "first_gen_sec": t_first,
                "resample_gen_sec": t_resample,
                "feedback_gen_sec": t_feedback,
            }
            timing_f.write(json.dumps(record) + "\n")
            timing_f.flush()
            print(f"done {uid}", flush=True)


if __name__ == "__main__":
    main()

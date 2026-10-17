# src/rq2/run_rq2_3.py

"""
Step1X RQ2.3 generation driver: feedback vs resampling on K-small.

Fixed for RQ2.3 (RQ2_plan.md): thinking on with the frozen plan (the cached
rewrite replayed with pipeline thinking off, S4), the RQ2.1-selected guidance
and the RQ2.2 frozen step count. One process per seed runs, for every example:

    1. Feedback run: reflection on, max_try_cnt=2 (Step1X's native correction).
       Attempt 1 is candidate 1. If the reflection does not accept it, Step1X
       makes candidate 2: <#Reflection> edits candidate 1 with the refined
       prompt ("refine"); <#Failed> or an invalid reflection redraws from the
       source image with the same prompt ("restart"). The second attempt
       continues the same diffusion generator stream. Native selection picks
       the final image from the reflection scores.

    2. Plain run: reflection off, same prompt and same seed. This is
       candidate 1 of the resampling arm. Step1XRunner seeds the generator from
       (base seed, UID, "diffusion_0") on every call, so it must be
       pixel-identical to the feedback run's candidate 1 (both arms share
       candidate 1); the driver checks and logs this. Its latency is the
       resampling arm's candidate-1 cost.

    3. Resampling redraw, only where the feedback run made a second candidate
       (budget matching: when the reflection accepts candidate 1, both arms
       stop at one candidate). Same cached prompt, reflection off, new noise:
       Step1XRunner is called with base seed derive_seed(seed, UID,
       "resample_1"), so the generator seed is
       derive_seed(that, UID, "diffusion_0"). Both are logged.

The EditReward selector runs separately (src/rq2/select_2_3.py, its own
environment) on the saved candidates. Every pipeline call is timed once
(CUDA-synchronised, after an unlogged warm-up); thinking is charged from the
cache.

Outputs: {output_dir}/{cand1,feedback_cand2,resample_cand2}/k_small/seed_{seed}/{uid}.png
Log:     {log_dir}/rq2_3_step1x_think-cached_cfg{cfg}_steps{steps}_k_small_seed{seed}.jsonl
         (one row per example; resume skips logged examples)
"""

import argparse
import copy
import time
from pathlib import Path

from src.rq2.common import (
    UNCONDITIONAL_BRANCH,
    collect_env,
    condition_name,
    count_tokens,
    enable_determinism,
    file_sha256,
    fix_text_encoder_offload,
    load_rq2_examples,
    load_thought_cache,
    offload_mode,
)


EXPERIMENT = "rq2_3_step1x"

CANDIDATE_DIRS = (
    "cand1",
    "feedback_cand2",
    "resample_cand2",
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Step1X RQ2.3 (feedback vs resampling)."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--split", default="k_small", choices=["k_small"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--true_cfg_scale", type=float, required=True)
    parser.add_argument("--num_inference_steps", type=int, required=True)

    parser.add_argument(
        "--thought_cache",
        required=True,
        help="Cached rewrites for this seed (a K-test cache covers K-small).",
    )

    parser.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Stratified subset (smoke only).",
    )

    parser.add_argument("--output_dir", default="outputs/rq2_3")
    parser.add_argument("--log_dir", default="logs/rq2_3")

    parser.add_argument(
        "--force_redraw",
        action="store_true",
        help="Smoke only: redraw even when the reflection accepted "
             "candidate 1, so the resampling path is exercised.",
    )

    return parser.parse_args()


def classify_reflection(text):
    """
    Step1X's decision on one reflection string, mirroring
    Step1XEditThinker.format_text (pipeline_step1x_edit_thinker.py).

    Returns (decision, tag): decision is accept / refine / restart.
    """

    from diffusers.pipelines.step1x_edit.pipeline_step1x_edit_thinker import (
        validate_string,
    )

    text = str(text)

    if not validate_string(text):
        return "restart", "invalid"

    if "<#Reflection>" in text:
        return "refine", "reflection"

    if "<#Success>" in text:
        return "accept", "success"

    if "<#Others>" in text:
        return "accept", "others"

    return "restart", "failed"


def pixel_diff(a, b):
    """
    (identical, max absolute difference) of two PIL images.
    """

    import numpy as np

    a = np.asarray(a.convert("RGB"), dtype=np.int16)
    b = np.asarray(b.convert("RGB"), dtype=np.int16)

    if a.shape != b.shape:
        return False, None

    return bool((a == b).all()), int(np.abs(a - b).max())


def build_configs(base_config, args, determinism):
    """
    Validated configs for the feedback run and the plain runs.
    """

    from src.config import validate_config

    configs = {}

    for name, reflection, max_try_cnt in (
        ("feedback", True, 2),
        ("plain", False, 1),
    ):
        config = copy.deepcopy(base_config)
        generation = config["generation"]

        # Every flag explicit (rule 3). Thinking is replayed from the cache,
        # so the pipeline's own thinking stays off.
        generation["true_cfg_scale"] = float(args.true_cfg_scale)
        generation["num_inference_steps"] = args.num_inference_steps
        generation["enable_thinking_mode"] = False
        generation["enable_reflection_mode"] = reflection
        generation["max_try_cnt"] = max_try_cnt

        config["experiment_name"] = condition_name(
            EXPERIMENT,
            "cached",
            args.true_cfg_scale,
            args.num_inference_steps,
        )

        config["output_dir"] = args.output_dir
        config["log_file"] = args.log_dir

        config["rq2"] = {
            "subquestion": "2.3",
            "run": name,
            "thinking_condition": "cached",
            "thought_cache": args.thought_cache,
            "determinism": determinism,
        }

        # Shared RQ1 rules: reflection on <=> max_try_cnt == 2.
        validate_config(config)

        configs[name] = config

    return configs


def generator_calls(generation, attempts):
    """
    Transformer forward passes: two per step with true CFG.
    """

    return (
        generation["num_inference_steps"]
        * (2 if generation["true_cfg_scale"] > 1 else 1)
        * attempts
    )


def run_example(
    *,
    runner,
    configs,
    raw_example,
    example,
    prompt,
    args,
    out_paths,
    tokenizer,
):
    """
    Feedback run, plain candidate 1 and (if needed) the resampling redraw
    for one example. Returns the per-arm log fields.
    """

    from src.dataset import get_decoded_source_images
    from src.seeding import derive_seed

    uid = example["example_uid"]

    preprocess_start = time.perf_counter()
    images = get_decoded_source_images(raw_example)
    latency_preprocess_sec = time.perf_counter() - preprocess_start

    # =========================================================
    # 1. Feedback run (native reflection, at most 2 attempts)
    # =========================================================

    runner.config = configs["feedback"]

    fb = runner.run(
        images=images,
        instruction=prompt,
        example_uid=uid,
        base_seed=args.seed,
    )

    fb_candidates = fb["candidate_images"]
    think_info = [str(text) for text in (fb["think_info"] or [])]

    if len(think_info) != len(fb_candidates):
        raise RuntimeError(
            f"{uid}: {len(fb_candidates)} candidates but "
            f"{len(think_info)} reflections."
        )

    decisions = [classify_reflection(text) for text in think_info]
    retried = len(fb_candidates) == 2

    # Step1X retries exactly when its reflection does not accept candidate 1.
    if retried != (decisions[0][0] != "accept"):
        raise RuntimeError(
            f"{uid}: reflection decision {decisions[0]} does not match "
            f"{len(fb_candidates)} candidate(s)."
        )

    native_index = fb["selected_candidate_index"]
    native_matches_final, _ = pixel_diff(
        fb_candidates[native_index],
        fb["final_image"],
    )

    # =========================================================
    # 2. Plain run: candidate 1 of the resampling arm
    # =========================================================

    runner.config = configs["plain"]

    plain = runner.run(
        images=images,
        instruction=prompt,
        example_uid=uid,
        base_seed=args.seed,
    )

    cand1 = plain["final_image"]
    cand1_identical, cand1_max_abs_diff = pixel_diff(cand1, fb_candidates[0])

    # =========================================================
    # 3. Resampling redraw (budget-matched)
    # =========================================================

    redraw = None
    redraw_base_seed = None

    if retried or args.force_redraw:
        redraw_base_seed = derive_seed(args.seed, uid, "resample_1")

        redraw = runner.run(
            images=images,
            instruction=prompt,
            example_uid=uid,
            base_seed=redraw_base_seed,
        )

    # =========================================================
    # Save
    # =========================================================

    postprocess_start = time.perf_counter()

    paths = {}

    cand1.save(out_paths["cand1"])
    paths["cand1"] = out_paths["cand1"]

    if retried:
        fb_candidates[1].save(out_paths["feedback_cand2"])
        paths["feedback_cand2"] = out_paths["feedback_cand2"]

    if redraw is not None:
        redraw["final_image"].save(out_paths["resample_cand2"])
        paths["resample_cand2"] = out_paths["resample_cand2"]

    # Diagnosis only: kept when the two arms' candidate 1 differ.
    if not cand1_identical:
        out_paths["feedback_cand1"].parent.mkdir(parents=True, exist_ok=True)
        fb_candidates[0].save(out_paths["feedback_cand1"])
        paths["feedback_cand1"] = out_paths["feedback_cand1"]

    latency_postprocess_sec = time.perf_counter() - postprocess_start

    # =========================================================
    # Log fields
    # =========================================================

    feedback_generation = configs["feedback"]["generation"]
    plain_generation = configs["plain"]["generation"]

    feedback = {
        "stage_seeds": fb["stage_seeds"],
        "generator_seed": fb["stage_seeds"]["diffusion_0"],
        "attempt_count": len(fb_candidates),
        "retry_triggered": retried,
        "retry_type": decisions[0][0] if retried else None,
        "reflection_decisions": [decision for decision, _ in decisions],
        "reflection_tags": [tag for _, tag in decisions],

        # The prompt and input image of attempt 2: a refine edits
        # candidate 1, a restart redraws from the source.
        "attempt2_prompt": (
            think_info[0].split("<#Reflection>")[1]
            if retried and decisions[0][0] == "refine"
            else (prompt if retried else None)
        ),
        "attempt2_input": (
            ("candidate_1" if decisions[0][0] == "refine" else "source")
            if retried
            else None
        ),

        "think_info": think_info,
        "best_info": fb["best_info"],
        "reflection_token_counts": [
            count_tokens(tokenizer, text) for text in think_info
        ],

        "native_selected_index": native_index,
        "native_selected_is_last": fb["selected_is_last_candidate"],
        "native_index_matches_final_image": native_matches_final,
        "stop_reason": fb["stop_reason"],

        # Generation and reflection rounds of every attempt, and native
        # selection; one timed call.
        "latency_pipeline_sec": fb["latency_pipeline_sec"],
        "peak_vram_gb": fb["peak_vram_gb"],
        "generator_calls": generator_calls(
            feedback_generation, len(fb_candidates)
        ),
        "reflection_rounds": len(think_info),
    }

    resample = {
        "cand1_stage_seeds": plain["stage_seeds"],
        "cand1_generator_seed": plain["stage_seeds"]["diffusion_0"],
        "cand1_latency_pipeline_sec": plain["latency_pipeline_sec"],
        "cand1_peak_vram_gb": plain["peak_vram_gb"],

        "redraw": redraw is not None,
        "redraw_forced": redraw is not None and not retried,
        "redraw_base_seed": redraw_base_seed,
        "redraw_generator_seed": (
            redraw["stage_seeds"]["diffusion_0"] if redraw is not None else None
        ),
        "redraw_latency_pipeline_sec": (
            redraw["latency_pipeline_sec"] if redraw is not None else None
        ),
        "redraw_peak_vram_gb": (
            redraw["peak_vram_gb"] if redraw is not None else None
        ),

        "generator_calls": generator_calls(
            plain_generation, 2 if redraw is not None else 1
        ),
    }

    return {
        "feedback": feedback,
        "resample": resample,
        "cand1_identical_across_arms": cand1_identical,
        "cand1_max_abs_diff": cand1_max_abs_diff,
        "candidate_paths": {name: str(path) for name, path in paths.items()},
        "candidate_sha256": {
            name: file_sha256(str(path)) for name, path in paths.items()
        },
        "num_source_images": len(images),
        "latency_preprocess_sec": latency_preprocess_sec,
        "latency_postprocess_sec": latency_postprocess_sec,
    }


def main():

    args = parse_args()

    # Before any CUDA work (S7).
    determinism = enable_determinism()

    from src.config import load_config
    from src.dataset import (
        get_decoded_source_images,
        prepare_example,
    )
    from src.logger import (
        ExperimentLogger,
        load_completed_run_keys,
        make_run_key,
    )
    from src.step1x_runner import Step1XRunner

    base_config = load_config(args.config)
    configs = build_configs(base_config, args, determinism)
    condition = configs["feedback"]["experiment_name"]

    # =========================================================
    # Examples and thought cache
    # =========================================================

    examples = load_rq2_examples(args.splits, args.split, args.n_examples)

    uids = [
        prepare_example(examples[idx])["example_uid"]
        for idx in range(len(examples))
    ]

    thought_cache, cache_sha256 = load_thought_cache(
        args.thought_cache,
        expected_uids=uids,
        base_seed=args.seed,
        model_id=base_config["model"]["model_id"],
    )

    # =========================================================
    # Log and resume
    # =========================================================

    log_root = Path(args.log_dir)
    log_path = log_root / f"{condition}_{args.split}_seed{args.seed}.jsonl"
    logger = ExperimentLogger(str(log_path))
    completed = load_completed_run_keys(str(log_path))

    pending = [
        uid for uid in uids
        if make_run_key(uid, condition, args.seed, args.split) not in completed
    ]

    print("=" * 60)
    print(f"RQ2.3 | split {args.split} | seed {args.seed} | examples {len(uids)}")
    print(f"  {condition} -> {log_path}")
    print(f"  thought cache {args.thought_cache}")
    print(f"Pending examples: {len(pending)}")
    print("=" * 60)

    if not pending:
        print("Nothing to do; every example is complete.")
        return

    # =========================================================
    # Model
    # =========================================================

    runner = Step1XRunner(base_config)
    fix_text_encoder_offload(runner.pipe)
    # Same tokenizer as the thought-cache token counts.
    tokenizer = runner.pipe.processor.tokenizer
    env = collect_env()

    # Warm-up: one unlogged feedback call (generation and reflection).
    first = prepare_example(examples[0])
    runner.config = configs["feedback"]

    runner.run(
        images=get_decoded_source_images(examples[0]),
        instruction=thought_cache[first["example_uid"]]["rewrite"],
        example_uid=first["example_uid"],
        base_seed=args.seed,
    )

    # =========================================================
    # Examples
    # =========================================================

    success_count = 0
    failure_count = 0

    for idx in range(len(examples)):

        raw_example = examples[idx]
        example = prepare_example(raw_example)
        uid = example["example_uid"]

        if uid not in pending:
            continue

        print(f"\n[{idx + 1}/{len(uids)}] {uid} | seed {args.seed}")

        cached_thought = thought_cache[uid]
        prompt = cached_thought["rewrite"]

        out_paths = {
            name: Path(args.output_dir) / name / args.split
            / f"seed_{args.seed}" / f"{uid}.png"
            for name in (*CANDIDATE_DIRS, "feedback_cand1")
        }

        for name in CANDIDATE_DIRS:
            out_paths[name].parent.mkdir(parents=True, exist_ok=True)

        identity = {
            "example_uid": uid,
            "example_id": example["id"],
            "category": example["category"],
            "reasoning_dimension": example["reasoning_dimension"],
            "knowledge_type": example["knowledge_type"],
            "instruction": example["instruction"],
            "model_id": base_config["model"]["model_id"],
            "condition": condition,
            "split_name": args.split,
            "base_seed": args.seed,
            "rq2_subquestion": "2.3",
        }

        try:

            arms = run_example(
                runner=runner,
                configs=configs,
                raw_example=raw_example,
                example=example,
                prompt=prompt,
                args=args,
                out_paths=out_paths,
                tokenizer=tokenizer,
            )

            record = {
                **identity,
                "status": "success",

                "true_cfg_scale": float(args.true_cfg_scale),
                "num_inference_steps": args.num_inference_steps,
                "thinking_condition": "cached",

                # The exact string the pipeline received in every run.
                "prompt_used": prompt,
                "thought_cache_path": args.thought_cache,
                "thought_cache_sha256": cache_sha256,
                "thinking_seed": cached_thought.get("thinking_seed"),
                "thinking_token_count": cached_thought["thinking_token_count"],

                # One thinking pass charged to each arm (S4).
                "latency_thinking_sec": cached_thought["latency_thinking_sec"],
                "latency_thinking_charged_from_cache": True,
                "peak_vram_thinking_gb": cached_thought.get("peak_vram_thinking_gb"),

                **arms,

                "generation_configs": {
                    name: config["generation"]
                    for name, config in configs.items()
                },
                "unconditional_branch": UNCONDITIONAL_BRANCH,
                "offload_mode": offload_mode(base_config),
                "determinism": determinism,
                "env": env,
            }

            logger.log(record)
            success_count += 1

            fb = arms["feedback"]
            print(
                f"Feedback: {fb['attempt_count']} candidate(s) "
                f"[{', '.join(fb['reflection_tags'])}] "
                f"{fb['latency_pipeline_sec']:.1f}s | "
                f"plain {arms['resample']['cand1_latency_pipeline_sec']:.1f}s | "
                f"redraw {arms['resample']['redraw']} | "
                f"cand1 shared {arms['cand1_identical_across_arms']}"
            )

        except Exception as exc:

            failure_count += 1

            print(f"FAILED: {uid}")
            print(repr(exc))

            logger.log({
                **identity,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": repr(exc),
            })

    print()
    print("=" * 60)
    print(f"Successful: {success_count}")
    print(f"Failed:     {failure_count}")
    print(f"Skipped:    {len(uids) - len(pending)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# src/rq2/run_rq2.py

"""
Step1X RQ2 generation driver (RQ2.1; reused by RQ2.2).

One process loads Step1X once and runs the grid

    thinking mode  x  true_cfg_scale

at one step count over one frozen split. Every grid point is its own
condition, with its own output directory and JSONL log, so resume works
per condition exactly as in the RQ1 runner.

Replay (S4): for thinking "cached", the rewrite cached by
src.rq2.build_thought_cache is passed to Step1XRunner as the instruction
with the pipeline's thinking off. Every guidance value therefore sees the
identical prompt. Step1XRunner passes its instruction straight to the
pipeline as `prompt` (step1x_runner.py:405-414).

Same starting noise: Step1XRunner builds a fresh CUDA generator seeded
from (base seed, example UID, "diffusion_0") on every call
(step1x_runner.py:385-389). That does not depend on guidance, steps or
prompt, so the noise is identical across the whole grid.

The RQ1 files are used unchanged. Step1XRunner reads its generation flags
from self.config["generation"] on every run() call (step1x_runner.py:308),
so the driver swaps runner.config per grid point instead of reloading the
model.
"""

import argparse
import copy
import statistics
import time
from pathlib import Path

from src.rq2.common import (
    THINKING_MODES,
    UNCONDITIONAL_BRANCH,
    collect_env,
    condition_name,
    enable_determinism,
    fix_text_encoder_offload,
    load_rq2_examples,
    load_thought_cache,
    offload_mode,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Step1X RQ2 generations."
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--splits",
        default="data/splits.json",
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["k_dev", "k_small", "k_test"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--subquestion",
        required=True,
        choices=["2.1", "2.2"],
    )

    parser.add_argument(
        "--thinking",
        nargs="+",
        required=True,
        choices=THINKING_MODES,
    )

    parser.add_argument(
        "--true_cfg_scale",
        nargs="+",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--thought_cache",
        default=None,
        help="Required when --thinking includes 'cached'.",
    )

    parser.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Stratified subset (smoke / S7 only).",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to output_dir in the config.",
    )

    parser.add_argument(
        "--log_dir",
        default=None,
        help="Defaults to the directory of log_file in the config.",
    )

    parser.add_argument(
        "--run_label",
        default=None,
        help="Suffix for repeat runs (e.g. the S7 check).",
    )

    parser.add_argument(
        "--timing_repeats",
        type=int,
        default=1,
        help="Extra identical pipeline calls per row, timing only.",
    )

    parser.add_argument(
        "--allow_k_test",
        action="store_true",
        help="K-test is used once, for the frozen RQ2.2 config.",
    )

    args = parser.parse_args()

    if args.split == "k_test" and not args.allow_k_test:
        parser.error(
            "K-test is only used for the frozen RQ2.2 "
            "configuration; pass --allow_k_test for that run."
        )

    if "cached" in args.thinking and not args.thought_cache:
        parser.error(
            "--thinking cached requires --thought_cache."
        )

    if args.thought_cache and "cached" not in args.thinking:
        parser.error(
            "--thought_cache is only used with --thinking cached."
        )

    if args.timing_repeats < 1:
        parser.error("--timing_repeats must be >= 1")

    return args


def build_conditions(
    base_config,
    args,
    output_root: Path,
    log_root: Path,
    determinism,
):
    """
    One validated config per (thinking, true_cfg_scale) grid point.
    """

    from src.config import validate_config

    conditions = {}

    for thinking in args.thinking:
        for cfg_scale in args.true_cfg_scale:

            config = copy.deepcopy(base_config)
            generation = config["generation"]

            # Every dial is set explicitly per grid point (rule 3).
            generation["true_cfg_scale"] = float(cfg_scale)
            generation["num_inference_steps"] = args.num_inference_steps
            generation["enable_thinking_mode"] = thinking == "live"

            name = condition_name(
                base_config["experiment_name"],
                thinking,
                cfg_scale,
                args.num_inference_steps,
                args.run_label,
            )

            config["experiment_name"] = name
            config["output_dir"] = str(output_root / name)
            config["log_file"] = str(log_root / f"{name}.jsonl")

            config["rq2"] = {
                "subquestion": args.subquestion,
                "thinking_condition": thinking,
                "thought_cache":
                    args.thought_cache if thinking == "cached" else None,
                "run_label": args.run_label,
                "timing_repeats": args.timing_repeats,
                "determinism": determinism,
            }

            # Shared RQ1 rules: reflection off => max_try_cnt == 1.
            validate_config(config)

            conditions[(thinking, float(cfg_scale))] = config

    return conditions


def build_rq2_record(
    *,
    example,
    num_source_images: int,
    config,
    args,
    thinking: str,
    cached_thought,
    cache_sha256,
    prompt: str,
    result,
    output_path: Path,
    latency_preprocess_sec: float,
    latency_postprocess_sec: float,
    latency_end_to_end_sec: float,
    pipeline_repeats,
    env,
):
    """
    Shared RQ1 generation record plus the RQ2 fields.
    """

    from src.logger import build_generation_record

    generation = config["generation"]

    # =========================================================
    # Thought and thinking latency
    # =========================================================

    if thinking == "cached":
        reformat_prompt = cached_thought["rewrite"]
        thinking_token_count = cached_thought["thinking_token_count"]

        # One thinking pass charged per generation (S4).
        latency_thinking_sec = cached_thought["latency_thinking_sec"]

    elif thinking == "live":
        reformat_prompt = result["reformat_prompt"]
        thinking_token_count = result["thinking_token_count"]

        # Inside the pipeline call; not separable without editing Step1X.
        latency_thinking_sec = None

    else:
        reformat_prompt = None
        thinking_token_count = 0
        latency_thinking_sec = None

    # The pipeline call is pure generation only when it contains no
    # thinking (reflection is always off in RQ2.1/2.2).
    latency_generate_sec = (
        result["latency_pipeline_sec"]
        if thinking != "live"
        else None
    )

    latency_pipeline_median_sec = statistics.median(
        pipeline_repeats
    )

    attempt_count = result["attempt_count"] or 1

    record = build_generation_record(
        example_uid=example["example_uid"],
        example_id=example["id"],
        category=example["category"],
        reasoning_dimension=example["reasoning_dimension"],
        knowledge_type=example["knowledge_type"],

        instruction=example["instruction"],
        num_source_images=num_source_images,

        model_id=config["model"]["model_id"],
        condition=config["experiment_name"],
        split_name=args.split,
        full_config=config,

        base_seed=args.seed,
        stage_seeds=result["stage_seeds"],

        num_inference_steps=generation["num_inference_steps"],
        true_cfg_scale=generation["true_cfg_scale"],

        # Experimental condition, not the pipeline flag (see
        # pipeline_enable_thinking_mode below).
        thinking_enabled=thinking != "off",
        reflection_enabled=generation["enable_reflection_mode"],
        max_try_cnt=generation["max_try_cnt"],

        output_path=str(output_path),
        candidate_paths=[str(output_path)],
        selected_candidate_index=result["selected_candidate_index"],
        attempt_count=result["attempt_count"],
        retry_triggered=result["retry_triggered"],
        stop_reason=result["stop_reason"],
        selected_is_last_candidate=result["selected_is_last_candidate"],

        reformat_prompt=reformat_prompt,
        think_info=result["think_info"],
        thinking_token_count=thinking_token_count,
        best_info=result["best_info"],
        reflection_info=result["reflection_info"],
        reflection_token_count=None,

        latency_end_to_end_sec=latency_end_to_end_sec,
        latency_pipeline_sec=result["latency_pipeline_sec"],
        latency_preprocess_sec=latency_preprocess_sec,
        latency_thinking_sec=latency_thinking_sec,
        latency_diffusion_attempts_sec=(
            [latency_generate_sec]
            if latency_generate_sec is not None
            else []
        ),
        latency_reflection_rounds_sec=[],
        latency_postprocess_sec=latency_postprocess_sec,
        latency_offload_sec=None,

        peak_vram_gb=result["peak_vram_gb"],
    )

    record.update({
        "rq2_subquestion": args.subquestion,
        "thinking_condition": thinking,
        "pipeline_enable_thinking_mode":
            generation["enable_thinking_mode"],

        # The exact string the pipeline received.
        "prompt_used": prompt,

        "thought_cache_path":
            args.thought_cache if thinking == "cached" else None,
        "thought_cache_sha256": cache_sha256,

        "latency_thinking_charged_from_cache": thinking == "cached",
        "latency_generate_sec": latency_generate_sec,
        "latency_pipeline_repeats_sec": pipeline_repeats,
        "latency_pipeline_median_sec": latency_pipeline_median_sec,

        # Reconstructed stage total (Shared efficiency). Kept separate
        # from latency_end_to_end_sec, which is measured directly and,
        # for replayed rows, contains no thinking.
        "latency_reconstructed_total_sec": (
            latency_preprocess_sec
            + (latency_thinking_sec or 0.0)
            + latency_pipeline_median_sec
            + latency_postprocess_sec
        ),

        "peak_vram_thinking_gb": (
            cached_thought.get("peak_vram_thinking_gb")
            if cached_thought is not None
            else None
        ),

        # Transformer forward passes: two per step with true CFG.
        "generator_calls": (
            generation["num_inference_steps"]
            * (2 if generation["true_cfg_scale"] > 1 else 1)
            * attempt_count
        ),

        "unconditional_branch": UNCONDITIONAL_BRANCH,
        "offload_mode": offload_mode(config),
        "run_label": args.run_label,
        "env": env,
    })

    return record


def main():

    args = parse_args()

    # Before any CUDA work (S7).
    determinism = enable_determinism()

    from src.config import (
        load_config,
        print_config_summary,
    )
    from src.dataset import (
        get_decoded_source_images,
        prepare_example,
    )
    from src.logger import (
        ExperimentLogger,
        load_completed_run_keys,
        log_failure,
        make_run_key,
    )
    from src.seeding import get_rq1_stage_seeds
    from src.step1x_runner import Step1XRunner

    base_config = load_config(args.config)

    if base_config["generation"]["enable_reflection_mode"]:
        raise SystemExit(
            "RQ2.1/2.2 run with reflection off; use a config with "
            "enable_reflection_mode: false."
        )

    output_root = Path(
        args.output_dir or base_config["output_dir"]
    )

    log_root = Path(
        args.log_dir or Path(base_config["log_file"]).parent
    )

    conditions = build_conditions(
        base_config,
        args,
        output_root,
        log_root,
        determinism,
    )

    # =========================================================
    # Examples and thought cache
    # =========================================================

    examples = load_rq2_examples(
        args.splits,
        args.split,
        args.n_examples,
    )

    uids = [
        prepare_example(examples[idx])["example_uid"]
        for idx in range(len(examples))
    ]

    thought_cache = {}
    cache_sha256 = None

    if args.thought_cache:
        thought_cache, cache_sha256 = load_thought_cache(
            args.thought_cache,
            expected_uids=uids,
            base_seed=args.seed,
            model_id=base_config["model"]["model_id"],
        )

    def prompt_for(thinking, example):
        if thinking == "cached":
            return thought_cache[example["example_uid"]]["rewrite"]

        return example["instruction"]

    # =========================================================
    # Logs and resume
    # =========================================================

    loggers = {}
    completed = {}
    log_paths = {}

    for key, config in conditions.items():

        log_path = (
            log_root
            / f"{config['experiment_name']}_{args.split}_seed{args.seed}.jsonl"
        )

        log_paths[key] = log_path
        loggers[key] = ExperimentLogger(str(log_path))
        completed[key] = load_completed_run_keys(str(log_path))

    pending = sum(
        make_run_key(uid, config["experiment_name"], args.seed, args.split)
        not in completed[key]
        for uid in uids
        for key, config in conditions.items()
    )

    print("=" * 60)
    print(f"RQ{args.subquestion} | split {args.split} "
          f"| seed {args.seed} | examples {len(uids)}")

    for key, config in conditions.items():
        print(f"  {config['experiment_name']} -> {log_paths[key]}")

    print(f"Pending generations: {pending}")
    print("=" * 60)

    if pending == 0:
        print("Nothing to do; every grid point is complete.")
        return

    print_config_summary(
        next(iter(conditions.values())),
        seed=args.seed,
    )

    # =========================================================
    # Model (loaded once for the whole grid)
    # =========================================================

    runner = Step1XRunner(base_config)
    fix_text_encoder_offload(runner.pipe)
    env = collect_env()

    # Warm-up: one unlogged call so the first timed row isn't inflated.
    first_key = next(iter(conditions))
    first_example = prepare_example(examples[0])

    runner.config = conditions[first_key]

    runner.run(
        images=get_decoded_source_images(examples[0]),
        instruction=prompt_for(first_key[0], first_example),
        example_uid=first_example["example_uid"],
        base_seed=args.seed,
    )

    # =========================================================
    # Grid
    # =========================================================

    success_count = 0
    failure_count = 0
    skipped_count = 0

    for idx in range(len(examples)):

        raw_example = examples[idx]
        example = prepare_example(raw_example)
        uid = example["example_uid"]

        for key, config in conditions.items():

            thinking, _ = key
            condition = config["experiment_name"]
            generation = config["generation"]

            run_key = make_run_key(
                uid,
                condition,
                args.seed,
                args.split,
            )

            if run_key in completed[key]:
                skipped_count += 1
                continue

            print(f"\n[{idx + 1}/{len(uids)}] {uid} | {condition}")

            runner.config = config

            cached_thought = (
                thought_cache[uid]
                if thinking == "cached"
                else None
            )

            prompt = prompt_for(thinking, example)

            stage_seeds = get_rq1_stage_seeds(
                base_seed=args.seed,
                example_uid=uid,
                max_try_cnt=generation["max_try_cnt"],
            )

            end_to_end_start = time.perf_counter()

            try:

                # ---------------------------------------------
                # Preprocess
                # ---------------------------------------------

                preprocess_start = time.perf_counter()

                images = get_decoded_source_images(raw_example)

                latency_preprocess_sec = (
                    time.perf_counter() - preprocess_start
                )

                # ---------------------------------------------
                # Generate
                # ---------------------------------------------

                result = runner.run(
                    images=images,
                    instruction=prompt,
                    example_uid=uid,
                    base_seed=args.seed,
                )

                # ---------------------------------------------
                # Postprocess
                # ---------------------------------------------

                postprocess_start = time.perf_counter()

                output_dir = (
                    Path(config["output_dir"])
                    / args.split
                    / f"seed_{args.seed}"
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                output_path = output_dir / f"{uid}.png"

                result["final_image"].save(output_path)

                latency_postprocess_sec = (
                    time.perf_counter() - postprocess_start
                )

                latency_end_to_end_sec = (
                    time.perf_counter() - end_to_end_start
                )

                # ---------------------------------------------
                # Timing repeats (images identical, not saved)
                # ---------------------------------------------

                pipeline_repeats = [result["latency_pipeline_sec"]]

                for _ in range(args.timing_repeats - 1):
                    pipeline_repeats.append(
                        runner.run(
                            images=images,
                            instruction=prompt,
                            example_uid=uid,
                            base_seed=args.seed,
                        )["latency_pipeline_sec"]
                    )

                record = build_rq2_record(
                    example=example,
                    num_source_images=len(images),
                    config=config,
                    args=args,
                    thinking=thinking,
                    cached_thought=cached_thought,
                    cache_sha256=(
                        cache_sha256 if thinking == "cached" else None
                    ),
                    prompt=prompt,
                    result=result,
                    output_path=output_path,
                    latency_preprocess_sec=latency_preprocess_sec,
                    latency_postprocess_sec=latency_postprocess_sec,
                    latency_end_to_end_sec=latency_end_to_end_sec,
                    pipeline_repeats=pipeline_repeats,
                    env=env,
                )

                loggers[key].log(record)
                completed[key].add(run_key)
                success_count += 1

                print(
                    f"Saved: {output_path} "
                    f"| pipeline {result['latency_pipeline_sec']:.2f}s "
                    f"| peak VRAM {result['peak_vram_gb']:.1f} GB"
                )

            except Exception as exc:

                failure_count += 1

                print(f"FAILED: {uid} | {condition}")
                print(repr(exc))

                log_failure(
                    loggers[key],
                    example_uid=uid,
                    example_id=example["id"],
                    category=example["category"],
                    reasoning_dimension=example["reasoning_dimension"],
                    knowledge_type=example["knowledge_type"],
                    instruction=example["instruction"],
                    num_source_images=example["num_source_images"],
                    model_id=config["model"]["model_id"],
                    condition=condition,
                    split_name=args.split,
                    full_config=config,
                    base_seed=args.seed,
                    stage_seeds=stage_seeds,
                    num_inference_steps=generation["num_inference_steps"],
                    true_cfg_scale=generation["true_cfg_scale"],
                    thinking_enabled=thinking != "off",
                    reflection_enabled=generation["enable_reflection_mode"],
                    max_try_cnt=generation["max_try_cnt"],
                    error=exc,
                    latency_end_to_end_sec=(
                        time.perf_counter() - end_to_end_start
                    ),
                )

    # =========================================================
    # Summary
    # =========================================================

    print()
    print("=" * 60)
    print(f"Successful: {success_count}")
    print(f"Failed:     {failure_count}")
    print(f"Skipped:    {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()

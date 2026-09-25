# src/run_experiment.py

import argparse
import time
from pathlib import Path

from src.config import (
    load_config,
    print_config_summary,
)

from src.dataset import (
    load_kris_bench,
    select_split_examples,
    get_decoded_source_images,
    prepare_example,
)

from src.logger import (
    ExperimentLogger,
    build_generation_record,
    load_completed_run_keys,
    make_run_key,
    log_failure,
)

from src.seeding import (
    get_rq1_stage_seeds,
)

from src.step1x_runner import (
    Step1XRunner,
)


def build_log_path(
    base_log_file: str,
    split_name: str,
    seed: int,
) -> Path:
    """
    Create a separate log file for each split and seed.

    Example:
        logs/rq1_thinking.jsonl
            ->
        logs/rq1_thinking_k_test_seed42.jsonl
    """

    path = Path(base_log_file)

    suffix = path.suffix or ".jsonl"

    filename = (
        f"{path.stem}_"
        f"{split_name}_"
        f"seed{seed}"
        f"{suffix}"
    )

    return path.parent / filename


def save_candidate_images(
    candidate_images,
    output_dir: Path,
    example_uid: str,
):
    """
    Save every candidate image returned by Step1X.

    This is particularly important for the RQ1.2
    correction condition.

    Note:
    The exact Step1X correction candidate semantics
    will still need to be verified before the final
    RQ1.2 run.
    """

    candidate_paths = []

    if candidate_images is None:
        return candidate_paths

    for candidate_idx, image in enumerate(
        candidate_images
    ):
        candidate_path = (
            output_dir
            / (
                f"{example_uid}_"
                f"candidate{candidate_idx}.png"
            )
        )

        image.save(candidate_path)

        candidate_paths.append(
            str(candidate_path)
        )

    return candidate_paths


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run Step1X RQ1 image-editing experiments."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to RQ1 YAML config.",
    )

    parser.add_argument(
        "--splits",
        default="data/splits.json",
        help="Path to frozen splits.json.",
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=[
            "k_dev",
            "k_test",
            "k_small",
        ],
        help=(
            "Dataset split to run. "
            "RQ1 final experiments use k_test "
            "and k_small."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Base seed. "
            "Use 42 for K-test. "
            "Use 42, 1234 and 7 for "
            "three-seed K-small runs."
        ),
    )

    args = parser.parse_args()

    # =========================================================
    # Config
    # =========================================================

    config = load_config(
        args.config
    )

    print_config_summary(
        config,
        seed=args.seed,
    )

    condition = config[
        "experiment_name"
    ]

    model_id = config[
        "model"
    ][
        "model_id"
    ]

    generation_cfg = config[
        "generation"
    ]

    # =========================================================
    # Dataset
    # =========================================================

    print(
        f"Loading KRIS-Bench..."
    )

    dataset = load_kris_bench()

    examples = select_split_examples(
        dataset=dataset,
        splits_path=args.splits,
        split_name=args.split,
    )

    total = len(examples)

    print(
        f"Split: {args.split}"
    )

    print(
        f"Examples: {total}"
    )

    print(
        f"Base seed: {args.seed}"
    )

    # =========================================================
    # Output directory
    # =========================================================

    output_dir = (
        Path(
            config["output_dir"]
        )
        / args.split
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    # =========================================================
    # Log file
    # =========================================================

    log_file = build_log_path(
        base_log_file=config["log_file"],
        split_name=args.split,
        seed=args.seed,
    )

    logger = ExperimentLogger(
        str(log_file)
    )

    completed = load_completed_run_keys(
        str(log_file)
    )

    print(
        f"Log file: {log_file}"
    )

    print(
        f"Already completed: "
        f"{len(completed)}"
    )

    # =========================================================
    # Model
    # =========================================================

    runner = Step1XRunner(
        config
    )

    # =========================================================
    # Experiment
    # =========================================================

    success_count = 0
    failure_count = 0
    skipped_count = 0

    for idx in range(total):

        raw_example = examples[idx]

        example = prepare_example(
            raw_example
        )

        example_uid = example[
            "example_uid"
        ]

        run_key = make_run_key(
            example_uid=example_uid,
            condition=condition,
            base_seed=args.seed,
            split_name=args.split,
        )

        # -----------------------------------------------------
        # Resume support
        # -----------------------------------------------------

        if run_key in completed:

            print(
                f"[{idx + 1}/{total}] "
                f"Skipping completed: "
                f"{example_uid}"
            )

            skipped_count += 1

            continue

        print()

        print(
            f"[{idx + 1}/{total}] "
            f"{example_uid}"
        )

        print(
            f"Category: "
            f"{example['category']}"
        )

        print(
            f"Knowledge type: "
            f"{example['knowledge_type']}"
        )

        print(
            f"Instruction: "
            f"{example['instruction']}"
        )

        # Stage seeds are derived here as well so that
        # a failure can still be logged with the intended
        # deterministic seeds.
        stage_seeds = get_rq1_stage_seeds(
            base_seed=args.seed,
            example_uid=example_uid,
            max_try_cnt=generation_cfg[
                "max_try_cnt"
            ],
        )

        try:

            # =================================================
            # Preprocessing
            # =================================================

            preprocess_start = (
                time.perf_counter()
            )

            images = (
                get_decoded_source_images(
                    raw_example
                )
            )

            latency_preprocess_sec = (
                time.perf_counter()
                - preprocess_start
            )

            if len(images) != 1:
                raise RuntimeError(
                    f"RQ1 expected exactly one "
                    f"source image, got "
                    f"{len(images)}."
                )

            # =================================================
            # Inference
            # =================================================

            result = runner.run(
                images=images,
                instruction=example[
                    "instruction"
                ],
                example_uid=example_uid,
                base_seed=args.seed,
            )

            # =================================================
            # Save final + candidates
            # =================================================

            postprocess_start = (
                time.perf_counter()
            )

            output_path = (
                output_dir
                / f"{example_uid}.png"
            )

            result[
                "final_image"
            ].save(
                output_path
            )

            candidate_paths = (
                save_candidate_images(
                    candidate_images=result.get(
                        "candidate_images"
                    ),
                    output_dir=output_dir,
                    example_uid=example_uid,
                )
            )

            latency_postprocess_sec = (
                time.perf_counter()
                - postprocess_start
            )

            # If Step1X returned only one image and that is
            # the final image, this is safe to record.
            selected_candidate_index = (
                result.get(
                    "selected_candidate_index"
                )
            )

            if (
                selected_candidate_index
                is None
                and len(candidate_paths) == 1
            ):
                selected_candidate_index = 0

            attempt_count = result.get(
                "attempt_count"
            )

            retry_triggered = result.get(
                "retry_triggered"
            )

            # =================================================
            # Build log record
            # =================================================

            record = (
                build_generation_record(
                    # Identity
                    example_uid=
                        example_uid,

                    example_id=
                        example["id"],

                    category=
                        example["category"],

                    reasoning_dimension=
                        example[
                            "reasoning_dimension"
                        ],

                    knowledge_type=
                        example[
                            "knowledge_type"
                        ],

                    # Input
                    instruction=
                        example[
                            "instruction"
                        ],

                    num_source_images=
                        len(images),

                    # Experiment
                    model_id=
                        model_id,

                    condition=
                        condition,

                    split_name=
                        args.split,

                    # Randomness
                    base_seed=
                        args.seed,

                    stage_seeds=
                        result[
                            "stage_seeds"
                        ],

                    # Generation
                    num_inference_steps=
                        generation_cfg[
                            "num_inference_steps"
                        ],

                    true_cfg_scale=
                        generation_cfg[
                            "true_cfg_scale"
                        ],

                    thinking_enabled=
                        generation_cfg[
                            "enable_thinking_mode"
                        ],

                    reflection_enabled=
                        generation_cfg[
                            "enable_reflection_mode"
                        ],

                    max_try_cnt=
                        generation_cfg[
                            "max_try_cnt"
                        ],

                    # Main output
                    output_path=
                        str(output_path),

                    # Candidates
                    candidate_paths=
                        candidate_paths,

                    selected_candidate_index=
                        selected_candidate_index,

                    attempt_count=
                        attempt_count,

                    retry_triggered=
                        retry_triggered,

                    # Reasoning
                    reformat_prompt=
                        result.get(
                            "reformat_prompt"
                        ),

                    think_info=
                        result.get(
                            "think_info"
                        ),

                    best_info=
                        result.get(
                            "best_info"
                        ),

                    reflection_info=
                        result.get(
                            "reflection_info"
                        ),

                    # Timing
                    latency_total_sec=
                        result.get(
                            "latency_total_sec"
                        ),

                    latency_preprocess_sec=
                        latency_preprocess_sec,

                    latency_thinking_sec=
                        result.get(
                            "latency_thinking_sec"
                        ),

                    latency_diffusion_sec=
                        result.get(
                            "latency_diffusion_sec"
                        ),

                    latency_reflection_sec=
                        result.get(
                            "latency_reflection_sec"
                        ),

                    latency_postprocess_sec=
                        latency_postprocess_sec,

                    # Resources
                    peak_vram_gb=
                        result.get(
                            "peak_vram_gb"
                        ),

                    # Status
                    status="success",
                    error=None,
                )
            )

            logger.log(
                record
            )

            success_count += 1

            print(
                f"Saved: "
                f"{output_path}"
            )

            print(
                f"Latency: "
                f"{result['latency_total_sec']:.2f}s"
            )

            if result.get(
                "reformat_prompt"
            ):
                print(
                    "Rewritten prompt:",
                    result[
                        "reformat_prompt"
                    ],
                )

            if attempt_count is not None:
                print(
                    f"Attempts: "
                    f"{attempt_count}"
                )

        except Exception as exc:

            failure_count += 1

            print(
                f"FAILED: "
                f"{example_uid}"
            )

            print(
                repr(exc)
            )

            log_failure(
                logger=logger,

                # Identity
                example_uid=
                    example_uid,

                example_id=
                    example["id"],

                category=
                    example["category"],

                reasoning_dimension=
                    example[
                        "reasoning_dimension"
                    ],

                knowledge_type=
                    example[
                        "knowledge_type"
                    ],

                # Input
                instruction=
                    example[
                        "instruction"
                    ],

                num_source_images=
                    example[
                        "num_source_images"
                    ],

                # Experiment
                model_id=
                    model_id,

                condition=
                    condition,

                split_name=
                    args.split,

                # Randomness
                base_seed=
                    args.seed,

                stage_seeds=
                    stage_seeds,

                # Generation
                num_inference_steps=
                    generation_cfg[
                        "num_inference_steps"
                    ],

                true_cfg_scale=
                    generation_cfg[
                        "true_cfg_scale"
                    ],

                thinking_enabled=
                    generation_cfg[
                        "enable_thinking_mode"
                    ],

                reflection_enabled=
                    generation_cfg[
                        "enable_reflection_mode"
                    ],

                max_try_cnt=
                    generation_cfg[
                        "max_try_cnt"
                    ],

                error=exc,
            )

    # =========================================================
    # Summary
    # =========================================================

    print()
    print("=" * 60)

    print(
        f"Experiment finished: "
        f"{condition}"
    )

    print(
        f"Split: "
        f"{args.split}"
    )

    print(
        f"Seed: "
        f"{args.seed}"
    )

    print(
        f"Successful: "
        f"{success_count}"
    )

    print(
        f"Failed: "
        f"{failure_count}"
    )

    print(
        f"Skipped: "
        f"{skipped_count}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
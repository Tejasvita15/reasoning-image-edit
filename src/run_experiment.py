# src/run_experiment.py

import argparse
from pathlib import Path

from src.config import (
    load_config,
    print_config_summary,
)

from src.dataset import (
    load_kris_bench,
    select_evaluation_examples,
    get_decoded_source_images,
)

from src.taxonomy import (
    get_reasoning_dimension,
    get_knowledge_type,
)

from src.logger import (
    ExperimentLogger,
    build_generation_record,
    load_logged_example_uids,
    log_failure,
)

from src.step1x_runner import Step1XRunner


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--evaluation-set",
        default="data/evaluation_set.json",
    )

    args = parser.parse_args()

    # -----------------------------
    # Config
    # -----------------------------

    config = load_config(
        args.config
    )

    print_config_summary(
        config
    )

    # -----------------------------
    # Dataset
    # -----------------------------

    dataset = load_kris_bench()

    examples = (
        select_evaluation_examples(
            dataset,
            args.evaluation_set,
        )
    )

    print(
        f"Evaluation examples: "
        f"{len(examples)}"
    )

    # -----------------------------
    # Output + logging
    # -----------------------------

    output_dir = Path(
        config["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = ExperimentLogger(
        config["log_file"]
    )

    completed = (
        load_logged_example_uids(
            config["log_file"]
        )
    )

    print(
        f"Already completed: "
        f"{len(completed)}"
    )

    # -----------------------------
    # Model
    # -----------------------------

    runner = Step1XRunner(
        config
    )

    # -----------------------------
    # Experiment loop
    # -----------------------------

    total = len(examples)

    for idx in range(total):

        example = examples[idx]

        category = str(
            example["category"]
        )

        example_id = str(
            example["id"]
        )

        example_uid = (
            f"{category}__{example_id}"
        )

        if example_uid in completed:
            print(
                f"[{idx + 1}/{total}] "
                f"Skipping {example_uid}"
            )
            continue

        reasoning_dimension = (
            get_reasoning_dimension(
                category
            )
        )

        knowledge_type = (
            get_knowledge_type(
                category
            )
        )

        instruction = (
            example["instruction"]
        )

        print(
            f"\n[{idx + 1}/{total}] "
            f"{example_uid}"
        )

        print(
            f"Instruction: "
            f"{instruction}"
        )

        try:

            images = (
                get_decoded_source_images(
                    example
                )
            )

            if len(images) == 0:
                raise RuntimeError(
                    "No source image found."
                )

            result = runner.run(
                images=images,
                instruction=instruction,
                seed=config["seed"],
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

            record = (
                build_generation_record(
                    example_uid=
                        example_uid,

                    example_id=
                        example_id,

                    category=
                        category,

                    reasoning_dimension=
                        reasoning_dimension,

                    knowledge_type=
                        knowledge_type,

                    instruction=
                        instruction,

                    condition=
                        config[
                            "experiment_name"
                        ],

                    seed=
                        config["seed"],

                    num_inference_steps=
                        config[
                            "generation"
                        ][
                            "num_inference_steps"
                        ],

                    true_cfg_scale=
                        config[
                            "generation"
                        ][
                            "true_cfg_scale"
                        ],

                    thinking_enabled=
                        config[
                            "generation"
                        ][
                            "enable_thinking_mode"
                        ],

                    reflection_enabled=
                        config[
                            "generation"
                        ][
                            "enable_reflection_mode"
                        ],

                    output_path=
                        str(output_path),

                    latency_total_sec=
                        result[
                            "latency_total_sec"
                        ],

                    peak_vram_gb=
                        result[
                            "peak_vram_gb"
                        ],

                    reformat_prompt=
                        result[
                            "reformat_prompt"
                        ],

                    think_info=
                        result[
                            "think_info"
                        ],

                    best_info=
                        result[
                            "best_info"
                        ],

                    num_source_images=
                        len(images),
                )
            )

            logger.log(
                record
            )

            print(
                f"Saved: "
                f"{output_path}"
            )

            print(
                f"Latency: "
                f"{result['latency_total_sec']:.2f}s"
            )

        except Exception as exc:

            print(
                f"FAILED: "
                f"{example_uid}"
            )

            print(
                repr(exc)
            )

            log_failure(
                logger,

                example_uid=
                    example_uid,

                example_id=
                    example_id,

                category=
                    category,

                reasoning_dimension=
                    reasoning_dimension,

                knowledge_type=
                    knowledge_type,

                instruction=
                    instruction,

                condition=
                    config[
                        "experiment_name"
                    ],

                seed=
                    config["seed"],

                num_inference_steps=
                    config[
                        "generation"
                    ][
                        "num_inference_steps"
                    ],

                true_cfg_scale=
                    config[
                        "generation"
                    ][
                        "true_cfg_scale"
                    ],

                thinking_enabled=
                    config[
                        "generation"
                    ][
                        "enable_thinking_mode"
                    ],

                reflection_enabled=
                    config[
                        "generation"
                    ][
                        "enable_reflection_mode"
                    ],

                error=exc,
            )


if __name__ == "__main__":
    main()
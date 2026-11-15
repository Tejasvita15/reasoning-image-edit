# src/run_experiment.py



import argparse

import traceback

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

    Create a separate log file

    for each split and seed.



    Example:

        logs/rq1_thinking.jsonl

            ->

        logs/rq1_thinking_k_test_seed42.jsonl

    """



    path = Path(

        base_log_file

    )



    suffix = (

        path.suffix

        or ".jsonl"

    )



    filename = (

        f"{path.stem}_"

        f"{split_name}_"

        f"seed{seed}"

        f"{suffix}"

    )



    return (

        path.parent

        / filename

    )





def save_candidate_images(

    candidate_images,

    output_dir: Path,

    example_uid: str,

):

    """

    Save candidate images returned

    by Step1X.



    Important:

    For RQ1.2, exact candidate semantics

    must still be verified from the

    Step1X retry loop.

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



        image.save(

            candidate_path

        )



        candidate_paths.append(

            str(candidate_path)

        )



    return candidate_paths





def main():



    parser = argparse.ArgumentParser(

        description=(

            "Run Step1X RQ1 "

            "image-editing experiments."

        )

    )



    parser.add_argument(

        "--config",

        required=True,

        help=(

            "Path to RQ1 YAML config."

        ),

    )



    parser.add_argument(

        "--splits",

        default="data/splits.json",

        help=(

            "Path to frozen splits.json."

        ),

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

            "Dataset split to run."

        ),

    )



    parser.add_argument(

        "--seed",

        type=int,

        required=True,

        help=(

            "Base seed. "

            "Use 42 for single-seed runs. "

            "K-small additionally uses "

            "1234 and 7."

        ),

    )


    parser.add_argument(

        "--stop-on-failure",

        action="store_true",

        help=(

            "Stop immediately after the first failed example. "

            "Useful for debugging deployment/runtime errors."

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



    condition = (

        config[

            "experiment_name"

        ]

    )



    model_id = (

        config[

            "model"

        ][

            "model_id"

        ]

    )



    generation_cfg = (

        config[

            "generation"

        ]

    )



    # =========================================================

    # Dataset

    # =========================================================



    print(

        "Loading KRIS-Bench..."

    )



    dataset = (

        load_kris_bench()

    )



    examples = (

        select_split_examples(

            dataset=dataset,

            splits_path=args.splits,

            split_name=args.split,

        )

    )



    total = len(

        examples

    )



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

            config[

                "output_dir"

            ]

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



    log_file = (

        build_log_path(

            base_log_file=

                config[

                    "log_file"

                ],



            split_name=

                args.split,



            seed=

                args.seed,

        )

    )



    logger = (

        ExperimentLogger(

            str(log_file)

        )

    )



    completed = (

        load_completed_run_keys(

            str(log_file)

        )

    )



    print(

        f"Log file: "

        f"{log_file}"

    )



    print(

        f"Already completed: "

        f"{len(completed)}"

    )



    # =========================================================

    # Model

    # =========================================================



    runner = (

        Step1XRunner(

            config

        )

    )



    # =========================================================

    # Experiment counters

    # =========================================================



    success_count = 0

    failure_count = 0

    skipped_count = 0



    # =========================================================

    # Experiment loop

    # =========================================================



    for idx in range(total):



        raw_example = (

            examples[idx]

        )



        example = (

            prepare_example(

                raw_example

            )

        )



        example_uid = (

            example[

                "example_uid"

            ]

        )



        run_key = (

            make_run_key(

                example_uid=

                    example_uid,



                condition=

                    condition,



                base_seed=

                    args.seed,



                split_name=

                    args.split,

            )

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



        # -----------------------------------------------------

        # Precompute stage seeds

        # -----------------------------------------------------



        stage_seeds = (

            get_rq1_stage_seeds(

                base_seed=

                    args.seed,



                example_uid=

                    example_uid,



                max_try_cnt=

                    generation_cfg[

                        "max_try_cnt"

                    ],

            )

        )



        # True end-to-end timer.

        #

        # Starts before decoding and ends

        # after images have been saved.

        end_to_end_start = (

            time.perf_counter()

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

                    f"RQ1 expected exactly "

                    f"one source image, got "

                    f"{len(images)}."

                )



            # =================================================

            # Inference

            # =================================================



            result = (

                runner.run(

                    images=images,



                    instruction=

                        example[

                            "instruction"

                        ],



                    example_uid=

                        example_uid,



                    base_seed=

                        args.seed,

                )

            )



            # =================================================

            # Save final output

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



            # =================================================

            # Save candidates

            # =================================================



            candidate_paths = (

                save_candidate_images(

                    candidate_images=

                        result.get(

                            "candidate_images"

                        ),



                    output_dir=

                        output_dir,



                    example_uid=

                        example_uid,

                )

            )



            latency_postprocess_sec = (

                time.perf_counter()

                - postprocess_start

            )



            # =================================================

            # Direct end-to-end timing

            # =================================================



            latency_end_to_end_sec = (

                time.perf_counter()

                - end_to_end_start

            )



            # =================================================

            # Candidate metadata

            # =================================================



            selected_candidate_index = (

                result.get(

                    "selected_candidate_index"

                )

            )



            attempt_count = (

                result.get(

                    "attempt_count"

                )

            )



            retry_triggered = (

                result.get(

                    "retry_triggered"

                )

            )



            stop_reason = (

                result.get(

                    "stop_reason"

                )

            )



            selected_is_last_candidate = (

                result.get(

                    "selected_is_last_candidate"

                )

            )



            # For single-attempt Base / Thinking,

            # if Step1X exposes exactly one returned

            # image and no explicit index, we can

            # safely record candidate 0.

            if (

                selected_candidate_index

                is None

                and len(candidate_paths) == 1

                and generation_cfg[

                    "max_try_cnt"

                ] == 1

            ):

                selected_candidate_index = 0



            # =================================================

            # Build generation record

            # =================================================



            record = (

                build_generation_record(



                    # -----------------------------------------

                    # Identity

                    # -----------------------------------------



                    example_uid=

                        example_uid,



                    example_id=

                        example[

                            "id"

                        ],



                    category=

                        example[

                            "category"

                        ],



                    reasoning_dimension=

                        example[

                            "reasoning_dimension"

                        ],



                    knowledge_type=

                        example[

                            "knowledge_type"

                        ],



                    # -----------------------------------------

                    # Input

                    # -----------------------------------------



                    instruction=

                        example[

                            "instruction"

                        ],



                    num_source_images=

                        len(images),



                    # -----------------------------------------

                    # Experiment

                    # -----------------------------------------



                    model_id=

                        model_id,



                    condition=

                        condition,



                    split_name=

                        args.split,



                    full_config=

                        config,



                    # -----------------------------------------

                    # Randomness

                    # -----------------------------------------



                    base_seed=

                        args.seed,



                    stage_seeds=

                        result[

                            "stage_seeds"

                        ],



                    # -----------------------------------------

                    # Generation

                    # -----------------------------------------



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



                    # -----------------------------------------

                    # Main output

                    # -----------------------------------------



                    output_path=

                        str(output_path),



                    # -----------------------------------------

                    # Candidate / correction

                    # -----------------------------------------



                    candidate_paths=

                        candidate_paths,



                    selected_candidate_index=

                        selected_candidate_index,



                    attempt_count=

                        attempt_count,



                    retry_triggered=

                        retry_triggered,



                    stop_reason=

                        stop_reason,



                    selected_is_last_candidate=

                        selected_is_last_candidate,



                    # -----------------------------------------

                    # Reasoning

                    # -----------------------------------------



                    reformat_prompt=

                        result.get(

                            "reformat_prompt"

                        ),



                    think_info=

                        result.get(

                            "think_info"

                        ),



                    thinking_token_count=

                        result.get(

                            "thinking_token_count"

                        ),



                    best_info=

                        result.get(

                            "best_info"

                        ),



                    reflection_info=

                        result.get(

                            "reflection_info"

                        ),



                    reflection_token_count=

                        result.get(

                            "reflection_token_count"

                        ),



                    # -----------------------------------------

                    # Timing

                    # -----------------------------------------



                    latency_end_to_end_sec=

                        latency_end_to_end_sec,



                    latency_pipeline_sec=

                        result.get(

                            "latency_pipeline_sec"

                        ),



                    latency_preprocess_sec=

                        latency_preprocess_sec,



                    latency_thinking_sec=

                        result.get(

                            "latency_thinking_sec"

                        ),



                    latency_diffusion_attempts_sec=

                        result.get(

                            "latency_diffusion_attempts_sec"

                        ),



                    latency_reflection_rounds_sec=

                        result.get(

                            "latency_reflection_rounds_sec"

                        ),



                    latency_postprocess_sec=

                        latency_postprocess_sec,



                    latency_offload_sec=

                        result.get(

                            "latency_offload_sec"

                        ),



                    # -----------------------------------------

                    # Resources

                    # -----------------------------------------



                    peak_vram_gb=

                        result.get(

                            "peak_vram_gb"

                        ),



                    # -----------------------------------------

                    # Status

                    # -----------------------------------------



                    status=

                        "success",



                    error=

                        None,

                )

            )



            logger.log(

                record

            )



            # Update in-memory completed set as well.

            completed.add(

                run_key

            )



            success_count += 1



            # =================================================

            # Console output

            # =================================================



            print(

                f"Saved: "

                f"{output_path}"

            )



            print(

                f"Pipeline latency: "

                f"{result['latency_pipeline_sec']:.2f}s"

            )



            print(

                f"End-to-end latency: "

                f"{latency_end_to_end_sec:.2f}s"

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



            if (

                result.get(

                    "thinking_token_count"

                )

                is not None

            ):



                print(

                    f"Rewrite tokens: "

                    f"{result['thinking_token_count']}"

                )



            if attempt_count is not None:



                print(

                    f"Attempts: "

                    f"{attempt_count}"

                )



        except Exception as exc:



            failure_count += 1



            latency_end_to_end_sec = (

                time.perf_counter()

                - end_to_end_start

            )



            print(

                f"FAILED: "

                f"{example_uid}"

            )



            print(

                repr(exc)

            )


            traceback.print_exc()



            log_failure(

                logger=logger,



                # ---------------------------------------------

                # Identity

                # ---------------------------------------------



                example_uid=

                    example_uid,



                example_id=

                    example[

                        "id"

                    ],



                category=

                    example[

                        "category"

                    ],



                reasoning_dimension=

                    example[

                        "reasoning_dimension"

                    ],



                knowledge_type=

                    example[

                        "knowledge_type"

                    ],



                # ---------------------------------------------

                # Input

                # ---------------------------------------------



                instruction=

                    example[

                        "instruction"

                    ],



                num_source_images=

                    example[

                        "num_source_images"

                    ],



                # ---------------------------------------------

                # Experiment

                # ---------------------------------------------



                model_id=

                    model_id,



                condition=

                    condition,



                split_name=

                    args.split,



                full_config=

                    config,



                # ---------------------------------------------

                # Randomness

                # ---------------------------------------------



                base_seed=

                    args.seed,



                stage_seeds=

                    stage_seeds,



                # ---------------------------------------------

                # Generation

                # ---------------------------------------------



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



                # ---------------------------------------------

                # Failure

                # ---------------------------------------------



                error=

                    exc,



                latency_end_to_end_sec=

                    latency_end_to_end_sec,

            )


            if args.stop_on_failure:

                print(

                    "Stopping after first failure "

                    "(--stop-on-failure enabled)."

                )

                break



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
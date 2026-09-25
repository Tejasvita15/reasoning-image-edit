# src/step1x_runner.py

import time
from typing import Any, Dict

import torch
from diffusers import Step1XEditPipelineV1P2

from src.seeding import (
    get_rq1_stage_seeds,
    set_global_seed,
)


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class Step1XRunner:
    """
    Wrapper around Step1X-Edit-v1p2 for RQ1 experiments.

    Supports:
    - Base / direct editing
    - Thinking-only
    - Thinking + correction

    All generation flags are passed explicitly.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        self.config = config

        model_cfg = config["model"]

        dtype_name = model_cfg["dtype"]

        if dtype_name not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype: {dtype_name}"
            )

        dtype = DTYPE_MAP[dtype_name]

        print(
            f"Loading model: "
            f"{model_cfg['model_id']}"
        )

        self.pipe = (
            Step1XEditPipelineV1P2.from_pretrained(
                model_cfg["model_id"],
                torch_dtype=dtype,
            )
        )

        if model_cfg.get(
            "cpu_offload",
            False,
        ):
            self.pipe.enable_model_cpu_offload()

        else:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required when "
                    "cpu_offload=False."
                )

            self.pipe.to("cuda")

        print("Step1X loaded.")

    def run(
        self,
        images,
        instruction: str,
        example_uid: str,
        base_seed: int,
    ) -> Dict[str, Any]:
        """
        Run one RQ1 generation.

        Args:
            images:
                Decoded source images.

            instruction:
                Original KRIS-Bench instruction.

            example_uid:
                Stable category__id identifier.

            base_seed:
                Experiment-level seed, e.g.
                42, 1234 or 7.
        """

        generation_cfg = self.config[
            "generation"
        ]

        # -----------------------------------------------------
        # Validate input
        # -----------------------------------------------------

        if len(images) == 0:
            raise ValueError(
                f"{example_uid}: "
                "no source image found."
            )

        if len(images) != 1:
            raise ValueError(
                f"{example_uid}: RQ1 requires exactly "
                f"one source image, but got "
                f"{len(images)}."
            )

        image_input = images[0]

        # -----------------------------------------------------
        # Read experiment configuration
        # -----------------------------------------------------

        num_inference_steps = int(
            generation_cfg[
                "num_inference_steps"
            ]
        )

        true_cfg_scale = float(
            generation_cfg[
                "true_cfg_scale"
            ]
        )

        thinking_enabled = bool(
            generation_cfg[
                "enable_thinking_mode"
            ]
        )

        reflection_enabled = bool(
            generation_cfg[
                "enable_reflection_mode"
            ]
        )

        max_try_cnt = int(
            generation_cfg[
                "max_try_cnt"
            ]
        )

        # -----------------------------------------------------
        # Deterministic stage seeds
        # -----------------------------------------------------

        stage_seeds = get_rq1_stage_seeds(
            base_seed=base_seed,
            example_uid=example_uid,
            max_try_cnt=max_try_cnt,
        )

        # Step1X thinking uses the global torch RNG.
        #
        # Set it immediately before the pipeline call.
        # For Base this is also harmless and makes the
        # complete run deterministic.
        set_global_seed(
            stage_seeds["thinking"]
        )

        # Diffusion uses its own generator.
        generator = torch.Generator(
            device="cuda"
        ).manual_seed(
            stage_seeds["diffusion_0"]
        )

        # -----------------------------------------------------
        # GPU measurement setup
        # -----------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Important for correct wall-clock GPU timing.
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # -----------------------------------------------------
        # Step1X inference
        # -----------------------------------------------------

        result = self.pipe(
            image=image_input,
            prompt=instruction,

            # Always explicit.
            num_inference_steps=
                num_inference_steps,

            true_cfg_scale=
                true_cfg_scale,

            enable_thinking_mode=
                thinking_enabled,

            enable_reflection_mode=
                reflection_enabled,

            max_try_cnt=
                max_try_cnt,

            generator=
                generator,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latency_total_sec = (
            time.perf_counter()
            - start_time
        )

        # -----------------------------------------------------
        # GPU memory
        # -----------------------------------------------------

        peak_vram_gb = None

        if torch.cuda.is_available():
            peak_vram_gb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

        # -----------------------------------------------------
        # Outputs
        # -----------------------------------------------------

        result_images = getattr(
            result,
            "images",
            None,
        )

        if result_images is None:
            raise RuntimeError(
                f"{example_uid}: Step1X result "
                "does not contain 'images'."
            )

        result_images = list(
            result_images
        )

        if len(result_images) == 0:
            raise RuntimeError(
                f"{example_uid}: Step1X returned "
                "zero output images."
            )

        final_image = result_images[0]

        # -----------------------------------------------------
        # Extract optional Step1X metadata
        # -----------------------------------------------------

        reformat_prompt = getattr(
            result,
            "reformat_prompt",
            None,
        )

        think_info = getattr(
            result,
            "think_info",
            None,
        )

        best_info = getattr(
            result,
            "best_info",
            None,
        )

        # Keep these generic for now.
        #
        # We will populate more precise correction information
        # after inspecting the exact v1p2 pipeline return
        # structure.
        reflection_info = getattr(
            result,
            "reflection_info",
            None,
        )

        attempt_count = getattr(
            result,
            "attempt_count",
            None,
        )

        selected_candidate_index = getattr(
            result,
            "selected_candidate_index",
            None,
        )

        retry_triggered = None

        if attempt_count is not None:
            try:
                retry_triggered = (
                    int(attempt_count) > 1
                )
            except (TypeError, ValueError):
                retry_triggered = None

        # -----------------------------------------------------
        # Return standardized information
        # -----------------------------------------------------

        return {
            "final_image":
                final_image,

            "candidate_images":
                result_images,

            # Reasoning
            "reformat_prompt":
                reformat_prompt,

            "think_info":
                think_info,

            "best_info":
                best_info,

            "reflection_info":
                reflection_info,

            # Correction metadata
            "attempt_count":
                attempt_count,

            "selected_candidate_index":
                selected_candidate_index,

            "retry_triggered":
                retry_triggered,

            # Seeds
            "base_seed":
                base_seed,

            "stage_seeds":
                stage_seeds,

            # Timing
            "latency_total_sec":
                latency_total_sec,

            "latency_preprocess_sec":
                None,

            "latency_thinking_sec":
                None,

            "latency_diffusion_sec":
                None,

            "latency_reflection_sec":
                None,

            "latency_postprocess_sec":
                None,

            # Resources
            "peak_vram_gb":
                peak_vram_gb,
        }
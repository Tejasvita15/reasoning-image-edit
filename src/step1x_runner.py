# src/step1x_runner.py

import time
from typing import Any, Dict, Optional

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
    Step1X-Edit-v1p2 wrapper for RQ1 experiments.

    Supports:
    - direct editing
    - thinking-only
    - thinking + correction

    The Step1X library itself is left unchanged.
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
        model_id = model_cfg["model_id"]

        print(f"Loading model: {model_id}")

        self.pipe = Step1XEditPipelineV1P2.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        self.cpu_offload = bool(
            model_cfg.get(
                "cpu_offload",
                False,
            )
        )

        if self.cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required when "
                    "cpu_offload=False."
                )

            self.pipe.to("cuda")

        print("Step1X loaded.")

    @staticmethod
    def _sync_cuda() -> None:
        """
        Synchronize CUDA before/after wall-clock timing.
        """

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _count_text_tokens(
        tokenizer,
        text: Optional[str],
    ) -> Optional[int]:
        """
        Count tokens with the model tokenizer when available.
        """

        if not text:
            return 0

        if tokenizer is None:
            return None

        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            input_ids = encoded["input_ids"]

            if (
                len(input_ids) > 0
                and isinstance(
                    input_ids[0],
                    list,
                )
            ):
                input_ids = input_ids[0]

            return len(input_ids)

        except Exception:
            return None

    def _get_available_tokenizer(
        self,
    ):
        """
        Return the tokenizer actually exposed by Step1X.
        """

        processor = getattr(
            self.pipe,
            "processor",
            None,
        )

        if processor is not None:
            tokenizer = getattr(
                processor,
                "tokenizer",
                None,
            )

            if tokenizer is not None:
                return tokenizer

        for name in (
            "tokenizer",
            "tokenizer_2",
        ):
            tokenizer = getattr(
                self.pipe,
                name,
                None,
            )

            if tokenizer is not None:
                return tokenizer

        return None

    @staticmethod
    def _get_selected_candidate_index(
        best_info,
        think_info,
        num_candidates: int,
    ) -> int:
        """
        Reproduce Step1X-v1p2 native candidate selection.

        Step1X chooses the candidate maximizing:

            min(score1) * min(score2)

        Ties prefer a candidate containing <#Success>.
        If still tied, the later candidate wins.
        """

        if num_candidates <= 0:
            raise ValueError(
                "num_candidates must be positive."
            )

        if not best_info:
            return 0

        best_idx = 0
        max_score = -1
        best_has_success = False

        for i, info in enumerate(best_info):
            if i >= num_candidates:
                break

            try:
                s1_min = min(
                    info["score1"]["score"]
                )

                s2_min = min(
                    info["score2"]["score"]
                )

                current_score = (
                    s1_min * s2_min
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            current_think_str = ""

            if (
                think_info
                and i < len(think_info)
            ):
                current_think_str = str(
                    think_info[i]
                )

            current_has_success = (
                "<#Success>"
                in current_think_str
            )

            if current_score > max_score:
                best_idx = i
                max_score = current_score
                best_has_success = (
                    current_has_success
                )

            elif current_score == max_score:
                if (
                    current_has_success
                    and not best_has_success
                ):
                    best_idx = i
                    best_has_success = True

                elif (
                    current_has_success
                    == best_has_success
                ):
                    best_idx = i

        return best_idx

    @staticmethod
    def _get_stop_reason(
        think_info,
        retry_triggered: bool,
    ) -> str:
        """
        Interpret the final Step1X reflection result.

        This only logs what happened; it does not alter
        Step1X's native correction behavior.
        """

        if not think_info:
            return (
                "single_pass"
                if not retry_triggered
                else "max_attempts_or_unknown"
            )

        last_info = str(
            think_info[-1]
        )

        if "<#Success>" in last_info:
            return "reflection_success"

        if "<#Failed>" in last_info:
            return "reflection_failed"

        if "<#Reflection>" in last_info:
            return (
                "max_attempts_reflection_requested"
                if retry_triggered
                else "reflection_requested_retry"
            )

        return (
            "max_attempts_or_unknown"
            if retry_triggered
            else "single_pass"
        )

    def run(
        self,
        images,
        instruction: str,
        example_uid: str,
        base_seed: int,
    ) -> Dict[str, Any]:
        """
        Run one Step1X RQ1 generation.
        """

        generation_cfg = self.config[
            "generation"
        ]

        # =====================================================
        # Validate input
        # =====================================================

        if len(images) != 1:
            raise ValueError(
                f"{example_uid}: "
                f"RQ1 requires exactly one "
                f"source image, got "
                f"{len(images)}."
            )

        image_input = images[0]

        # =====================================================
        # Configuration
        # =====================================================

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

        # =====================================================
        # Seeds
        # =====================================================

        stage_seeds = get_rq1_stage_seeds(
            base_seed=base_seed,
            example_uid=example_uid,
            max_try_cnt=max_try_cnt,
        )

        # Step1X's thinking module samples through the
        # global torch RNG.
        set_global_seed(
            stage_seeds["thinking"]
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Step1X RQ1 generation "
                "requires CUDA."
            )

        # Step1X receives one explicitly seeded diffusion
        # generator. If its native correction loop retries,
        # that same generator continues its RNG stream.
        generator = torch.Generator(
            device="cuda"
        ).manual_seed(
            stage_seeds["diffusion_0"]
        )

        # =====================================================
        # GPU statistics
        # =====================================================

        torch.cuda.reset_peak_memory_stats()

        self._sync_cuda()

        pipeline_start = time.perf_counter()

        # =====================================================
        # Inference
        # =====================================================

        result = self.pipe(
            image=image_input,
            prompt=instruction,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            enable_thinking_mode=thinking_enabled,
            enable_reflection_mode=reflection_enabled,
            max_try_cnt=max_try_cnt,
            generator=generator,
        )

        self._sync_cuda()

        latency_pipeline_sec = (
            time.perf_counter()
            - pipeline_start
        )

        # =====================================================
        # GPU memory
        # =====================================================

        peak_vram_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        # =====================================================
        # Candidate outputs
        # =====================================================

        result_images = list(
            getattr(
                result,
                "images",
                [],
            )
        )

        if len(result_images) == 0:
            raise RuntimeError(
                f"{example_uid}: "
                "Step1X returned zero output images."
            )

        # Step1X-v1p2 returns all generated attempts in
        # result.images and its internally selected image in
        # result.final_images.
        final_images = list(
            getattr(
                result,
                "final_images",
                [],
            )
        )

        if len(final_images) > 0:
            final_image = final_images[0]
        else:
            final_image = result_images[0]

        # =====================================================
        # Reasoning / reflection metadata
        # =====================================================

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

        # In Step1X reflection mode, result.think_info contains
        # the reflection output for each generated attempt.
        reflection_info = (
            think_info
            if reflection_enabled
            else None
        )

        # =====================================================
        # Token count
        # =====================================================

        tokenizer = (
            self._get_available_tokenizer()
        )

        thinking_token_count = (
            self._count_text_tokens(
                tokenizer=tokenizer,
                text=reformat_prompt,
            )
        )

        # Reflection contains one string per attempt. Keep the
        # exact strings in reflection_info; do not collapse them
        # into a misleading single per-round token count.
        reflection_token_count = None

        # =====================================================
        # Correction metadata
        # =====================================================

        # result.images is the native Step1X candidate history,
        # so its length is the actual number of generated
        # attempts.
        attempt_count = len(
            result_images
        )

        retry_triggered = (
            attempt_count > 1
        )

        selected_candidate_index = (
            self._get_selected_candidate_index(
                best_info=best_info,
                think_info=think_info,
                num_candidates=attempt_count,
            )
        )

        selected_is_last_candidate = (
            selected_candidate_index
            == attempt_count - 1
        )

        stop_reason = (
            self._get_stop_reason(
                think_info=think_info,
                retry_triggered=retry_triggered,
            )
        )

        # =====================================================
        # Return
        # =====================================================

        return {
            # Main output
            "final_image":
                final_image,

            # All native Step1X candidates / attempts
            "candidate_images":
                result_images,

            # Reasoning
            "reformat_prompt":
                reformat_prompt,

            "think_info":
                think_info,

            "thinking_token_count":
                thinking_token_count,

            "best_info":
                best_info,

            "reflection_info":
                reflection_info,

            "reflection_token_count":
                reflection_token_count,

            # Correction metadata
            "attempt_count":
                attempt_count,

            "selected_candidate_index":
                selected_candidate_index,

            "retry_triggered":
                retry_triggered,

            "stop_reason":
                stop_reason,

            "selected_is_last_candidate":
                selected_is_last_candidate,

            # Seeds
            "base_seed":
                base_seed,

            "stage_seeds":
                stage_seeds,

            # Timing
            "latency_pipeline_sec":
                latency_pipeline_sec,

            # These require instrumentation inside the
            # Step1X implementation and are intentionally
            # not guessed.
            "latency_thinking_sec":
                None,

            "latency_diffusion_attempts_sec":
                [],

            "latency_reflection_rounds_sec":
                [],

            "latency_offload_sec":
                None,

            # Resources
            "peak_vram_gb":
                peak_vram_gb,
        }

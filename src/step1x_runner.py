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
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required when "
                    "cpu_offload=True."
                )

            self.pipe.enable_model_cpu_offload(
                gpu_id=0,
                device="cuda",
            )

            # Step1X calls the Qwen nested embed_tokens() directly during
            # prompt encoding. This bypasses the CPU-offload hook installed
            # on the top-level text_encoder.forward(). Explicitly trigger
            # that existing hook before each embedding encode operation.
            # This does not modify the installed Step1X library source.
            self._install_qwen_embedding_offload_bridge()

        else:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required when "
                    "cpu_offload=False."
                )

            self.pipe.to("cuda")

        self._print_deployment_summary()
        print("Step1X loaded.")

    def _install_qwen_embedding_offload_bridge(self) -> None:
        """On-load Qwen before Step1X directly calls nested embedding layers.

        Diffusers/Accelerate attaches a model-level CPU-offload hook to the
        text encoder's forward(). Step1X's _get_qwenvl_embeds() instead
        invokes text_encoder.model.language_model.embed_tokens() directly,
        so the parent hook does not run and the weight stays on CPU while
        token IDs are on CUDA.

        Trigger the *existing* parent hook before each Qwen embedding pass.
        The normal next-model offload chain still handles transitions to
        transformer/vae. No Step1X source or model weights are modified.
        """
        encoder = getattr(self.pipe, "text_encoder", None)
        offload_hook = getattr(encoder, "_hf_hook", None)
        pre_forward = getattr(offload_hook, "pre_forward", None)
        original_encode = getattr(self.pipe, "_get_qwenvl_embeds", None)

        if encoder is None or not callable(pre_forward):
            raise RuntimeError(
                "Model CPU offload was enabled but its text_encoder "
                "hook was not found. Check Accelerate/diffusers versions."
            )
        if not callable(original_encode):
            raise RuntimeError(
                "Step1X-v1p2 _get_qwenvl_embeds method was not found."
            )

        embedding = encoder.model.language_model.embed_tokens
        count = 0

        def encode_with_loaded_qwen(*args, **kwargs):
            nonlocal count
            # Invoke the top-level accelerate CpuOffload hook normally
            # reached when text_encoder.forward() is called.
            # Step1X bypasses text_encoder.forward() during embedding.
            offload_hook.pre_forward(encoder)
            device = embedding.weight.device
            if device.type != "cuda":
                raise RuntimeError(
                    "Text encoder was not moved onto CUDA before "
                    f"Step1X token embedding (weight device={device})."
                )
            count += 1
            if count == 1:
                print(f"Qwen embedding onloaded to {device} before encode.")
            return original_encode(*args, **kwargs)

        self.pipe._get_qwenvl_embeds = encode_with_loaded_qwen
        print(
            "Installed Qwen embedding bridge for model CPU offload "
            f"(hook={type(offload_hook).__name__})."
        )

        # Thinking uses text_encoder.generate(), not the pipeline's
        # _get_qwenvl_embeds() helper. The CPU-offload hook moves Qwen onto
        # CUDA only when forward() starts, which is AFTER generate() has
        # prepared/placed its inputs. In that case, input IDs can be on CPU
        # while the embedding weights have been moved to CUDA.
        #
        # Onload the same existing model hook before generate() begins,
        # and align caller-supplied tensors with that loaded model. This
        # keeps the fix in our wrapper; no diffusers/transformers patching.
        original_generate = getattr(encoder, "generate", None)
        if not callable(original_generate):
            if self.config["generation"].get("enable_thinking_mode", False):
                raise RuntimeError(
                    "Thinking enabled, but text_encoder.generate() "
                    "is not available for CPU-offload bridging."
                )
            return

        generate_count = 0

        def move_generation_data(value, device):
            if isinstance(value, torch.Tensor):
                return value.to(device=device, non_blocking=True)
            if isinstance(value, dict):
                return {
                    key: move_generation_data(item, device)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [move_generation_data(item, device) for item in value]
            if isinstance(value, tuple):
                return tuple(move_generation_data(item, device) for item in value)
            return value

        def generate_with_loaded_qwen(*args, **kwargs):
            nonlocal generate_count
            # Make the Qwen weights CUDA-resident *before* transformers
            # prepares decoding tensors, not just when forward() starts.
            offload_hook.pre_forward(encoder)
            model_device = embedding.weight.device
            if model_device.type != "cuda":
                raise RuntimeError(
                    "Thinking Qwen model was not loaded onto CUDA "
                    f"before generate (weight device={model_device})."
                )

            cuda_args = tuple(
                move_generation_data(item, model_device) for item in args
            )
            cuda_kwargs = {
                key: move_generation_data(value, model_device)
                for key, value in kwargs.items()
            }
            generate_count += 1
            if generate_count == 1:
                print(
                    "Qwen thinking generate onloaded to "
                    f"{model_device} with aligned input tensors."
                )
            return original_generate(*cuda_args, **cuda_kwargs)

        encoder.generate = generate_with_loaded_qwen
        print("Installed Qwen thinking-generation CPU-offload bridge.")

    def _print_deployment_summary(
        self,
    ) -> None:
        print(
            "Deployment mode:",
            (
                "model_cpu_offload"
                if self.cpu_offload
                else "full_cuda"
            ),
        )

        if self.cpu_offload:
            print(
                "Model offload sequence:",
                getattr(
                    self.pipe,
                    "model_cpu_offload_seq",
                    None,
                ),
            )

        for name in (
            "text_encoder",
            "transformer",
            "vae",
        ):
            module = getattr(
                self.pipe,
                name,
                None,
            )

            if module is None:
                continue

            try:
                parameter_device = next(
                    module.parameters()
                ).device
            except (
                StopIteration,
                AttributeError,
            ):
                parameter_device = "n/a"

            buffer_devices = sorted(
                {
                    str(buffer.device)
                    for buffer
                    in module.buffers()
                }
            )

            print(
                f"{name}: "
                f"parameter_device={parameter_device}, "
                f"buffer_devices={buffer_devices}"
            )

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _count_text_tokens(
        tokenizer,
        text: Optional[str],
    ) -> Optional[int]:
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
        generation_cfg = self.config[
            "generation"
        ]

        if len(images) != 1:
            raise ValueError(
                f"{example_uid}: "
                f"RQ1 requires exactly one "
                f"source image, got "
                f"{len(images)}."
            )

        image_input = images[0]

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

        stage_seeds = get_rq1_stage_seeds(
            base_seed=base_seed,
            example_uid=example_uid,
            max_try_cnt=max_try_cnt,
        )

        set_global_seed(
            stage_seeds["thinking"]
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Step1X RQ1 generation "
                "requires CUDA."
            )

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(
            stage_seeds["diffusion_0"]
        )

        torch.cuda.reset_peak_memory_stats()

        self._sync_cuda()
        pipeline_start = time.perf_counter()

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

        peak_vram_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

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

        reflection_info = (
            think_info
            if reflection_enabled
            else None
        )

        tokenizer = (
            self._get_available_tokenizer()
        )

        thinking_token_count = (
            self._count_text_tokens(
                tokenizer=tokenizer,
                text=reformat_prompt,
            )
        )

        reflection_token_count = None

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

        return {
            "final_image":
                final_image,

            "candidate_images":
                result_images,

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

            "base_seed":
                base_seed,

            "stage_seeds":
                stage_seeds,

            "latency_pipeline_sec":
                latency_pipeline_sec,

            "latency_thinking_sec":
                None,

            "latency_diffusion_attempts_sec":
                [],

            "latency_reflection_rounds_sec":
                [],

            "latency_offload_sec":
                None,

            "peak_vram_gb":
                peak_vram_gb,
        }

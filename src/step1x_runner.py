# src/step1x_runner.py

import time
from typing import Any, Dict, List

import torch
from diffusers import Step1XEditPipelineV1P2


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class Step1XRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        model_cfg = config["model"]

        dtype_name = model_cfg["dtype"]

        if dtype_name not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype: {dtype_name}"
            )

        dtype = DTYPE_MAP[dtype_name]

        print(
            f"Loading model: {model_cfg['model_id']}"
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
            self.pipe.to("cuda")

        print("Step1X loaded.")

    def run(
        self,
        images: List,
        instruction: str,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Run one Step1X edit.

        images:
            List of decoded PIL source images.

        instruction:
            KRIS-Bench editing instruction.

        seed:
            Diffusion seed for this example.
        """

        generation_cfg = self.config["generation"]

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        result = self.pipe(
            image=images,
            prompt=instruction,

            num_inference_steps=
                generation_cfg[
                    "num_inference_steps"
                ],

            true_cfg_scale=
                generation_cfg[
                    "true_cfg_scale"
                ],

            generator=generator,

            enable_thinking_mode=
                generation_cfg[
                    "enable_thinking_mode"
                ],

            enable_reflection_mode=
                generation_cfg[
                    "enable_reflection_mode"
                ],
        )

        latency_total = (
            time.perf_counter()
            - start
        )

        peak_vram_gb = None

        if torch.cuda.is_available():
            peak_vram_gb = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

        return {
            "final_image":
                result.final_images[0],

            "candidate_images":
                getattr(
                    result,
                    "images",
                    None,
                ),

            "reformat_prompt":
                getattr(
                    result,
                    "reformat_prompt",
                    None,
                ),

            "think_info":
                getattr(
                    result,
                    "think_info",
                    None,
                ),

            "best_info":
                getattr(
                    result,
                    "best_info",
                    None,
                ),

            "latency_total_sec":
                latency_total,

            "peak_vram_gb":
                peak_vram_gb,
        }
# src/logger.py

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class ExperimentLogger:
    """
    JSONL logger for image-editing experiments.

    Each generation is written as one JSON object per line.
    This makes it easy to append results incrementally and
    recover partial runs if a job is interrupted.
    """

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(self, record: Dict[str, Any]) -> None:
        """
        Append one generation record to the JSONL file.
        """

        record = dict(record)

        record["logged_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )

        with open(
            self.log_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=self._json_default,
                )
                + "\n"
            )

    @staticmethod
    def _json_default(obj):
        """
        Convert otherwise non-serializable objects to strings.
        """
        return str(obj)


def build_generation_record(
    *,
    example_uid: str,
    example_id: str,
    category: str,
    reasoning_dimension: str,
    knowledge_type: str,
    instruction: str,

    condition: str,
    seed: int,

    num_inference_steps: int,
    true_cfg_scale: float,
    thinking_enabled: bool,
    reflection_enabled: bool,

    output_path: str,

    latency_total_sec: float,
    peak_vram_gb: Optional[float] = None,

    reformat_prompt: Optional[str] = None,
    think_info: Optional[Any] = None,
    best_info: Optional[Any] = None,

    latency_thinking_sec: Optional[float] = None,
    latency_diffusion_sec: Optional[float] = None,
    latency_reflection_sec: Optional[float] = None,

    num_source_images: Optional[int] = None,

    status: str = "success",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build one standard experiment record.

    Keep the schema identical across:
    - base
    - thinking
    - thinking + reflection

    so results can later be merged directly.
    """

    return {
        # Example identity
        "example_uid": example_uid,
        "example_id": example_id,
        "category": category,
        "reasoning_dimension": reasoning_dimension,
        "knowledge_type": knowledge_type,

        # Input
        "instruction": instruction,
        "num_source_images": num_source_images,

        # Experimental condition
        "condition": condition,
        "seed": seed,

        # Generation parameters
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "thinking_enabled": thinking_enabled,
        "reflection_enabled": reflection_enabled,

        # Reasoning outputs
        "reformat_prompt": reformat_prompt,
        "think_info": think_info,
        "best_info": best_info,

        # Timing
        "latency_total_sec": latency_total_sec,
        "latency_thinking_sec": latency_thinking_sec,
        "latency_diffusion_sec": latency_diffusion_sec,
        "latency_reflection_sec": latency_reflection_sec,

        # Hardware usage
        "peak_vram_gb": peak_vram_gb,

        # Output
        "output_path": output_path,

        # Run status
        "status": status,
        "error": error,
    }
    
def load_logged_example_uids(log_file: str):
    """
    Return example UIDs already successfully completed.

    Useful for resuming interrupted experiments.
    """

    path = Path(log_file)

    if not path.exists():
        return set()

    completed = set()

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record.get("status") == "success":
                completed.add(
                    record["example_uid"]
                )

    return completed


def log_failure(
    logger: ExperimentLogger,
    *,
    example_uid: str,
    example_id: str,
    category: str,
    reasoning_dimension: str,
    knowledge_type: str,
    instruction: str,
    condition: str,
    seed: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    thinking_enabled: bool,
    reflection_enabled: bool,
    error: Exception,
):
    """
    Log a failed example without stopping the entire experiment.
    """

    record = build_generation_record(
        example_uid=example_uid,
        example_id=example_id,
        category=category,
        reasoning_dimension=reasoning_dimension,
        knowledge_type=knowledge_type,
        instruction=instruction,

        condition=condition,
        seed=seed,

        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        thinking_enabled=thinking_enabled,
        reflection_enabled=reflection_enabled,

        output_path="",

        latency_total_sec=0.0,

        status="failed",
        error=repr(error),
    )

    logger.log(record)
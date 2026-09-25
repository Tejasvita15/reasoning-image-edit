# src/logger.py

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple, Set


class ExperimentLogger:
    """
    JSONL logger for RQ1 image-editing experiments.

    One generation = one JSON object per line.

    This allows:
    - incremental logging,
    - recovery after SLURM interruption,
    - multi-seed experiments,
    - consistent comparison across Base / Thinking / Correction.
    """

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(
        self,
        record: Dict[str, Any],
    ) -> None:
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
    # ---------------------------------------------------------
    # Example identity
    # ---------------------------------------------------------
    example_uid: str,
    example_id: str,
    category: str,
    reasoning_dimension: str,
    knowledge_type: str,

    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------
    instruction: str,
    num_source_images: int,

    # ---------------------------------------------------------
    # Experiment identity
    # ---------------------------------------------------------
    model_id: str,
    condition: str,
    split_name: str,

    # ---------------------------------------------------------
    # Randomness
    # ---------------------------------------------------------
    base_seed: int,
    stage_seeds: Optional[Dict[str, int]],

    # ---------------------------------------------------------
    # Generation config
    # ---------------------------------------------------------
    num_inference_steps: int,
    true_cfg_scale: float,
    thinking_enabled: bool,
    reflection_enabled: bool,
    max_try_cnt: int,

    # ---------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------
    output_path: str,

    candidate_paths: Optional[List[str]] = None,
    selected_candidate_index: Optional[int] = None,
    attempt_count: Optional[int] = None,
    retry_triggered: Optional[bool] = None,

    # ---------------------------------------------------------
    # Reasoning / correction information
    # ---------------------------------------------------------
    reformat_prompt: Optional[str] = None,
    think_info: Optional[Any] = None,
    best_info: Optional[Any] = None,

    reflection_info: Optional[Any] = None,

    # ---------------------------------------------------------
    # Timing
    # ---------------------------------------------------------
    latency_total_sec: Optional[float] = None,
    latency_preprocess_sec: Optional[float] = None,
    latency_thinking_sec: Optional[float] = None,
    latency_diffusion_sec: Optional[float] = None,
    latency_reflection_sec: Optional[float] = None,
    latency_postprocess_sec: Optional[float] = None,

    # ---------------------------------------------------------
    # Hardware
    # ---------------------------------------------------------
    peak_vram_gb: Optional[float] = None,

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------
    status: str = "success",
    error: Optional[str] = None,

    # ---------------------------------------------------------
    # Evaluation placeholder
    # ---------------------------------------------------------
    judge_result: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build one standardized RQ1 generation record.

    The same schema is used for:
    - Base
    - Thinking-only
    - Thinking + correction
    """

    if candidate_paths is None:
        candidate_paths = []

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

        # Experiment identity
        "model_id": model_id,
        "condition": condition,
        "split_name": split_name,

        # Randomness
        "base_seed": base_seed,
        "stage_seeds": stage_seeds,

        # Generation parameters
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "thinking_enabled": thinking_enabled,
        "reflection_enabled": reflection_enabled,
        "max_try_cnt": max_try_cnt,

        # Reasoning / correction outputs
        "reformat_prompt": reformat_prompt,
        "think_info": think_info,
        "best_info": best_info,
        "reflection_info": reflection_info,

        # Candidate information
        "candidate_paths": candidate_paths,
        "selected_candidate_index":
            selected_candidate_index,
        "attempt_count": attempt_count,
        "retry_triggered": retry_triggered,

        # Timing
        "latency_total_sec": latency_total_sec,
        "latency_preprocess_sec":
            latency_preprocess_sec,
        "latency_thinking_sec":
            latency_thinking_sec,
        "latency_diffusion_sec":
            latency_diffusion_sec,
        "latency_reflection_sec":
            latency_reflection_sec,
        "latency_postprocess_sec":
            latency_postprocess_sec,

        # Hardware
        "peak_vram_gb": peak_vram_gb,

        # Main output
        "output_path": output_path,

        # Evaluation placeholder
        "judge_result": judge_result,

        # Status
        "status": status,
        "error": error,
    }


def make_run_key(
    example_uid: str,
    condition: str,
    base_seed: int,
    split_name: str,
) -> Tuple[str, str, int, str]:
    """
    Create a unique identity for one RQ1 generation.

    Important:
    example_uid alone is NOT sufficient because K-small
    is evaluated with multiple seeds.
    """

    return (
        str(example_uid),
        str(condition),
        int(base_seed),
        str(split_name),
    )


def load_completed_run_keys(
    log_file: str,
) -> Set[Tuple[str, str, int, str]]:
    """
    Return successfully completed RQ1 run keys.

    Used to resume interrupted jobs.

    A run is identified by:
        example_uid
        condition
        base_seed
        split_name
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

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Ignore a potentially incomplete final line
                # if a job was killed during logging.
                continue

            if record.get("status") != "success":
                continue

            example_uid = record.get(
                "example_uid"
            )

            condition = record.get(
                "condition"
            )

            # Support old logs that used "seed".
            base_seed = record.get(
                "base_seed",
                record.get("seed"),
            )

            split_name = record.get(
                "split_name",
                "legacy",
            )

            if (
                example_uid is None
                or condition is None
                or base_seed is None
            ):
                continue

            completed.add(
                make_run_key(
                    example_uid=example_uid,
                    condition=condition,
                    base_seed=int(base_seed),
                    split_name=split_name,
                )
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
    num_source_images: int,

    model_id: str,
    condition: str,
    split_name: str,

    base_seed: int,
    stage_seeds: Optional[Dict[str, int]],

    num_inference_steps: int,
    true_cfg_scale: float,
    thinking_enabled: bool,
    reflection_enabled: bool,
    max_try_cnt: int,

    error: Exception,
) -> None:
    """
    Log a failed example without terminating the whole RQ1 run.
    """

    record = build_generation_record(
        # Identity
        example_uid=example_uid,
        example_id=example_id,
        category=category,
        reasoning_dimension=reasoning_dimension,
        knowledge_type=knowledge_type,

        # Input
        instruction=instruction,
        num_source_images=num_source_images,

        # Experiment
        model_id=model_id,
        condition=condition,
        split_name=split_name,

        # Randomness
        base_seed=base_seed,
        stage_seeds=stage_seeds,

        # Generation
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        thinking_enabled=thinking_enabled,
        reflection_enabled=reflection_enabled,
        max_try_cnt=max_try_cnt,

        # No output
        output_path="",

        # Status
        status="failed",
        error=repr(error),
    )

    logger.log(record)
# src/logger.py

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import (Any,Dict,Optional,List,Tuple,Set,)


class ExperimentLogger:
    """
    JSONL logger for RQ1 image-editing experiments.

    One generation = one JSON object per line.

    Supports:
    - incremental logging,
    - SLURM interruption recovery,
    - multi-seed experiments,
    - Base / Thinking / Correction conditions.
    """

    def __init__(
        self,
        log_file: str,
    ):
        self.log_file = Path(
            log_file
        )

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(
        self,
        record: Dict[str, Any],
    ) -> None:
        """
        Append one generation record
        to the JSONL file.
        """

        record = dict(record)

        record["logged_at_utc"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
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
        Convert otherwise non-serializable
        objects to strings.
        """

        return str(obj)


def build_generation_record(
    *,

    # =========================================================
    # Example identity
    # =========================================================

    example_uid: str,
    example_id: str,
    category: str,
    reasoning_dimension: str,
    knowledge_type: str,

    # =========================================================
    # Input
    # =========================================================

    instruction: str,
    num_source_images: int,

    # =========================================================
    # Experiment identity
    # =========================================================

    model_id: str,
    condition: str,
    split_name: str,

    # Complete frozen config used for this generation.
    full_config: Dict[str, Any],

    # =========================================================
    # Randomness
    # =========================================================

    base_seed: int,
    stage_seeds: Optional[
        Dict[str, int]
    ],

    # =========================================================
    # Generation configuration
    # =========================================================

    num_inference_steps: int,
    true_cfg_scale: float,

    thinking_enabled: bool,
    reflection_enabled: bool,

    max_try_cnt: int,

    # =========================================================
    # Main output
    # =========================================================

    output_path: str,

    # =========================================================
    # Candidate / retry information
    # =========================================================

    candidate_paths: Optional[
        List[str]
    ] = None,

    selected_candidate_index:
        Optional[int] = None,

    attempt_count:
        Optional[int] = None,

    retry_triggered:
        Optional[bool] = None,

    stop_reason:
        Optional[str] = None,

    selected_is_last_candidate:
        Optional[bool] = None,

    # =========================================================
    # Reasoning / correction information
    # =========================================================

    reformat_prompt:
        Optional[str] = None,

    think_info:
        Optional[Any] = None,

    thinking_token_count:
        Optional[int] = None,

    best_info:
        Optional[Any] = None,

    reflection_info:
        Optional[Any] = None,

    reflection_token_count:
        Optional[int] = None,

    # =========================================================
    # Timing
    # =========================================================

    # Directly measured entire example:
    # decode/preprocess -> inference -> save.
    latency_end_to_end_sec:
        Optional[float] = None,

    # Entire Step1X pipeline call only.
    latency_pipeline_sec:
        Optional[float] = None,

    latency_preprocess_sec:
        Optional[float] = None,

    latency_thinking_sec:
        Optional[float] = None,

    # Each generation attempt separately.
    latency_diffusion_attempts_sec:
        Optional[List[float]] = None,

    # Each reflection round separately.
    latency_reflection_rounds_sec:
        Optional[List[float]] = None,

    latency_postprocess_sec:
        Optional[float] = None,

    latency_offload_sec:
        Optional[float] = None,

    # =========================================================
    # Hardware
    # =========================================================

    peak_vram_gb:
        Optional[float] = None,

    # =========================================================
    # Status
    # =========================================================

    status: str = "success",

    error:
        Optional[str] = None,

    # =========================================================
    # Evaluation placeholder
    # =========================================================

    judge_result:
        Optional[Any] = None,

) -> Dict[str, Any]:

    """
    Build one standardized RQ1 generation record.

    Used by:
    - direct editing,
    - thinking-only,
    - thinking + correction.
    """

    if candidate_paths is None:
        candidate_paths = []

    if (
        latency_diffusion_attempts_sec
        is None
    ):
        latency_diffusion_attempts_sec = []

    if (
        latency_reflection_rounds_sec
        is None
    ):
        latency_reflection_rounds_sec = []

    return {

        # =====================================================
        # Example
        # =====================================================

        "example_uid":
            example_uid,

        "example_id":
            example_id,

        "category":
            category,

        "reasoning_dimension":
            reasoning_dimension,

        "knowledge_type":
            knowledge_type,

        # =====================================================
        # Input
        # =====================================================

        "instruction":
            instruction,

        "num_source_images":
            num_source_images,

        # =====================================================
        # Experiment
        # =====================================================

        "model_id":
            model_id,

        "condition":
            condition,

        "split_name":
            split_name,

        "full_config":
            full_config,

        # =====================================================
        # Randomness
        # =====================================================

        "base_seed":
            base_seed,

        "stage_seeds":
            stage_seeds,

        # =====================================================
        # Generation parameters
        # =====================================================

        "num_inference_steps":
            num_inference_steps,

        "true_cfg_scale":
            true_cfg_scale,

        "thinking_enabled":
            thinking_enabled,

        "reflection_enabled":
            reflection_enabled,

        "max_try_cnt":
            max_try_cnt,

        # =====================================================
        # Reasoning / correction
        # =====================================================

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

        # =====================================================
        # Candidate information
        # =====================================================

        "candidate_paths":
            candidate_paths,

        "selected_candidate_index":
            selected_candidate_index,

        "attempt_count":
            attempt_count,

        "retry_triggered":
            retry_triggered,

        "stop_reason":
            stop_reason,

        "selected_is_last_candidate":
            selected_is_last_candidate,

        # =====================================================
        # Timing
        # =====================================================

        "latency_end_to_end_sec":
            latency_end_to_end_sec,

        "latency_pipeline_sec":
            latency_pipeline_sec,

        "latency_preprocess_sec":
            latency_preprocess_sec,

        "latency_thinking_sec":
            latency_thinking_sec,

        "latency_diffusion_attempts_sec":
            latency_diffusion_attempts_sec,

        "latency_reflection_rounds_sec":
            latency_reflection_rounds_sec,

        "latency_postprocess_sec":
            latency_postprocess_sec,

        "latency_offload_sec":
            latency_offload_sec,

        # =====================================================
        # Hardware
        # =====================================================

        "peak_vram_gb":
            peak_vram_gb,

        # =====================================================
        # Output
        # =====================================================

        "output_path":
            output_path,

        # =====================================================
        # Evaluation placeholder
        # =====================================================

        "judge_result":
            judge_result,

        # =====================================================
        # Status
        # =====================================================

        "status":
            status,

        "error":
            error,
    }


def make_run_key(
    example_uid: str,
    condition: str,
    base_seed: int,
    split_name: str,
) -> Tuple[
    str,
    str,
    int,
    str,
]:
    """
    Create a unique identity for one
    RQ1 generation.

    example_uid alone is insufficient
    because K-small uses multiple seeds.
    """

    return (
        str(example_uid),
        str(condition),
        int(base_seed),
        str(split_name),
    )


def load_completed_run_keys(
    log_file: str,
) -> Set[
    Tuple[
        str,
        str,
        int,
        str,
    ]
]:
    """
    Return successfully completed
    RQ1 run keys.

    Used for resuming interrupted jobs.
    """

    path = Path(
        log_file
    )

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
                record = json.loads(
                    line
                )

            except json.JSONDecodeError:

                # A SLURM job may have been killed
                # while writing the final line.
                continue

            if (
                record.get("status")
                != "success"
            ):
                continue

            example_uid = (
                record.get(
                    "example_uid"
                )
            )

            condition = (
                record.get(
                    "condition"
                )
            )

            # Compatibility with older logs.
            base_seed = (
                record.get(
                    "base_seed",
                    record.get(
                        "seed"
                    ),
                )
            )

            split_name = (
                record.get(
                    "split_name",
                    "legacy",
                )
            )

            if (
                example_uid is None
                or condition is None
                or base_seed is None
            ):
                continue

            completed.add(
                make_run_key(
                    example_uid=
                        example_uid,

                    condition=
                        condition,

                    base_seed=
                        int(
                            base_seed
                        ),

                    split_name=
                        split_name,
                )
            )

    return completed


def log_failure(
    logger: ExperimentLogger,
    *,

    # =========================================================
    # Identity
    # =========================================================

    example_uid: str,
    example_id: str,
    category: str,
    reasoning_dimension: str,
    knowledge_type: str,

    # =========================================================
    # Input
    # =========================================================

    instruction: str,
    num_source_images: int,

    # =========================================================
    # Experiment
    # =========================================================

    model_id: str,
    condition: str,
    split_name: str,

    full_config: Dict[
        str,
        Any,
    ],

    # =========================================================
    # Randomness
    # =========================================================

    base_seed: int,

    stage_seeds:
        Optional[
            Dict[str, int]
        ],

    # =========================================================
    # Generation
    # =========================================================

    num_inference_steps: int,
    true_cfg_scale: float,

    thinking_enabled: bool,
    reflection_enabled: bool,

    max_try_cnt: int,

    # =========================================================
    # Failure
    # =========================================================

    error: Exception,

    latency_end_to_end_sec:
        Optional[float] = None,

) -> None:
    """
    Log one failed RQ1 example without
    terminating the full experiment.
    """

    record = build_generation_record(

        # Identity
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

        # Input
        instruction=
            instruction,

        num_source_images=
            num_source_images,

        # Experiment
        model_id=
            model_id,

        condition=
            condition,

        split_name=
            split_name,

        full_config=
            full_config,

        # Randomness
        base_seed=
            base_seed,

        stage_seeds=
            stage_seeds,

        # Generation
        num_inference_steps=
            num_inference_steps,

        true_cfg_scale=
            true_cfg_scale,

        thinking_enabled=
            thinking_enabled,

        reflection_enabled=
            reflection_enabled,

        max_try_cnt=
            max_try_cnt,

        # No successful output
        output_path="",

        # Timing if failure occurred
        # after processing started
        latency_end_to_end_sec=
            latency_end_to_end_sec,

        # Status
        status="failed",

        error=repr(
            error
        ),
    )

    logger.log(record)
# src/config.py

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a YAML experiment configuration.
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Config file is empty: {path}"
        )

    validate_config(config)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the minimum fields required for a
    reasoning-image-edit experiment configuration.

    Step1X (default) and UniReason (model.backend: unireason)
    share the top-level layout but have different required fields.
    The shared fields are validated here; backend-specific checks
    are delegated to _validate_step1x or _validate_unireason.
    """

    # ---------------------------------------------------------
    # Top-level fields (all backends)
    # ---------------------------------------------------------

    for key in (
        "experiment_name",
        "model",
        "generation",
        "output_dir",
        "log_file",
    ):
        if key not in config:
            raise ValueError(
                f"Missing required config field: {key}"
            )

    backend = config["model"].get("backend", "step1x")

    if backend == "unireason":
        _validate_unireason_config(config)
    else:
        _validate_step1x_config(config)


def _validate_step1x_config(config: Dict[str, Any]) -> None:
    """Validation rules specific to Step1X configs."""

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------

    for key in ("model_id", "dtype", "cpu_offload"):
        if key not in config["model"]:
            raise ValueError(
                f"Missing model config field: {key}"
            )

    if config["model"]["dtype"] not in {
        "bfloat16", "float16", "float32"
    }:
        raise ValueError(
            "model.dtype must be one of: "
            "bfloat16, float16, float32"
        )

    if not isinstance(config["model"]["cpu_offload"], bool):
        raise ValueError(
            "model.cpu_offload must be true or false"
        )

    # ---------------------------------------------------------
    # Generation configuration
    # ---------------------------------------------------------

    for key in (
        "num_inference_steps",
        "true_cfg_scale",
        "enable_thinking_mode",
        "enable_reflection_mode",
        "max_try_cnt",
    ):
        if key not in config["generation"]:
            raise ValueError(
                f"Missing generation config field: {key}"
            )

    num_steps = config["generation"]["num_inference_steps"]
    cfg_scale = config["generation"]["true_cfg_scale"]
    thinking_enabled = config["generation"]["enable_thinking_mode"]
    reflection_enabled = config["generation"]["enable_reflection_mode"]
    max_try_cnt = config["generation"]["max_try_cnt"]

    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(
            "num_inference_steps must be a positive integer"
        )

    if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0:
        raise ValueError(
            "true_cfg_scale must be a non-negative number"
        )

    if not isinstance(thinking_enabled, bool):
        raise ValueError(
            "enable_thinking_mode must be true or false"
        )

    if not isinstance(reflection_enabled, bool):
        raise ValueError(
            "enable_reflection_mode must be true or false"
        )

    if not isinstance(max_try_cnt, int) or max_try_cnt < 1:
        raise ValueError(
            "max_try_cnt must be an integer >= 1"
        )

    if not reflection_enabled and max_try_cnt != 1:
        raise ValueError(
            "When reflection is disabled, "
            "max_try_cnt must be 1 for RQ1."
        )

    if reflection_enabled and max_try_cnt != 2:
        raise ValueError(
            "For the RQ1 correction condition, "
            "reflection is enabled and max_try_cnt "
            "must be 2."
        )


def _validate_unireason_config(config: Dict[str, Any]) -> None:
    """Validation rules specific to UniReason (BAGEL) configs."""

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------

    for key in ("model_id", "model_path"):
        if key not in config["model"]:
            raise ValueError(
                f"Missing model config field: {key}"
            )

    # ---------------------------------------------------------
    # Generation configuration
    # ---------------------------------------------------------

    for key in (
        "num_inference_steps",
        "true_cfg_scale",
        "enable_thinking_mode",
    ):
        if key not in config["generation"]:
            raise ValueError(
                f"Missing generation config field: {key}"
            )

    num_steps = config["generation"]["num_inference_steps"]
    cfg_text = config["generation"]["true_cfg_scale"]
    thinking = config["generation"]["enable_thinking_mode"]

    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(
            "num_inference_steps must be a positive integer"
        )

    if not isinstance(cfg_text, (int, float)) or cfg_text < 0:
        raise ValueError(
            "true_cfg_scale must be a non-negative number"
        )

    if not isinstance(thinking, bool):
        raise ValueError(
            "enable_thinking_mode must be true or false"
        )

    cfg_img = config["generation"].get("cfg_img_scale")
    if cfg_img is not None and (
        not isinstance(cfg_img, (int, float)) or cfg_img < 0
    ):
        raise ValueError(
            "cfg_img_scale must be a non-negative number"
        )


def print_config_summary(
    config: Dict[str, Any],
    seed: int | None = None,
) -> None:
    """
    Print a compact experiment summary before execution.

    The experiment seed is normally supplied by the
    command-line runner rather than stored in YAML.
    """

    print("=" * 60)

    print(
        "Experiment:",
        config["experiment_name"],
    )

    print(
        "Model:",
        config["model"]["model_id"],
    )

    print(
        "Dtype:",
        config["model"]["dtype"],
    )

    print(
        "CPU offload:",
        config["model"]["cpu_offload"],
    )

    backend = config["model"].get("backend", "step1x")

    print(
        "Backend:",
        backend,
    )

    print(
        "Steps:",
        config["generation"]["num_inference_steps"],
    )

    print(
        "CFG text:",
        config["generation"]["true_cfg_scale"],
    )

    if backend == "unireason":
        print(
            "CFG img:",
            config["generation"].get("cfg_img_scale", 1.5),
        )
    else:
        print(
            "Reflection:",
            config["generation"].get(
                "enable_reflection_mode"
            ),
        )

        print(
            "Max attempts:",
            config["generation"].get("max_try_cnt"),
        )

    print(
        "Thinking:",
        config["generation"][
            "enable_thinking_mode"
        ],
    )

    if seed is not None:
        print(
            "Base seed:",
            seed,
        )
    elif "seed" in config:
        # Backward compatibility with old configs.
        print(
            "Base seed:",
            config["seed"],
        )
    else:
        print(
            "Base seed:",
            "provided by command line",
        )

    print(
        "Output:",
        config["output_dir"],
    )

    print(
        "Log:",
        config["log_file"],
    )

    print("=" * 60)
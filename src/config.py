# src/config.py

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML experiment configuration.
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

    validate_config(config)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the minimum fields required by the
    image-editing experiment runner.
    """

    required_top_level = [
        "experiment_name",
        "model",
        "generation",
        "seed",
        "output_dir",
        "log_file",
    ]

    for key in required_top_level:
        if key not in config:
            raise ValueError(
                f"Missing required config field: {key}"
            )

    required_model_fields = [
        "model_id",
        "dtype",
        "cpu_offload",
    ]

    for key in required_model_fields:
        if key not in config["model"]:
            raise ValueError(
                f"Missing model config field: {key}"
            )

    required_generation_fields = [
        "num_inference_steps",
        "true_cfg_scale",
        "enable_thinking_mode",
        "enable_reflection_mode",
    ]

    for key in required_generation_fields:
        if key not in config["generation"]:
            raise ValueError(
                f"Missing generation config field: {key}"
            )

    if config["generation"]["num_inference_steps"] <= 0:
        raise ValueError(
            "num_inference_steps must be > 0"
        )

    if config["generation"]["true_cfg_scale"] < 0:
        raise ValueError(
            "true_cfg_scale must be >= 0"
        )


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
    Validate the minimum fields required for the
    RQ1 Step1X image-editing experiments.
    """

    # ---------------------------------------------------------
    # Top-level fields
    # ---------------------------------------------------------

    required_top_level = [
        "experiment_name",
        "model",
        "generation",
        "output_dir",
        "log_file",
    ]

    for key in required_top_level:
        if key not in config:
            raise ValueError(
                f"Missing required config field: {key}"
            )

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------

    required_model_fields = [
        "model_id",
        "dtype",
        "cpu_offload",
    ]

    for key in required_model_fields:
        if key not in config["model"]:
            raise ValueError(
                f"Missing model config field: {key}"
            )

    valid_dtypes = {
        "bfloat16",
        "float16",
        "float32",
    }

    if config["model"]["dtype"] not in valid_dtypes:
        raise ValueError(
            "model.dtype must be one of: "
            "bfloat16, float16, float32"
        )

    if not isinstance(
        config["model"]["cpu_offload"],
        bool,
    ):
        raise ValueError(
            "model.cpu_offload must be true or false"
        )

    # ---------------------------------------------------------
    # Generation configuration
    # ---------------------------------------------------------

    required_generation_fields = [
        "num_inference_steps",
        "true_cfg_scale",
        "enable_thinking_mode",
        "enable_reflection_mode",
        "max_try_cnt",
    ]

    for key in required_generation_fields:
        if key not in config["generation"]:
            raise ValueError(
                f"Missing generation config field: {key}"
            )

    num_steps = config["generation"][
        "num_inference_steps"
    ]

    cfg_scale = config["generation"][
        "true_cfg_scale"
    ]

    thinking_enabled = config["generation"][
        "enable_thinking_mode"
    ]

    reflection_enabled = config["generation"][
        "enable_reflection_mode"
    ]

    max_try_cnt = config["generation"][
        "max_try_cnt"
    ]

    if not isinstance(num_steps, int):
        raise ValueError(
            "num_inference_steps must be an integer"
        )

    if num_steps <= 0:
        raise ValueError(
            "num_inference_steps must be > 0"
        )

    if not isinstance(
        cfg_scale,
        (int, float),
    ):
        raise ValueError(
            "true_cfg_scale must be numeric"
        )

    if cfg_scale < 0:
        raise ValueError(
            "true_cfg_scale must be >= 0"
        )

    if not isinstance(
        thinking_enabled,
        bool,
    ):
        raise ValueError(
            "enable_thinking_mode must be true or false"
        )

    if not isinstance(
        reflection_enabled,
        bool,
    ):
        raise ValueError(
            "enable_reflection_mode must be true or false"
        )

    if not isinstance(max_try_cnt, int):
        raise ValueError(
            "max_try_cnt must be an integer"
        )

    if max_try_cnt < 1:
        raise ValueError(
            "max_try_cnt must be >= 1"
        )

    # ---------------------------------------------------------
    # RQ1 consistency checks
    # ---------------------------------------------------------

    # Direct/Base and Thinking-only experiments should
    # generate one attempt only.
    if (
        not reflection_enabled
        and max_try_cnt != 1
    ):
        raise ValueError(
            "When reflection is disabled, "
            "max_try_cnt must be 1 for RQ1."
        )

    # Correction experiment uses one possible retry:
    # initial generation + one correction = 2 attempts.
    if (
        reflection_enabled
        and max_try_cnt != 2
    ):
        raise ValueError(
            "For the RQ1 correction condition, "
            "reflection is enabled and max_try_cnt "
            "must be 2."
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

    print(
        "Steps:",
        config["generation"]["num_inference_steps"],
    )

    print(
        "CFG:",
        config["generation"]["true_cfg_scale"],
    )

    print(
        "Thinking:",
        config["generation"][
            "enable_thinking_mode"
        ],
    )

    print(
        "Reflection:",
        config["generation"][
            "enable_reflection_mode"
        ],
    )

    print(
        "Max attempts:",
        config["generation"]["max_try_cnt"],
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
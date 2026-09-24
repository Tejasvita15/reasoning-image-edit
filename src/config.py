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


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a compact experiment summary before execution.
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
        config["generation"]["enable_thinking_mode"],
    )

    print(
        "Reflection:",
        config["generation"]["enable_reflection_mode"],
    )

    print(
        "Seed:",
        config["seed"],
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
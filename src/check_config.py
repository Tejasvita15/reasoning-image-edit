from config import (
    load_config,
    print_config_summary,
)


for config_path in [
    "configs/base.yaml",
    "configs/thinking.yaml",
    "configs/thinking_reflection.yaml",
]:

    config = load_config(config_path)

    print_config_summary(config)
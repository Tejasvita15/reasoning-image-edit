# src/seeding.py

import hashlib
from typing import Dict

import torch


def derive_seed(
    base_seed: int,
    example_uid: str,
    stage: str,
) -> int:
    """
    Derive a deterministic stage-specific seed.

    The seed depends on:
    - base experiment seed
    - example UID
    - stage name

    This avoids relying on Python's built-in hash(),
    which is not stable across processes.
    """

    key = (
        f"{base_seed}|"
        f"{example_uid}|"
        f"{stage}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        key
    ).digest()

    # Keep the result inside a safe integer range
    # for PyTorch manual_seed().
    seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return seed % (2**31 - 1)


def set_global_seed(
    seed: int,
) -> None:
    """
    Set the global PyTorch RNG.

    This is important for Step1X thinking because the
    thinker uses the global torch RNG rather than the
    diffusion generator.
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rq1_stage_seeds(
    *,
    base_seed: int,
    example_uid: str,
    max_try_cnt: int,
) -> Dict[str, int]:
    """
    Generate deterministic seeds needed by RQ1.

    Always includes:
        thinking
        diffusion_0

    When correction is enabled with max_try_cnt=2,
    also prepare:
        reflection_1
        diffusion_1

    Note:
    diffusion_1 is only a prepared deterministic seed.
    Whether Step1X's internal retry can directly consume it
    depends on the pipeline implementation and will be checked
    before the RQ1.2 correction run.
    """

    if max_try_cnt < 1:
        raise ValueError(
            "max_try_cnt must be >= 1"
        )

    seeds = {
        "thinking": derive_seed(
            base_seed=base_seed,
            example_uid=example_uid,
            stage="thinking",
        ),

        "diffusion_0": derive_seed(
            base_seed=base_seed,
            example_uid=example_uid,
            stage="diffusion_0",
        ),
    }

    # RQ1 correction condition:
    # initial generation + at most one retry.
    if max_try_cnt > 1:

        seeds["reflection_1"] = derive_seed(
            base_seed=base_seed,
            example_uid=example_uid,
            stage="reflection_1",
        )

        seeds["diffusion_1"] = derive_seed(
            base_seed=base_seed,
            example_uid=example_uid,
            stage="diffusion_1",
        )

    return seeds
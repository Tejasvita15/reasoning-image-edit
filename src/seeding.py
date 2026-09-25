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
    Derive deterministic seeds for every stochastic
    stage of one Step1X RQ1 generation.

    For each attempt:
        diffusion_i
        reflection_i

    The initial instruction rewrite has its own
    independent thinking seed.
    """

    if max_try_cnt < 1:
        raise ValueError(
            "max_try_cnt must be >= 1."
        )

    seeds = {
        "thinking": derive_seed(
            base_seed,
            example_uid,
            "thinking",
        )
    }

    for attempt_idx in range(
        max_try_cnt
    ):

        seeds[
            f"diffusion_{attempt_idx}"
        ] = derive_seed(
            base_seed,
            example_uid,
            f"diffusion_{attempt_idx}",
        )

        seeds[
            f"reflection_{attempt_idx}"
        ] = derive_seed(
            base_seed,
            example_uid,
            f"reflection_{attempt_idx}",
        )

    return seeds
# src/rq2/common.py

"""
Shared helpers for the Step1X RQ2 experiments.

All RQ2 code lives in src/rq2/. It imports the RQ1 modules
(config, dataset, logger, seeding, step1x_runner) read-only and
never modifies them.

Heavy dependencies (torch, datasets, diffusers) are imported inside
the functions that need them, so the pure helpers here can be tested
without a GPU environment.
"""

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.taxonomy import get_knowledge_type


KNOWLEDGE_TYPE_ORDER = (
    "factual",
    "conceptual",
    "procedural",
)

# off:    raw instruction, pipeline thinking off.
# cached: replay (S4). The rewrite cached by build_thought_cache is the
#         prompt and pipeline thinking stays off, so every guidance value
#         and every arm sees the identical thought.
# live:   pipeline thinks internally. Only used to check that replay
#         reproduces native thinking-on.
THINKING_MODES = (
    "off",
    "cached",
    "live",
)

# Step1XRunner passes no negative prompt, so Step1X-v1p2 uses its default
# negative_prompt="" encoded together with the reference image. Neither the
# raw instruction nor the cached rewrite is in the unconditional branch, so
# true_cfg_scale scales instruction + rewrite jointly (RQ2.1 caveat).
UNCONDITIONAL_BRANCH = (
    "reference image + empty text (Step1X-v1p2 default negative_prompt=''); "
    "instruction and rewrite are both absent"
)


# =============================================================
# Determinism (S7)
# =============================================================

def enable_determinism() -> Dict[str, Any]:
    """
    Request deterministic kernels before any CUDA work.

    CUBLAS_WORKSPACE_CONFIG must be set before the first cuBLAS call.
    jobs/rq2_step1x.sh exports it; setdefault covers direct runs.

    warn_only=True keeps ops without a deterministic kernel from
    crashing. The S7 pixel check decides whether the result is
    reproducible enough.
    """

    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )

    import torch

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    torch.backends.cudnn.benchmark = False

    return {
        "use_deterministic_algorithms": True,
        "warn_only": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config":
            os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


# =============================================================
# Examples
# =============================================================

def uid_knowledge_type(
    example_uid: str,
) -> str:
    """
    Knowledge type of a "category__id" UID.
    """

    category = example_uid.rsplit("__", 1)[0]

    return get_knowledge_type(category)


def stratified_subset(
    uids: Sequence[str],
    n_examples: int,
) -> List[str]:
    """
    Deterministically pick n_examples UIDs, round-robin over the
    knowledge types in sorted-UID order.

    Only used for smoke and determinism-check subsets. Experiments
    always run the full frozen split.
    """

    if not 0 < n_examples <= len(uids):
        raise ValueError(
            f"n_examples must be in [1, {len(uids)}], "
            f"got {n_examples}."
        )

    groups = {
        knowledge_type: []
        for knowledge_type in KNOWLEDGE_TYPE_ORDER
    }

    for uid in sorted(uids):
        groups[
            uid_knowledge_type(uid)
        ].append(uid)

    selected = []

    while len(selected) < n_examples:
        for knowledge_type in KNOWLEDGE_TYPE_ORDER:
            if (
                groups[knowledge_type]
                and len(selected) < n_examples
            ):
                selected.append(
                    groups[knowledge_type].pop(0)
                )

    return sorted(selected)


def load_rq2_examples(
    splits_path: str,
    split_name: str,
    n_examples: Optional[int] = None,
):
    """
    Load one frozen split (optionally a stratified subset) with lazy
    image decoding, using the shared RQ1 dataset code.
    """

    from src.dataset import (
        load_kris_bench,
        make_example_uid,
        select_split_examples,
    )

    dataset = select_split_examples(
        dataset=load_kris_bench(),
        splits_path=splits_path,
        split_name=split_name,
    )

    if n_examples is None:
        return dataset

    # Column access reads metadata only; images stay encoded.
    uids = [
        make_example_uid(str(category), str(example_id))
        for category, example_id in zip(
            dataset["category"],
            dataset["id"],
        )
    ]

    keep = set(
        stratified_subset(uids, n_examples)
    )

    return dataset.select(
        [
            idx
            for idx, uid in enumerate(uids)
            if uid in keep
        ]
    )


# =============================================================
# Naming
# =============================================================

def condition_name(
    experiment_name: str,
    thinking: str,
    true_cfg_scale: float,
    num_inference_steps: int,
    run_label: Optional[str] = None,
) -> str:
    """
    Unique name of one grid point. It is the logged condition, the
    output sub-directory and the log file stem, so resume never mixes
    grid points.

    Example: rq2_step1x_think-cached_cfg6_steps50
    """

    if thinking not in THINKING_MODES:
        raise ValueError(
            f"Unknown thinking mode: {thinking}"
        )

    name = (
        f"{experiment_name}"
        f"_think-{thinking}"
        f"_cfg{float(true_cfg_scale):g}"
        f"_steps{int(num_inference_steps)}"
    )

    if run_label:
        name += f"_{run_label}"

    return name


def offload_mode(
    config: Dict[str, Any],
) -> str:
    """
    Human-readable offload mode for timing tables (S5).
    """

    if config["model"]["cpu_offload"]:
        return "model_cpu_offload"

    return "none_all_on_gpu"


def fix_text_encoder_offload(
    pipe,
) -> None:
    """
    Make Step1X-v1p2's text encoder work under enable_model_cpu_offload().

    Accelerate onloads a model only when its top-level forward() runs.
    Step1X uses the text encoder without that in two places:
      - the thinker (think / reflect) builds its inputs on
        text_encoder.device, which is the CPU while offloaded, then calls
        text_encoder.generate()
      - pipeline prompt encoding calls text_encoder.model.language_model
        .embed_tokens and .visual directly, before text_encoder(...)

    Both paths now onload the encoder to the execution device first. The
    offload chain (text_encoder -> transformer -> vae) still evicts it when
    the transformer runs. No RNG is touched. No-op without offload.
    """

    text_encoder = pipe.text_encoder

    if not hasattr(text_encoder, "_hf_hook"):
        return

    from accelerate.utils import send_to_device

    device = pipe._execution_device

    def onload():
        if text_encoder.device != device:
            text_encoder.to(device)

    original_generate = text_encoder.generate

    def generate(*args, **kwargs):
        onload()
        return original_generate(
            *send_to_device(args, device),
            **send_to_device(kwargs, device),
        )

    text_encoder.generate = generate

    text_encoder.model.language_model.embed_tokens.register_forward_pre_hook(
        lambda module, args: onload()
    )


# =============================================================
# Thought cache
# =============================================================

def file_sha256(
    path: str,
) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_thought_cache(
    path: str,
    *,
    expected_uids: Sequence[str],
    base_seed: int,
    model_id: str,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Load a thought cache and check that it covers every example and
    was built with the same base seed and model.

    Returns (cache keyed by example_uid, sha256 of the cache file).
    """

    cache_path = Path(path)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Thought cache not found: {cache_path}. "
            "Build it first with src.rq2.build_thought_cache."
        )

    cache = {}

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            cache[record["example_uid"]] = record

    missing = sorted(
        set(expected_uids) - set(cache)
    )

    if missing:
        raise RuntimeError(
            f"{len(missing)} examples missing from "
            f"{cache_path}. First: {missing[:5]}"
        )

    for uid in expected_uids:
        record = cache[uid]

        if record["base_seed"] != base_seed:
            raise RuntimeError(
                f"{uid}: cache base_seed "
                f"{record['base_seed']} != {base_seed}"
            )

        if record["model_id"] != model_id:
            raise RuntimeError(
                f"{uid}: cache model_id "
                f"{record['model_id']} != {model_id}"
            )

    return cache, file_sha256(str(cache_path))


def count_tokens(
    tokenizer,
    text: Optional[str],
) -> int:
    """
    Token count of a decoded rewrite, without special tokens.
    """

    if not text:
        return 0

    return len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )


# =============================================================
# Provenance
# =============================================================

def _git(
    args: List[str],
    cwd: Optional[Path] = None,
) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )

    except (OSError, subprocess.CalledProcessError):
        return None

    return completed.stdout.strip()


def collect_env() -> Dict[str, Any]:
    """
    Versions and code revisions logged with every row.

    repo_dirty ignores untracked files; commit src/rq2 before real
    runs so repo_commit identifies the exact code.
    """

    import diffusers
    import torch
    import transformers

    diffusers_dir = Path(
        diffusers.__file__
    ).resolve().parent

    repo_status = _git(
        ["status", "--porcelain", "--untracked-files=no"]
    )

    return {
        "repo_commit":
            _git(["rev-parse", "HEAD"]),

        "repo_dirty":
            None if repo_status is None else bool(repo_status),

        "diffusers_version":
            diffusers.__version__,

        "diffusers_path":
            str(diffusers_dir),

        # None when diffusers is not installed from a git checkout.
        "diffusers_commit":
            _git(["rev-parse", "HEAD"], cwd=diffusers_dir),

        "transformers_version":
            transformers.__version__,

        "torch_version":
            torch.__version__,

        "cuda_version":
            torch.version.cuda,

        "gpu":
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,

        "hostname":
            platform.node(),

        "slurm_job_id":
            os.environ.get("SLURM_JOB_ID"),
    }

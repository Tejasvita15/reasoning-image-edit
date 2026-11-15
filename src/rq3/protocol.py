"""CPU-only validation and manifests for the standalone RQ3 experiment.

Paths in YAML are relative to the repository, never the shell's working directory.
No dataset, model, torch, or judge API is loaded when planning an experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.taxonomy import get_knowledge_type, is_rq1_category_allowed


REPO_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = {
    "A": {"thinking": False, "regione": False, "rikv": False, "avd": False},
    "B": {"thinking": True, "regione": False, "rikv": False, "avd": False},
    "C": {"thinking": False, "regione": True, "rikv": True, "avd": True},
    "D": {"thinking": True, "regione": True, "rikv": True, "avd": True},
    "E": {"thinking": True, "regione": True, "rikv": False, "avd": True},
    "F": {"thinking": True, "regione": True, "rikv": True, "avd": False},
}
KNOWLEDGE_TYPES = ("factual", "conceptual", "procedural")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def derive_seed(base_seed: int, example_uid: str, stage: str) -> int:
    """Exactly src.seeding.derive_seed, without importing torch for planning."""
    digest = hashlib.sha256(f"{base_seed}|{example_uid}|{stage}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


def _resolve(path: str | Path, root: Path = REPO_ROOT) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _require_equal(value: Any, expected: Any, field: str) -> None:
    if value != expected or isinstance(expected, bool) and not isinstance(value, bool):
        raise ValueError(f"RQ3 requires {field}={expected!r}; received {value!r}")


def _uids(values: Any, field: str) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError(f"{field} must be a list of category__id strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate IDs")
    for uid in values:
        category, separator, example_id = uid.rpartition("__")
        if not separator or not example_id or not is_rq1_category_allowed(category):
            raise ValueError(f"{field} contains an invalid or excluded example: {uid}")
    return values


def _counts(values: list[str]) -> dict[str, int]:
    return dict(Counter(get_knowledge_type(uid.rsplit("__", 1)[0]) for uid in values))


def validate_splits(splits: dict, subset: dict) -> None:
    expected_counts = {
        "k_dev": {"factual": 30, "conceptual": 30, "procedural": 30},
        "k_test": {"factual": 100, "conceptual": 100, "procedural": 100},
        "k_small": {"factual": 14, "conceptual": 13, "procedural": 13},
    }
    for name, counts in expected_counts.items():
        values = _uids(splits.get(name), name)
        if _counts(values) != counts:
            raise ValueError(f"{name} knowledge-type counts must be {counts}, got {_counts(values)}")
    if set(splits["k_dev"]) & set(splits["k_test"]):
        raise ValueError("K-dev must be disjoint from K-test")
    if not set(splits["k_small"]) <= set(splits["k_test"]):
        raise ValueError("K-small must be a subset of K-test")
    _require_equal(subset.get("parent_split"), "k_small", "subset.parent_split")
    ablation = _uids(subset.get("example_uids"), "RQ3.2 subset")
    if not set(ablation) <= set(splits["k_small"]):
        raise ValueError("RQ3.2 subset must be contained in K-small")
    expected_ablation = {"factual": 7, "conceptual": 7, "procedural": 6}
    if _counts(ablation) != expected_ablation or subset.get("per_knowledge_type") != expected_ablation:
        raise ValueError(f"RQ3.2 subset must have frozen stratification {expected_ablation}")


def validate_config(config: dict) -> None:
    """Reject changes that invalidate the predeclared RQ3 comparison."""
    expected = {
        ("schema_version",): 1,
        ("model", "model_id"): "stepfun-ai/Step1X-Edit-v1p2",
        ("model", "dtype"): "bfloat16",
        ("data", "dataset_id"): "Liang0223/KRIS_Bench",
        ("data", "split"): "train",
        ("generation", "num_inference_steps"): 28,
        ("generation", "true_cfg_scale"): 6.0,
        ("generation", "guidance_scale"): 6.0,
        ("generation", "negative_prompt"): "",
        ("generation", "enable_reflection_mode"): False,
        ("generation", "max_try_cnt"): 1,
        ("preprocessing", "mode"): "step1x_native",
        ("preprocessing", "size_level"): 1024,
        ("preprocessing", "color_mode"): "RGB",
        ("preprocessing", "exif_transpose"): True,
        ("thinking", "temperature"): 0.7,
        ("thinking", "top_p"): 0.9,
        ("thinking", "max_new_tokens"): 128,
        ("seeds", "default"): 42,
        ("seeds", "additional"): [1234, 7],
        ("seeds", "derivation"): "sha256_first8_mod_2^31_minus1",
        ("seeds", "stages"): ["thinking", "diffusion_0"],
        ("regione", "no_rikv_mode"): "no_contextual_injection",
        ("statistics", "bootstrap_resamples"): 10000,
        ("statistics", "confidence"): 0.95,
        ("timing", "synchronize_cuda"): True,
        ("validation", "deterministic_algorithms"): True,
        ("validation", "determinism_examples"): 10,
    }
    defaults = {"num_inference_steps": 28, "warmup_step": 6, "post_step": 2,
                "refresh_step": "16", "threshold": 0.88, "cache_threshold": 0.02,
                "erosion_dilation": True}
    for path, required in expected.items():
        value: Any = config
        for part in path:
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"Missing required protocol field: {'.'.join(path)}")
            value = value[part]
        _require_equal(value, required, ".".join(path))
    _require_equal(config["regione"].get("params"), defaults, "regione.params")
    if config["model"].get("offload_mode") not in {"model", "none"}:
        raise ValueError("model.offload_mode must be model or none; sequential offloading is excluded")
    if config["regione"].get("no_avd_cache_threshold", 0) >= 0:
        raise ValueError("AVDCache ablation requires a negative cache threshold")
    if config["timing"].get("repeats", 0) < 3 or config["timing"].get("warmup_generations", 0) < 1:
        raise ValueError("Timing needs at least one warmup generation and three repeats")
    stats = config["statistics"]
    delta = stats.get("equivalence_margin")
    mmd = stats.get("minimum_meaningful_difference")
    if delta is not None and (type(delta) not in (int, float) or not math.isfinite(delta) or delta <= 0):
        raise ValueError("equivalence_margin must be positive or null before calibration")
    if mmd is not None and (type(mmd) not in (int, float) or not math.isfinite(mmd) or mmd < 0):
        raise ValueError("minimum_meaningful_difference must be nonnegative or null before calibration")
    if delta is not None and mmd is not None and delta < mmd:
        raise ValueError("equivalence_margin must be at least minimum_meaningful_difference")


def load_protocol(config_path: str | Path = "configs/rq3.yaml") -> dict:
    path = _resolve(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Protocol must be a YAML mapping: {path}")
    validate_config(config)
    paths = {"config": str(path)}
    for name, value in {
        "splits": config["data"]["splits_path"],
        "subset": config["data"]["subset_path"],
        "judge": config["judge"]["config_path"],
        "diffusers": config["source"]["diffusers_path"],
        "regione": config["source"]["regione_path"],
        "output": config["output_dir"],
    }.items():
        paths[name] = str(_resolve(value))
    # Keep source-key aliases for consumers that address the YAML field names.
    paths["diffusers_path"] = paths["diffusers"]
    paths["regione_path"] = paths["regione"]
    paths["output_dir"] = paths["output"]
    for name in ("splits", "subset"):
        expected = config["data"].get(f"{name}_sha256")
        actual = sha256_file(paths[name])
        if expected != actual:
            raise ValueError(f"Frozen {name} checksum mismatch: expected {expected}, got {actual}")
    splits = json.loads(Path(paths["splits"]).read_text(encoding="utf-8"))
    subset = json.loads(Path(paths["subset"]).read_text(encoding="utf-8"))
    if subset.get("source_splits_sha256") != config["data"]["splits_sha256"]:
        raise ValueError("RQ3.2 subset belongs to a different splits.json")
    validate_splits(splits, subset)
    judge = yaml.safe_load(Path(paths["judge"]).read_text(encoding="utf-8"))
    if not isinstance(judge, dict):
        raise ValueError("Judge configuration must be a YAML mapping")
    # Freeze the local adapter/analysis implementation too. A clean upstream
    # checkout alone cannot detect edits to this experiment's own code.
    implementation_files = sorted((REPO_ROOT / "src" / "rq3").glob("*.py"))
    implementation_files += [REPO_ROOT / name for name in
                             ("src/taxonomy.py", "src/dataset.py", "requirements-rq3.txt", "environment.yml")]
    config["implementation_files_sha256"] = {
        str(file.relative_to(REPO_ROOT)): sha256_file(file) for file in implementation_files
    }
    payload = {"config": config, "splits": splits, "subset": subset, "judge": judge}
    digest = canonical_hash(payload)
    return {**config, "protocol_hash": digest, "_paths": paths,
            "_splits": splits, "_subset": subset, "_judge": judge}


def validate_runtime_protocol(config: dict, *, require_equivalence: bool = False) -> None:
    """Pin checks for real execution; dry-run planning does not need downloads."""
    revisions = {
        "model.revision": config["model"].get("revision"),
        "data.revision": config["data"].get("revision"),
        "source.step1x_revision": config["source"].get("step1x_revision"),
        "source.diffusers_revision": config["source"].get("diffusers_revision"),
        "source.regione_revision": config["source"].get("regione_revision"),
        "judge.official_revision": config["_judge"].get("official_revision"),
    }
    for field, value in revisions.items():
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"Real runs require an immutable 40-character commit for {field}")
    if require_equivalence:
        stats = config["statistics"]
        if stats.get("equivalence_margin") is None or stats.get("minimum_meaningful_difference") is None:
            raise ValueError("Freeze equivalence_margin and minimum_meaningful_difference before test evaluation")


def verify_checkout(path: str | Path, revision: str) -> None:
    """Reject source checkout drift, including modified tracked source files."""
    try:
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ValueError(f"Missing or invalid source checkout: {path}") from exc
    if head != revision:
        raise ValueError(f"Source revision mismatch for {path}: expected {revision}, got {head}")
    if changes.strip():
        raise ValueError(f"Tracked files modified in pinned source checkout: {path}")


def validate_for_execution(config: dict, phase: str = "all") -> None:
    """Check deployment pins and require successful pilots before scaled runs."""
    validate_runtime_protocol(config)
    for name in ("diffusers", "regione"):
        verify_checkout(config["_paths"][name], config["source"][f"{name}_revision"])
    if phase == "pilot":
        return
    if phase not in {"all", "rq3_1", "rq3_2"}:
        raise ValueError(f"Unsupported execution phase: {phase}")
    kinds = ["determinism"] + (["components"] if phase in {"all", "rq3_2"} else [])
    for kind in kinds:
        value = config["validation"].get(f"{kind}_report")
        if not value:
            raise ValueError(f"Configure validation.{kind}_report before test generation")
        path = _resolve(value)
        if not path.exists():
            raise ValueError(f"Missing {kind} development validation report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("protocol_hash") != config["protocol_hash"]
                or report.get("kind") != kind or report.get("passed") is not True):
            raise ValueError(f"A passing {kind} report for this exact protocol is required: {path}")


def pilot_uids(config: dict, count: int = 10) -> list[str]:
    """Balanced K-dev pilot, deterministic and independent of generated outcomes."""
    if not isinstance(count, int) or not 1 <= count <= len(config["_splits"]["k_dev"]):
        raise ValueError("Pilot count must be between 1 and the K-dev size")
    groups = {kind: sorted(uid for uid in config["_splits"]["k_dev"]
                           if get_knowledge_type(uid.rsplit("__", 1)[0]) == kind)
              for kind in KNOWLEDGE_TYPES}
    ordered = [groups[kind][index] for index in range(30) for kind in KNOWLEDGE_TYPES]
    return ordered[:count]


def build_jobs(config: dict, phase: str = "all", seeds: list[int] | None = None,
               k_dev: int = 10) -> list[dict]:
    """Return 160 factorial + 40 additional ablation jobs per seed.

    B and D are reused for the component analysis; they are never regenerated in
    rq3_2. Pilot uses all six conditions on disjoint K-dev examples.
    """
    if phase not in {"all", "rq3_1", "rq3_2", "pilot"}:
        raise ValueError("phase must be all, rq3_1, rq3_2, or pilot")
    seeds = [config["seeds"]["default"]] if seeds is None else seeds
    allowed = {config["seeds"]["default"], *config["seeds"]["additional"]}
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed not in allowed for seed in seeds):
        raise ValueError("Seeds must be a nonempty unique selection of 42, 1234, and 7")
    if phase == "pilot":
        selected = pilot_uids(config, k_dev)
    elif phase == "rq3_2":
        selected = config["_subset"]["example_uids"]
    else:
        selected = config["_splits"]["k_small"]
    subset = set(config["_subset"]["example_uids"])
    jobs = []
    for seed in seeds:
        for uid in selected:
            conditions = list("ABCDEF") if phase == "pilot" else list("EF") if phase == "rq3_2" else list("ABCD")
            if phase == "all" and uid in subset:
                conditions.extend("EF")
            for condition in conditions:
                jobs.append({"example_uid": uid, "condition": condition,
                             "base_seed": seed, "phase": phase,
                             "knowledge_type": get_knowledge_type(uid.rsplit("__", 1)[0]),
                             "stage_seeds": {stage: derive_seed(seed, uid, stage) for stage in config["seeds"]["stages"]},
                             **CONDITIONS[condition]})
    return jobs


def protocol_manifest(config: dict) -> dict:
    """Portable, complete manifest written once beside generation artifacts."""
    return {"protocol_hash": config["protocol_hash"],
            "config": {key: value for key, value in config.items() if not key.startswith("_") and key != "protocol_hash"},
            "splits": config["_splits"], "subset": config["_subset"], "judge": config["_judge"]}


def freeze_protocol(config: dict, output_dir: str | Path) -> Path:
    """Do not mix differing protocols when resuming into the same directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    manifest = protocol_manifest(config)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(f"Output directory has a different frozen protocol: {path}")
        return path
    # Exclusive creation prevents simultaneous launches from overwriting a freeze.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path

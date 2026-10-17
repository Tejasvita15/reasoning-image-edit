"""CPU-only NSCC preparation, immutable source staging and offline data loading.

No model is instantiated. The checkpoint is downloaded to the Hugging Face
cache, and every frozen split is materialized as EXIF-corrected RGB PNGs plus a
verified manifest. Statistics/judge choices do not invalidate unchanged data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SCHEMA_VERSION = 1
SOURCE_COLUMNS = ("image", "image_1", "image_2", "image_3", "image_4")
ALL_IMAGE_COLUMNS = SOURCE_COLUMNS + ("gt_image",)

# Audited against the immutable Hugging Face tree and model_index.json:
# https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2/tree/ca85b97fd19f2235dc0d6fd3633d1319f169e149
# Keep the complete five Diffusers components, including processor/tokenizer
# assets used by the pipeline's internal AutoProcessor reload. Identical blobs
# in processor/ and text_encoder/ are shared by the HF cache automatically.
# This revision has NO separate redundant monolithic checkpoint. Excluding its
# README, repository metadata, examples and artwork avoids 10,159,466 bytes.
MODEL_ID = "stepfun-ai/Step1X-Edit-v1p2"
MODEL_REVISION = "ca85b97fd19f2235dc0d6fd3633d1319f169e149"
MODEL_COMPONENTS = {
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
    "transformer": ["diffusers", "Step1XEditTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
    "processor": ["transformers", "Qwen2_5_VLProcessor"],
}
MODEL_SNAPSHOT_FILES = {
    "model_index.json": 441,
    "processor/added_tokens.json": 605,
    "processor/chat_template.jinja": 1017,
    "processor/merges.txt": 1671853,
    "processor/preprocessor_config.json": 791,
    "processor/special_tokens_map.json": 613,
    "processor/tokenizer.json": 11421896,
    "processor/tokenizer_config.json": 4778,
    "processor/video_preprocessor_config.json": 1811,
    "processor/vocab.json": 2776833,
    "scheduler/scheduler_config.json": 486,
    "text_encoder/added_tokens.json": 605,
    "text_encoder/chat_template.jinja": 1017,
    "text_encoder/config.json": 2521,
    "text_encoder/generation_config.json": 214,
    "text_encoder/merges.txt": 1671853,
    "text_encoder/model-00001-of-00004.safetensors": 4968243304,
    "text_encoder/model-00002-of-00004.safetensors": 4991495816,
    "text_encoder/model-00003-of-00004.safetensors": 4932751040,
    "text_encoder/model-00004-of-00004.safetensors": 1691924384,
    "text_encoder/model.safetensors.index.json": 57619,
    "text_encoder/preprocessor_config.json": 575,
    "text_encoder/special_tokens_map.json": 613,
    "text_encoder/tokenizer.json": 11421896,
    "text_encoder/tokenizer_config.json": 4730,
    "text_encoder/video_preprocessor_config.json": 1730,
    "text_encoder/vocab.json": 2776833,
    "transformer/config.json": 573,
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": 9978548592,
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": 9987127872,
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": 4919434840,
    "transformer/diffusion_pytorch_model.safetensors.index.json": 125448,
    "vae/config.json": 774,
    "vae/diffusion_pytorch_model.safetensors": 335306212,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def preparation_lock(output):
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is preparing {output}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def expected_uids(config):
    return sorted(set().union(*(set(config["_splits"][name]) for name in ("k_dev", "k_test", "k_small")),
                             set(config["_subset"]["example_uids"])))


def data_identity(config):
    from .protocol import REPO_ROOT
    # Only code that defines materialization/decoding belongs in this identity.
    # Freezing a judge margin later must not require downloading 50+ GB again.
    implementation = {name: sha256_file(REPO_ROOT / name)
                      for name in ("src/rq3/prepare.py", "src/dataset.py", "src/taxonomy.py")}
    return {"schema_version": SCHEMA_VERSION,
            "model": {key: config["model"][key] for key in ("model_id", "revision")},
            "model_snapshot_files_sha256": canonical_hash(MODEL_SNAPSHOT_FILES),
            "dataset": {key: config["data"][key] for key in ("dataset_id", "revision", "split")},
            "preprocessing": config["preprocessing"],
            "expected_uids": expected_uids(config), "implementation_sha256": implementation}


def source_specs(config):
    from .protocol import REPO_ROOT
    return [
        {"name": "diffusers", "url": "https://github.com/Peyton-Chen/diffusers.git",
         "path": config["_paths"]["diffusers"], "revision": config["source"]["diffusers_revision"]},
        {"name": "regione", "url": "https://github.com/Peyton-Chen/RegionE.git",
         "path": config["_paths"]["regione"], "revision": config["source"]["regione_revision"]},
        {"name": "kris_evaluator", "url": config["_judge"]["official_repository"].removesuffix(".git") + ".git",
         "path": str(REPO_ROOT / config["_judge"]["official_checkout"]),
         "revision": config["_judge"]["official_revision"]},
    ]


def stage_sources(config):
    from .protocol import verify_checkout
    for spec in source_specs(config):
        path, revision = Path(spec["path"]), spec["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Source {spec['name']} requires an immutable commit")
        if path.exists():
            # Never reset or clean an established checkout. A partial clone
            # without HEAD is intentionally diagnosed instead of overwritten.
            verify_checkout(path, revision)
            print(f"Verified {spec['name']}@{revision}", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".rq3-download")
        if temporary.exists():
            # Resume only our own uncommitted staging directory and same origin.
            remote = subprocess.check_output(["git", "-C", str(temporary), "remote", "get-url", "origin"], text=True).strip()
            if remote != spec["url"]:
                raise ValueError(f"Interrupted source staging has unexpected origin: {temporary}")
        else:
            subprocess.run(["git", "init", str(temporary)], check=True)
            subprocess.run(["git", "-C", str(temporary), "remote", "add", "origin", spec["url"]], check=True)
        subprocess.run(["git", "-C", str(temporary), "fetch", "--depth", "1", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(temporary), "checkout", "--detach", revision], check=True)
        verify_checkout(temporary, revision)
        temporary.rename(path)
        print(f"Staged {spec['name']}@{revision}", flush=True)


def _package_pins(requirements_path):
    pins = {}
    for line in Path(requirements_path).read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"NSCC requirements must contain exact package pins: {line}")
        pins[match[1]] = match[2]
    return pins


def environment_spec(config):
    from .protocol import REPO_ROOT
    requirements = REPO_ROOT / "requirements-rq3.txt"
    return {"schema_version": SCHEMA_VERSION, "python": "3.10",
            "requirements_sha256": sha256_file(requirements), "packages": _package_pins(requirements),
            "torch": os.environ.get("RQ3_TORCH_VERSION", "2.5.1"),
            "torchvision": os.environ.get("RQ3_TORCHVISION_VERSION", "0.20.1"),
            "torch_index_url": os.environ.get("RQ3_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121"),
            "torch_cuda": os.environ.get("RQ3_EXPECTED_CUDA", "12.1"),
            "conda_module": os.environ.get("RQ3_CONDA_MODULE", "miniforge3"),
            "cuda_module": os.environ.get("RQ3_CUDA_MODULE", "cuda/12.2.2"),
            "sources": source_specs(config)}


def installed_environment(spec):
    from .protocol import verify_checkout
    if f"{sys.version_info.major}.{sys.version_info.minor}" != spec["python"]:
        raise ValueError(f"Expected Python {spec['python']}; found {sys.version.split()[0]}")
    installed = {}
    for package, version in {**spec["packages"], "torch": spec["torch"], "torchvision": spec["torchvision"]}.items():
        actual = importlib.metadata.version(package)
        # CUDA wheels add +cu121 to the version's local segment.
        compared = actual.split("+", 1)[0] if package in {"torch", "torchvision"} else actual
        if compared != version:
            raise ValueError(f"Environment drift: {package} expected {version}, installed {actual}")
        installed[package] = actual
    import torch
    if torch.version.cuda != spec["torch_cuda"]:
        raise ValueError(f"Environment has CUDA runtime {torch.version.cuda}, expected {spec['torch_cuda']}")
    for source in spec["sources"]:
        verify_checkout(source["path"], source["revision"])
        if source["name"] == "kris_evaluator":
            continue
        module_name = "RegionE" if source["name"] == "regione" else "diffusers"
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None or module_spec.origin is None or not Path(module_spec.origin).resolve().is_relative_to(Path(source["path"]).resolve()):
            raise ValueError(f"{module_name} is not imported from the pinned editable checkout")
        installed[module_name] = importlib.metadata.version(module_name)
    # Resolve the exact backend entry points on the CPU host. Metadata alone
    # cannot catch a missing indirect dependency or an incompatible import API.
    # These imports define classes/kernels without constructing a model/GPU.
    from diffusers import Step1XEditPipelineV1P2
    from diffusers.models.transformers.transformer_step1x_edit import Step1XEditAttnProcessor
    from diffusers.pipelines.step1x_edit.pipeline_step1x_edit_thinker import Step1XEditThinker
    from RegionE.Step1XEditV1P2 import inplace
    if not all((Step1XEditPipelineV1P2, Step1XEditAttnProcessor, Step1XEditThinker, inplace)):
        raise ValueError("The pinned Step1X/RegionE entry points are unavailable")
    # Freeze transitive dependencies as installed, not just the direct pins.
    # This also makes ad hoc pip upgrades visible to subsequent GPU jobs.
    installed["all_distributions"] = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return installed


def environment_stamp(config, path, action):
    path = Path(path)
    spec = environment_spec(config)
    identity = canonical_hash(spec)
    if action == "check":
        stamp = json.loads(path.read_text(encoding="utf-8"))
        if stamp.get("environment_identity") != identity or stamp.get("prefix") != str(Path(sys.prefix).resolve()):
            raise ValueError("Ready environment no longer matches dependencies/source pins/prefix. Use a fresh prefix and run bootstrap; GPU jobs never reinstall.")
    installed = installed_environment(spec)
    if action == "write":
        if path.exists():
            raise ValueError("Refusing to replace an established environment readiness stamp")
        atomic_json(path, {"environment_identity": identity, "spec": spec, "installed": installed,
                           "prefix": str(Path(sys.prefix).resolve()), "created_at_utc": datetime.now(timezone.utc).isoformat()})
    elif installed != stamp.get("installed"):
        raise ValueError("Installed package versions changed after the environment readiness stamp")
    print(json.dumps({"environment_identity": identity, "prefix": str(Path(sys.prefix).resolve()), "status": "ready"}))


def _rgb_image(encoded):
    from PIL import Image, ImageOps
    if not isinstance(encoded, dict):
        raise TypeError("Dataset image must be encoded with datasets.Image(decode=False)")
    source = BytesIO(encoded["bytes"]) if encoded.get("bytes") is not None else encoded.get("path")
    if source is None:
        raise ValueError("Dataset image has neither bytes nor a path")
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def materialize_dataset(config, output, dataset):
    """Injectable dataset input permits network-free integrity tests."""
    from src.taxonomy import get_knowledge_type, get_reasoning_dimension, is_rq1_category_allowed
    output = Path(output)
    wanted = set(expected_uids(config))
    examples = {}
    total = 0
    categories = {}
    for item in dataset:
        total += 1
        category, example_id = str(item["category"]), str(item["id"])
        categories[category] = categories.get(category, 0) + 1
        uid = f"{category}__{example_id}"
        if uid not in wanted:
            continue
        if uid in examples:
            raise ValueError(f"Duplicate frozen UID in dataset: {uid}")
        if not is_rq1_category_allowed(category):
            raise ValueError(f"Frozen UID uses an excluded category: {uid}")
        source_columns = [name for name in SOURCE_COLUMNS if item.get(name) is not None]
        if len(source_columns) != 1:
            raise ValueError(f"Frozen example must have exactly one source image: {uid}, count={len(source_columns)}")
        if not isinstance(item.get("instruction"), str) or not item["instruction"].strip():
            raise ValueError(f"Frozen example has no instruction: {uid}")
        # File names are content-independent UID hashes, avoiding path traversal
        # and filesystem limitations from dataset-provided category/ID strings.
        filename = hashlib.sha256(uid.encode()).hexdigest() + ".png"
        source_path = Path("images") / filename
        (output / source_path).parent.mkdir(parents=True, exist_ok=True)
        image = _rgb_image(item[source_columns[0]])
        image.save(output / source_path, format="PNG")
        record = {"example_uid": uid, "id": example_id, "category": category,
                  "knowledge_type": get_knowledge_type(category), "reasoning_dimension": get_reasoning_dimension(category),
                  "instruction": item["instruction"], "explanation": item.get("explanation"), "num_source_images": 1,
                  "source_path": source_path.as_posix(), "source_sha256": sha256_file(output / source_path),
                  "source_size": list(image.size), "gt_path": None, "gt_sha256": None}
        if item.get("gt_image") is not None:
            gt_path = Path("ground_truth") / filename
            (output / gt_path).parent.mkdir(parents=True, exist_ok=True)
            _rgb_image(item["gt_image"]).save(output / gt_path, format="PNG")
            record.update(gt_path=gt_path.as_posix(), gt_sha256=sha256_file(output / gt_path))
        examples[uid] = record
    if set(examples) != wanted:
        raise ValueError(f"Pinned dataset is missing frozen IDs: {sorted(wanted-set(examples))[:10]}")
    return {"examples": examples, "dataset_rows": total, "category_counts": categories}


def _safe_file(output, relative):
    path = (Path(output) / relative).resolve()
    if not path.is_relative_to(Path(output).resolve()):
        raise ValueError("Prepared manifest contains a path outside its data directory")
    return path


def load_prepared_examples(config, directory, uids):
    """Load validated cached examples in the existing runner's encoded schema."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = data_identity(config)
    if manifest.get("data_identity") != canonical_hash(identity) or manifest.get("identity") != identity:
        raise ValueError("Prepared dataset identity differs from pinned model/dataset/preprocessing/implementation; rerun bootstrap --prepare into a fresh directory")
    examples = manifest.get("examples", {})
    if manifest.get("examples_sha256") != canonical_hash(examples):
        raise ValueError("Prepared example metadata changed after materialization")
    if set(examples) != set(expected_uids(config)):
        raise ValueError("Prepared manifest does not contain exactly all frozen split IDs")
    requested = set(uids)
    if not requested <= examples.keys():
        raise ValueError(f"Requested UIDs are absent from prepared data: {sorted(requested-examples.keys())}")
    result = {}
    for uid in sorted(requested):
        record = examples[uid]
        if record.get("example_uid") != uid or record.get("num_source_images") != 1:
            raise ValueError(f"Invalid prepared example metadata: {uid}")
        prepared = {key: record[key] for key in ("example_uid", "id", "category", "knowledge_type", "reasoning_dimension", "instruction", "explanation", "num_source_images")}
        prepared.update({name: None for name in ALL_IMAGE_COLUMNS})
        for key, column in (("source", "image"), ("gt", "gt_image")):
            relative = record.get(key + "_path")
            if relative is None:
                if key == "source":
                    raise ValueError(f"Prepared example has no source: {uid}")
                continue
            path = _safe_file(directory, relative)
            if not path.is_file() or sha256_file(path) != record[key + "_sha256"]:
                raise ValueError(f"Prepared {key} image is missing or changed: {uid}")
            prepared[column] = {"bytes": None, "path": str(path)}
        result[uid] = prepared
    return result


def _model_inventory(snapshot):
    """Validate the complete audited subset; tolerate unrelated cached files.

    HF local_files_only returns the snapshot directory even if a previous
    partial download omitted required files, so the allow-list alone is not an
    offline completeness check. Stat every required asset against its pinned
    size and verify both weight indexes name exactly the retained shards.
    """
    snapshot = Path(snapshot)
    inventory = {}
    for relative, expected_size in MODEL_SNAPSHOT_FILES.items():
        path = snapshot / relative
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"Required pinned model file is missing, truncated or changed: {relative}; rerun online preparation")
        inventory[relative] = expected_size
    index = json.loads((snapshot / "model_index.json").read_text(encoding="utf-8"))
    components = {key: value for key, value in index.items() if not key.startswith("_")}
    if components != MODEL_COMPONENTS:
        raise ValueError("Pinned model_index.json components differ from the audited download set")
    for relative in ("text_encoder/model.safetensors.index.json", "transformer/diffusion_pytorch_model.safetensors.index.json"):
        weight_index = json.loads((snapshot / relative).read_text(encoding="utf-8"))
        weight_map = weight_index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or not all(isinstance(value, str) for value in weight_map.values()):
            raise ValueError(f"Invalid pinned weight index: {relative}")
        folder = Path(relative).parent.as_posix()
        actual_shards = {f"{folder}/{filename}" for filename in weight_map.values()}
        required_shards = {name for name in MODEL_SNAPSHOT_FILES if name.startswith(folder + "/") and name.endswith(".safetensors")}
        if actual_shards != required_shards:
            raise ValueError(f"Weight index references files outside the audited model snapshot: {relative}")
    return inventory


def download_model_snapshot(model, *, local_files_only=False):
    """Shared preparation/backend entry point for the pinned Diffusers assets."""
    if model.get("model_id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("The model download allow-list was audited for Step1X-v1p2 at the pinned revision; review it before changing the checkpoint")
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(model["model_id"], revision=model["revision"],
                                 allow_patterns=sorted(MODEL_SNAPSHOT_FILES), local_files_only=local_files_only)
    _model_inventory(snapshot)
    return snapshot


def prepare_perceptual_assets():
    """Cache the small CPU LPIPS/AlexNet dependency for later offline analysis."""
    import lpips
    import torch
    from torchvision.models import AlexNet_Weights
    from urllib.parse import urlparse
    # LPIPS package supplies its linear calibration weights. Torchvision's
    # ImageNet AlexNet parameters are the only separate network download.
    network = lpips.LPIPS(net="alex").cpu().eval()
    del network
    filename = Path(urlparse(AlexNet_Weights.IMAGENET1K_V1.url).path).name
    root = Path(torch.hub.get_dir())
    relative = Path("checkpoints") / filename
    path = root / relative
    if not path.is_file():
        raise ValueError("LPIPS initialized but its expected AlexNet checkpoint was not cached")
    return {"network": "alex", "torch_hub_directory": str(root),
            "files": {relative.as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}}}


def check_perceptual_assets(manifest):
    import torch
    assets = manifest.get("perceptual_assets", {})
    if assets.get("network") != "alex" or not assets.get("files"):
        raise ValueError("Prepared manifest has no offline LPIPS/AlexNet assets")
    root = Path(torch.hub.get_dir())
    for relative, expected in assets["files"].items():
        path = _safe_file(root, relative)
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError("Offline LPIPS/AlexNet checkpoint is missing or changed; rerun online preparation into a fresh prepared-data directory")


def check_prepared(config, output):
    examples = load_prepared_examples(config, output, expected_uids(config))
    manifest = json.loads((Path(output) / "manifest.json").read_text(encoding="utf-8"))
    snapshot = download_model_snapshot(config["model"], local_files_only=True)
    if _model_inventory(snapshot) != manifest.get("model_files"):
        raise ValueError("The cached pinned model snapshot is incomplete or changed; rerun online preparation")
    check_perceptual_assets(manifest)
    print(json.dumps({"status": "prepared", "examples": len(examples), "data_identity": manifest["data_identity"],
                      "model_snapshot": snapshot, "model_files": len(manifest["model_files"]),
                      "model_logical_bytes": sum(manifest["model_files"].values())}, indent=2))


def prepare(config, output):
    from datasets import Image, load_dataset
    output = Path(output).resolve()
    with preparation_lock(output):
        if (output / "manifest.json").exists():
            check_prepared(config, output)
            return
        identity = data_identity(config)
        print(f"Downloading pinned checkpoint {config['model']['model_id']}@{config['model']['revision']}", flush=True)
        snapshot = download_model_snapshot(config["model"])
        data = config["data"]
        dataset = load_dataset(data["dataset_id"], revision=data["revision"], split=data["split"])
        for column in ALL_IMAGE_COLUMNS:
            if column in dataset.column_names:
                dataset = dataset.cast_column(column, Image(decode=False))
        records = materialize_dataset(config, output, dataset)
        perceptual_assets = prepare_perceptual_assets()
        model_files = _model_inventory(snapshot)
        if not model_files:
            raise ValueError("The downloaded model snapshot is empty")
        manifest = {"schema_version": SCHEMA_VERSION, "identity": identity, "data_identity": canonical_hash(identity),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_protocol_hash": config["protocol_hash"],
                    "model_snapshot": str(Path(snapshot).resolve()), "model_files": model_files,
                    "perceptual_assets": perceptual_assets, "examples_sha256": canonical_hash(records["examples"]), **records}
        atomic_json(output / "manifest.json", manifest)
        check_prepared(config, output)


def main():
    from .protocol import load_protocol, validate_runtime_protocol
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rq3.yaml")
    parser.add_argument("--output-dir", default=os.environ.get("RQ3_PREPARED_DATA", ".nscc/data/rq3"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="Verify all cached images and checkpoint files without network")
    modes.add_argument("--stage-sources", action="store_true", help="Fetch and verify the three pinned source checkouts")
    modes.add_argument("--environment-stamp", choices=("write", "check"))
    parser.add_argument("--stamp", default=os.environ.get("RQ3_ENV_STAMP"))
    args = parser.parse_args()
    config = load_protocol(args.config)
    validate_runtime_protocol(config)
    if args.stage_sources:
        stage_sources(config)
    elif args.environment_stamp:
        if not args.stamp:
            parser.error("--stamp or RQ3_ENV_STAMP is required for environment validation")
        environment_stamp(config, args.stamp, args.environment_stamp)
    elif args.check_only:
        check_prepared(config, args.output_dir)
    else:
        prepare(config, args.output_dir)


if __name__ == "__main__":
    main()

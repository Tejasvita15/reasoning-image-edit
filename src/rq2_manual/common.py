"""Portable integrity checks for manually judged RQ2 image bundles.

RQ2 counterpart of src/rq3_manual/common.py (origin/rq3_implementation@f232e68).
The judging policy, response template, prompt rendering and bundle checks are
the same. Only the generation identity differs: the successful rows of the
RQ2 generation logs instead of the RQ3 protocol archive.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Official KRIS-Bench evaluator pinned by the project (same as RQ3).
OFFICIAL_REVISION = "cf1475253a13f5e5666826c10d8967e38917468a"
PROMPT_FILE = "utils/prompts.py"

# src/rq3_manual/prompts.py at origin/rq3_implementation@f232e68. The RQ2 copy
# must stay byte-identical so both RQs present the judge the same prompt.
RQ3_PROMPT_RENDERER_SHA256 = "e74cc8b0dbe5b622e8cd95fea91d2cc991d7bd2813d9404872fb3397f01c5259"

# Fields of the generation snapshot that identify the judged RQ2 outputs.
GENERATION_SNAPSHOT_FIELDS = ("experiment", "split", "seed", "splits_sha256", "logs", "outputs")


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_hash(snapshot):
    return canonical_hash({name: snapshot[name] for name in GENERATION_SNAPSHOT_FIELDS})


def adapter_hashes():
    """Freeze the importer, prompt renderer and data-loading code too."""
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [REPO_ROOT / name for name in ("src/dataset.py", "src/taxonomy.py", "src/rq2/common.py")]
    return {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in files}


def make_policy(judge_label, ui_settings, evaluator):
    if not isinstance(judge_label, str) or not judge_label.strip():
        raise ValueError("Record the actual model label selected in ChatGPT")
    if not isinstance(ui_settings, str) or not ui_settings.strip():
        raise ValueError("Record the UI settings used for every judging session")
    from .prompts import PROMPT_VERSION, rubric_hashes
    renderer = file_sha256(Path(__file__).parent / "prompts.py")
    if renderer != RQ3_PROMPT_RENDERER_SHA256:
        raise ValueError("src/rq2_manual/prompts.py must stay identical to the RQ3 prompt renderer")
    return {
        "interface": "chatgpt_manual", "judge_label": judge_label.strip(),
        "ui_settings": ui_settings.strip(), "prompt_version": PROMPT_VERSION,
        "prompt_renderer_sha256": renderer,
        "aggregate": "mean_applicable_normalized", "score_range": [0, 100],
        "component_range": [1, 5], "component_type": "integer",
        "reference_source": "hf_dataset_generation_source",
        "ground_truth_source": "hf_dataset_gt_image",
        "annotation_text_source": "hf_dataset_generation_instruction",
        "official_reference_comparison": "diagnostic_only; no resizing or tolerance-based substitution",
        "official_text_comparison": "diagnostic_only; the HF text used for generation is judged",
        "official_revision": OFFICIAL_REVISION,
        "official_prompt_sha256": evaluator.prompt_sha256,
        "rubric_sha256": rubric_hashes(evaluator),
        "adapter_files_sha256": adapter_hashes(),
        "session_policy": "fresh chat per batch; no earlier scores, memory, custom instructions or tools",
        "model_snapshot": None, "temperature": None,
        "limitation": "Operator-recorded ChatGPT UI model/settings; API snapshot and sampling settings are not controlled.",
    }


def response_template(batch):
    return {"batch_id": batch["batch_id"], "results": [
        {"blind_id": item["blind_id"], "status": "unreadable", "consistency_score": None,
         "instruction_score": None, "quality_score": None, "knowledge_score": None,
         "rationale": ""} for item in batch["items"]]}


def build_prompt(batch, items_by_key, evaluator):
    from .prompts import build_prompt as render
    return render(batch, items_by_key, evaluator)


def bundle_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Bundle files must use relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Bundle path escapes its directory")
    return path


def selection_records(items):
    return [{"item_key": item["item_key"], **{field: item["row"][field] for field in
             ("example_uid", "condition", "base_seed", "output_sha256")}} for item in items]


def load_bundle(path):
    root = Path(path).resolve()
    manifest = json.loads((root / "private" / "manifest.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("schema_version") != 1 or manifest.get("manifest_hash") != canonical_hash(payload):
        raise ValueError("Bundle manifest checksum/version mismatch")
    if manifest.get("kind") not in {"calibration", "main"}:
        raise ValueError("Unknown bundle kind")
    policy = manifest["evaluation_protocol"]
    if (canonical_hash(policy) != manifest["evaluation_protocol_hash"]
            or policy.get("interface") != "chatgpt_manual"
            or policy.get("aggregate") != "mean_applicable_normalized"):
        raise ValueError("Manual evaluation identity mismatch")
    if policy.get("adapter_files_sha256") != adapter_hashes():
        raise ValueError("Manual judging code changed since export; use the archived code revision for this bundle")
    snapshot = manifest["generation_snapshot"]
    if (snapshot.get("experiment") != "rq2" or generation_hash(snapshot) != manifest["generation_protocol_hash"]
            or snapshot["protocol_hash"] != manifest["generation_protocol_hash"]):
        raise ValueError("Archived generation snapshot checksum mismatch")
    items = manifest["items"]
    index = {item["item_key"]: item for item in items}
    if not items or len(index) != len(items):
        raise ValueError("Duplicate or missing bundle items")
    if canonical_hash(selection_records(items)) != manifest["selection_hash"]:
        raise ValueError("Selected outputs checksum mismatch")
    for item in items:
        if item["row"]["generation_protocol_hash"] != manifest["generation_protocol_hash"]:
            raise ValueError("Generation row differs from the archived generation logs")
    files = manifest["files"]
    for relative, digest in files.items():
        if file_sha256(bundle_path(root, relative)) != digest:
            raise ValueError(f"Bundle artifact changed: {relative}")
    batch_ids, aliases, seen = set(), set(), {}
    for batch in manifest["batches"]:
        if batch["batch_id"] in batch_ids:
            raise ValueError("Duplicate batch ID")
        batch_ids.add(batch["batch_id"])
        if batch["prompt_path"] not in files:
            raise ValueError("Prompt missing from bundle checksums")
        pass_items = seen.setdefault(batch["pass"], [])
        for entry in batch["items"]:
            if entry["blind_id"] in aliases:
                raise ValueError("Duplicate blind ID")
            aliases.add(entry["blind_id"])
            pass_items.append(entry["item_key"])
            row = index[entry["item_key"]]["row"]
            for field, digest_field in (("source_file", "source_sha256"), ("edited_file", "output_sha256")):
                if files.get(entry[field]) != row[digest_field]:
                    raise ValueError("Uploaded image differs from generation record")
            item = index[entry["item_key"]]
            if item["official_family"] == "view":
                if (not item.get("ground_truth_sha256") or
                        files.get(entry.get("ground_truth_file")) != item["ground_truth_sha256"]):
                    raise ValueError("Viewpoint reference image is missing or changed")
    passes = {1, 2} if manifest["kind"] == "calibration" else {1}
    if set(seen) != passes or any(len(values) != len(index) or set(values) != set(index) for values in seen.values()):
        raise ValueError("Every pass must cover exactly the frozen outputs")
    return manifest

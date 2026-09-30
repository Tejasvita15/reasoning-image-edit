"""Portable integrity checks for manually judged RQ3 image bundles."""
from __future__ import annotations

import json
from pathlib import Path

from src.rq3.evaluate import file_sha256
from src.rq3.protocol import REPO_ROOT, canonical_hash


def adapter_hashes():
    """Freeze the importer, prompt renderer and statistical implementation too."""
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [REPO_ROOT / name for name in (
        "src/rq3/evaluate.py", "src/rq3/analyze.py", "src/rq3/protocol.py", "src/rq3/prepare.py",
        "src/dataset.py", "src/taxonomy.py")]
    return {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in files}


def make_policy(judge_label, ui_settings, evaluator):
    if not isinstance(judge_label, str) or not judge_label.strip():
        raise ValueError("Record the actual model label selected in ChatGPT")
    if not isinstance(ui_settings, str) or not ui_settings.strip():
        raise ValueError("Record the UI settings used for every judging session")
    from .prompts import PROMPT_VERSION, rubric_hashes
    return {
        "interface": "chatgpt_manual", "judge_label": judge_label.strip(),
        "ui_settings": ui_settings.strip(), "prompt_version": PROMPT_VERSION,
        "aggregate": "mean_applicable_normalized", "score_range": [0, 100],
        "component_range": [1, 5], "component_type": "integer",
        "reference_source": "pinned_prepared_hf_generation_source",
        "ground_truth_source": "pinned_prepared_hf_gt_image",
        "official_reference_comparison": "diagnostic_only; no resizing or tolerance-based substitution",
        "official_revision": "cf1475253a13f5e5666826c10d8967e38917468a",
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
    generation_hash = canonical_hash({name: snapshot[name] for name in ("config", "splits", "subset", "judge")})
    if generation_hash != manifest["generation_protocol_hash"] or snapshot["protocol_hash"] != generation_hash:
        raise ValueError("Archived generation protocol checksum mismatch")
    items = manifest["items"]
    index = {item["item_key"]: item for item in items}
    if not items or len(index) != len(items):
        raise ValueError("Duplicate or missing bundle items")
    if canonical_hash(selection_records(items)) != manifest["selection_hash"]:
        raise ValueError("Selected outputs checksum mismatch")
    for item in items:
        row = item["row"]
        if row["protocol_hash"] != generation_hash or row["config"] != {**snapshot["config"], "protocol_hash": generation_hash}:
            raise ValueError("Generation row differs from archived configuration")
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

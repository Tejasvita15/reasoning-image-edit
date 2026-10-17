"""Anchor manual judging to the pinned HF images actually used for generation.

The GitHub evaluator supplies annotations and rubrics. Its image copies are
compared and recorded, but the frozen, verified HF record is the reference for
this experiment. Differences never receive a guessed cause or a tolerance.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

from src.rq3.evaluate import OfficialEvaluator, file_sha256
from src.rq3.prepare import load_prepared_examples
from src.rq3.protocol import REPO_ROOT, canonical_hash
from src.taxonomy import get_knowledge_type, get_reasoning_dimension


DATA_IMPLEMENTATIONS = ("src/rq3/prepare.py", "src/dataset.py", "src/taxonomy.py")


def _rgb(path):
    with Image.open(path) as handle:
        return ImageOps.exif_transpose(handle).convert("RGB")


def _compare(prepared_path, official_path, prepared_hash, official_hash):
    prepared, official = _rgb(prepared_path), _rgb(official_path)
    prepared_pixels, official_pixels = prepared.tobytes(), official.tobytes()
    same_size = prepared.size == official.size
    mae, maximum = None, None
    if same_size:
        difference = ImageChops.difference(prepared, official)
        histogram = difference.histogram()
        mae = sum((index % 256) * count for index, count in enumerate(histogram)) / (prepared.width * prepared.height * 3)
        maximum = max(high for _, high in difference.getextrema())
    return {
        "prepared_file_sha256": prepared_hash,
        "rgb_sha256": hashlib.sha256(prepared_pixels).hexdigest(), "size": list(prepared.size),
        "official_file_sha256": official_hash,
        "official_rgb_sha256": hashlib.sha256(official_pixels).hexdigest(), "official_size": list(official.size),
        "official_pixels_equal": same_size and prepared_pixels == official_pixels,
        "official_mae": mae, "official_max_abs_difference": maximum,
    }


def _checked_file(path, expected, label):
    path = Path(path)
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} is missing or changed")
    return path


class PreparedReferenceEvaluator(OfficialEvaluator):
    """Keep official text checks and prove the generation's own HF reference."""

    def __init__(self, judge, snapshot, prepared_data, uids):
        if not prepared_data:
            raise ValueError("Manual export requires --prepared-data or RQ3_PREPARED_DATA; no unchecked reference fallback is available")
        if (not isinstance(snapshot, dict)
                or not {"config", "splits", "subset", "judge", "protocol_hash"} <= snapshot.keys()
                or snapshot["protocol_hash"] != canonical_hash({name: snapshot[name] for name in ("config", "splits", "subset", "judge")})):
            raise ValueError("Archived generation protocol snapshot is invalid")
        implementation = snapshot["config"].get("implementation_files_sha256", {})
        for relative in DATA_IMPLEMENTATIONS:
            if implementation.get(relative) != file_sha256(REPO_ROOT / relative):
                raise ValueError(f"Archived data implementation differs from current {relative}; use the original data-preparation code")
        requested = set(uids)
        if not requested or any(not isinstance(uid, str) or not uid for uid in requested):
            raise ValueError("Select valid frozen example IDs before validating references")
        self.prepared_directory = Path(prepared_data).expanduser().resolve()
        self._manifest_path = self.prepared_directory / "manifest.json"
        if not self._manifest_path.is_file():
            raise ValueError("Prepared HF dataset manifest is missing; use the existing staged RQ3 data directory")
        raw_manifest = self._manifest_path.read_bytes()
        self._manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
        manifest = json.loads(raw_manifest)
        config = {**snapshot["config"], "protocol_hash": snapshot["protocol_hash"],
                  "_splits": snapshot["splits"], "_subset": snapshot["subset"]}
        self._examples = load_prepared_examples(config, self.prepared_directory, requested)
        _checked_file(self._manifest_path, self._manifest_sha256, "Prepared HF manifest")
        self._records = {uid: manifest["examples"][uid] for uid in requested}
        identity = manifest["identity"]
        self.data_provenance = {
            "data_identity": manifest["data_identity"], "manifest_sha256": self._manifest_sha256,
            "dataset": deepcopy(identity["dataset"]), "preprocessing": deepcopy(identity["preprocessing"]),
            "implementation_sha256": deepcopy(identity["implementation_sha256"]),
        }
        self._public_config = {**snapshot["config"], "protocol_hash": snapshot["protocol_hash"]}
        self._cache = {}
        # Preserve the clean pinned checkout/rubric checks, without replacing
        # the strict API evaluator's annotation implementation globally.
        super().__init__(judge)

    def _official_path(self, category, relative, label):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"Official {label} must be one relative image path")
        base = (self.root / "KRIS_Bench" / category).resolve()
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError(f"Official {label} path is missing or outside its category")
        return path

    def _validate(self, row):
        uid = row.get("example_uid")
        if uid not in self._examples:
            raise ValueError("Generation example is not among the verified prepared HF records")
        if (row.get("protocol_hash") != self._public_config["protocol_hash"]
                or row.get("config") != self._public_config or row.get("status") != "ok"):
            raise ValueError("Generation reference has inconsistent frozen protocol/status")
        _checked_file(self._manifest_path, self._manifest_sha256, "Prepared HF manifest")
        category, image_id = uid.rsplit("__", 1)
        example, record = self._examples[uid], self._records[uid]
        if (example.get("example_uid") != uid or example.get("id") != image_id
                or example.get("category") != category or row.get("category") != category
                or example.get("num_source_images") != 1
                or example.get("knowledge_type") != get_knowledge_type(category)
                or example.get("reasoning_dimension") != get_reasoning_dimension(category)
                or row.get("knowledge_type") != example["knowledge_type"]):
            raise ValueError("Generation/prepared HF ID, category or taxonomy differs")
        if row.get("instruction") != example.get("instruction"):
            raise ValueError("Generation instruction differs from the prepared HF record")
        family = self.family(category)
        annotation_path = self._official_path(category, "annotation.json", "annotation")
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        if image_id not in annotations or not isinstance(annotations[image_id], dict):
            raise ValueError("Official annotation is missing for the frozen HF example")
        annotation = annotations[image_id]
        if annotation.get("ins_en") != row["instruction"]:
            raise ValueError(f"HF/official annotation instruction mismatch for {uid}")
        official_source = self._official_path(category, annotation.get("ori_img"), "single-reference original")
        if family == "knowledge":
            prepared_explanation = example.get("explanation")
            official_explanation = annotation.get("explain_en")
            prepared_explanation = "" if prepared_explanation is None else prepared_explanation
            official_explanation = "" if official_explanation is None else official_explanation
            if (not isinstance(prepared_explanation, str) or not isinstance(official_explanation, str)
                    or prepared_explanation != official_explanation):
                raise ValueError(f"HF/official knowledge explanation mismatch for {uid}; review the dataset annotation before judging")
            # An absent explanation is represented uniformly for format().
            annotation = {**annotation, "explain_en": official_explanation}
        for name in ("source", "output"):
            path = row.get(f"{name}_path")
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError("Generation image paths must be absolute")
            _checked_file(path, row.get(f"{name}_sha256"), f"Logged generation {name} image")
        source = Path(row["source_path"])
        prepared_source = _checked_file(example["image"]["path"], record["source_sha256"], "Prepared HF source image")
        official_source_hash = file_sha256(official_source)
        prepared_gt, official_gt, official_gt_hash = None, None, None
        if family == "view":
            if example.get("gt_image") is None or record.get("gt_sha256") is None:
                raise ValueError("Viewpoint judging requires the same frozen HF record's prepared ground-truth image")
            prepared_gt = _checked_file(example["gt_image"]["path"], record["gt_sha256"], "Prepared HF ground-truth image")
            official_gt = self._official_path(category, annotation.get("gt_img"), "ground-truth")
            official_gt_hash = file_sha256(official_gt)
        cache_key = (uid, row["source_sha256"], record["source_sha256"], official_source_hash,
                     record.get("gt_sha256"), official_gt_hash, canonical_hash(annotation))
        if cache_key not in self._cache:
            saved_pixels, prepared_pixels = _rgb(source), _rgb(prepared_source)
            if (saved_pixels.size != prepared_pixels.size
                    or saved_pixels.tobytes() != prepared_pixels.tobytes()):
                raise ValueError(f"Generation source differs from the exact prepared HF image for {uid}")
            provenance = _compare(prepared_source, official_source, record["source_sha256"], official_source_hash)
            provenance["generation_file_sha256"] = row["source_sha256"]
            ground_truth = (_compare(prepared_gt, official_gt, record["gt_sha256"], official_gt_hash)
                            if prepared_gt is not None else None)
            self._cache[cache_key] = {"source": provenance, "ground_truth": ground_truth}
        return (category, image_id, annotation), self._cache[cache_key], prepared_gt

    def annotation(self, row):
        annotation, _, _ = self._validate(row)
        return deepcopy(annotation)

    def reference_provenance(self, row):
        _, provenance, _ = self._validate(row)
        return deepcopy(provenance)

    def ground_truth(self, row):
        _, provenance, path = self._validate(row)
        if path is None:
            raise ValueError("Only viewpoint items have an attached ground-truth reference")
        return path, provenance["ground_truth"]["prepared_file_sha256"]

"""Anchor RQ2 manual judging to the HF inputs actually used for generation.

RQ2 counterpart of src/rq3_manual/references.py. The pinned GitHub evaluator
supplies the rubrics and the official category families. The judged text and
images are the Hugging Face records the generator received; their GitHub copies
are compared and recorded, never substituted, and differences never receive a
guessed cause or a tolerance. In K-dev, chemistry__4 and
multi-instruction_execution__53 have different GitHub instruction text.
"""

from __future__ import annotations

import ast
import base64
from copy import deepcopy
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time

from PIL import Image, ImageChops, ImageOps

from src.taxonomy import get_knowledge_type
from .common import OFFICIAL_REVISION, PROMPT_FILE, REPO_ROOT, file_sha256


DATA_IMPLEMENTATIONS = ("src/dataset.py", "src/taxonomy.py", "src/rq2/common.py")


def _official_namespace(path, base=None):
    """Compile original functions and literal assignments without CLI side effects.

    Copied from src/rq3/evaluate.py (origin/rq3_implementation@f232e68).
    """
    namespace = dict(base or {})
    namespace.update(os=os, json=json, base64=base64, time=time, re=re, logging=logging)
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = []
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            nodes.append(node)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def verify_official_checkout(root):
    """The checkout must be clean and at the pinned revision (as in RQ3)."""
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    text=True).strip()
    if revision != OFFICIAL_REVISION or dirty:
        raise ValueError("Official evaluator checkout must be clean and match the pinned revision")


def _rgb(source):
    with Image.open(source) as handle:
        return ImageOps.exif_transpose(handle).convert("RGB")


def _compare(prepared_png, official_path, prepared_hash, official_hash):
    prepared, official = _rgb(io.BytesIO(prepared_png)), _rgb(official_path)
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


def _png(image_data):
    """The exact RGB image the generator received (src.dataset.decode_image), as PNG."""
    from src.dataset import decode_image

    buffer = io.BytesIO()
    decode_image(image_data).save(buffer, format="PNG")
    data = buffer.getvalue()
    return data, hashlib.sha256(data).hexdigest()


class HFReferenceEvaluator:
    """Official rubrics and families, with the generation's own HF inputs."""

    def __init__(self, official_checkout, splits_path, split, uids):
        self.root = Path(official_checkout).resolve()
        verify_official_checkout(self.root)
        prompts = _official_namespace(self.root / PROMPT_FILE)
        common = _official_namespace(self.root / "metrics_common.py", prompts)
        self.modules = {"common": common}
        for family, filename in (("knowledge", "metrics_knowledge.py"), ("view", "metrics_view_change.py"),
                                 ("temporal", "metrics_temporal_prediction.py")):
            self.modules[family] = _official_namespace(self.root / filename, common)
        self.prompt_sha256 = file_sha256(self.root / PROMPT_FILE)

        requested = set(uids)
        if not requested or any(not isinstance(uid, str) or not uid for uid in requested):
            raise ValueError("Select valid frozen example IDs before validating references")

        from src.dataset import KRIS_DATASET_ID, prepare_example
        from src.rq2.common import load_rq2_examples

        dataset = load_rq2_examples(splits_path, split)
        self._examples = {}
        for index in range(len(dataset)):
            example = prepare_example(dataset[index])
            if example["example_uid"] in requested:
                self._examples[example["example_uid"]] = example
        missing = sorted(requested - set(self._examples))
        if missing:
            raise ValueError(f"Not found in the {split} split: {missing[:5]}")

        self.data_provenance = {
            "dataset": KRIS_DATASET_ID, "split": split,
            "splits_sha256": file_sha256(splits_path),
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "implementation_sha256": {name: file_sha256(REPO_ROOT / name) for name in DATA_IMPLEMENTATIONS},
        }
        self._cache = {}

    def family(self, category):
        for family, module in self.modules.items():
            if category in module["CATEGORIES"]:
                return family
        raise ValueError(f"No official evaluator for eligible category {category}")

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
            raise ValueError("Generation example is not among the loaded HF records")
        if uid in self._cache:
            return self._cache[uid]
        category, image_id = uid.rsplit("__", 1)
        example = self._examples[uid]
        if (example.get("id") != image_id or example.get("category") != category
                or example.get("num_source_images") != 1
                or example.get("knowledge_type") != get_knowledge_type(category)
                or row.get("knowledge_type") != example["knowledge_type"]):
            raise ValueError("Generation/HF ID, category or taxonomy differs")
        if row.get("instruction") != example.get("instruction"):
            raise ValueError(f"Generation instruction differs from the HF record for {uid}")
        family = self.family(category)
        if family not in ("common", "knowledge", "view"):
            raise ValueError("Manual single-reference judging does not support this official family")

        explanation = example.get("explanation")
        explanation = "" if explanation is None else explanation
        annotation = {"ins_en": example["instruction"], "explain_en": explanation}
        source_png, source_sha256 = _png(example["image"])
        ground_truth = None
        if family == "view":
            if example.get("gt_image") is None:
                raise ValueError("Viewpoint judging requires the HF record's ground-truth image")
            ground_truth = _png(example["gt_image"])
            # Presence flag read by the prompt renderer; the image is the HF one.
            annotation["gt_img"] = "hf_gt_image"

        annotation_path = self._official_path(category, "annotation.json", "annotation")
        official = json.loads(annotation_path.read_text(encoding="utf-8")).get(image_id)
        if not isinstance(official, dict):
            raise ValueError(f"Official annotation is missing for {uid}")
        official_explanation = "" if official.get("explain_en") is None else official["explain_en"]
        text = {
            "official_instruction_equal": official.get("ins_en") == annotation["ins_en"],
            "official_explanation_equal": official_explanation == explanation,
        }
        if not text["official_instruction_equal"]:
            text["official_instruction"] = official.get("ins_en")
        if not text["official_explanation_equal"]:
            text["official_explanation"] = official_explanation

        official_source = self._official_path(category, official.get("ori_img"), "single-reference original")
        provenance = {"source": _compare(source_png, official_source, source_sha256, file_sha256(official_source)),
                      "ground_truth": None}
        if ground_truth is not None:
            official_gt = self._official_path(category, official.get("gt_img"), "ground-truth")
            provenance["ground_truth"] = _compare(ground_truth[0], official_gt, ground_truth[1],
                                                  file_sha256(official_gt))

        self._cache[uid] = {
            "category": category, "image_id": image_id, "annotation": annotation,
            "source": (source_png, source_sha256), "ground_truth": ground_truth,
            "reference_provenance": provenance, "text_provenance": text,
        }
        return self._cache[uid]

    def annotation(self, row):
        record = self._validate(row)
        return record["category"], record["image_id"], deepcopy(record["annotation"])

    def source_image(self, row):
        """(PNG bytes, sha256) of the exact source image the generator received."""
        return self._validate(row)["source"]

    def ground_truth(self, row):
        record = self._validate(row)
        if record["ground_truth"] is None:
            raise ValueError("Only viewpoint items have an attached ground-truth reference")
        return record["ground_truth"]

    def reference_provenance(self, row):
        return deepcopy(self._validate(row)["reference_provenance"])

    def text_provenance(self, row):
        return deepcopy(self._validate(row)["text_provenance"])

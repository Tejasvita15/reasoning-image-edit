"""Strict HF reference provenance with local images; no downloads or API calls."""

from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from src.rq3.prepare import data_identity, materialize_dataset
from src.rq3.protocol import canonical_hash, load_protocol, protocol_manifest
from src.rq3_manual import references


def png_bytes(color=(50, 70, 90), size=(9, 7)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def write_json(path, value):
    Path(path).write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def reference_factory(tmp_path, monkeypatch):
    count = [0]

    def make(*, category="anomaly_correction", source=None, official=None, explanation=None,
             official_explanation=None, with_gt=True):
        count[0] += 1
        root = tmp_path / f"fixture_{count[0]}"
        root.mkdir()
        uid = category + "__60"
        config = load_protocol()
        config["_splits"] = {"k_dev": [uid], "k_test": [], "k_small": []}
        config["_subset"] = {"example_uids": []}
        snapshot = protocol_manifest(config)
        snapshot["protocol_hash"] = canonical_hash({name: snapshot[name] for name in ("config", "splits", "subset", "judge")})
        config["protocol_hash"] = snapshot["protocol_hash"]
        public = {**snapshot["config"], "protocol_hash": snapshot["protocol_hash"]}
        source = png_bytes() if source is None else source
        official = source if official is None else official
        item = {"id": "60", "category": category, "instruction": "Apply the requested edit.",
                "image": {"bytes": source, "path": None}, "explanation": explanation}
        if category == "viewpoint_change" and with_gt:
            item["gt_image"] = {"bytes": png_bytes((100, 110, 120)), "path": None}
        prepared = root / "prepared"
        records = materialize_dataset(config, prepared, [item])
        identity = data_identity(config)
        manifest = {"identity": identity, "data_identity": canonical_hash(identity),
                    "examples_sha256": canonical_hash(records["examples"]), **records}
        write_json(prepared / "manifest.json", manifest)
        source_path = root / "generation_source.png"
        source_path.write_bytes((prepared / manifest["examples"][uid]["source_path"]).read_bytes())
        output_path = root / "generation_output.png"
        output_path.write_bytes(png_bytes((130, 140, 150)))
        row = {"example_uid": uid, "category": category, "knowledge_type": records["examples"][uid]["knowledge_type"],
               "instruction": item["instruction"], "status": "ok", "config": public,
               "protocol_hash": snapshot["protocol_hash"], "source_path": str(source_path), "output_path": str(output_path),
               "source_sha256": references.file_sha256(source_path), "output_sha256": references.file_sha256(output_path)}
        official_root = root / "official"
        category_root = official_root / "KRIS_Bench" / category
        category_root.mkdir(parents=True)
        original = category_root / "original.png"
        original.write_bytes(official)
        gt = category_root / "target.png"
        gt.write_bytes(png_bytes((101, 112, 123)))
        annotation_path = category_root / "annotation.json"
        annotation = {"ins_en": item["instruction"], "ori_img": original.name,
                      "gt_img": gt.name, "explain_en": official_explanation}
        write_json(annotation_path, {"60": annotation})
        initialized = []

        def official_init(self, judge):
            initialized.append(judge)
            self.root = official_root
            self.prompt_sha256 = "a" * 64
            self.modules = {"common": {"CATEGORIES": ["anomaly_correction"]},
                            "knowledge": {"CATEGORIES": ["biology"]},
                            "view": {"CATEGORIES": ["viewpoint_change"]}}

        monkeypatch.setattr(references.OfficialEvaluator, "__init__", official_init)
        return SimpleNamespace(root=root, config=config, snapshot=snapshot, manifest=manifest, prepared=prepared,
                               uid=uid, row=row, source_path=source_path, output_path=output_path, original=original,
                               annotation_path=annotation_path, annotation=annotation, initialized=initialized,
                               create=lambda: references.PreparedReferenceEvaluator(snapshot["judge"], snapshot, prepared, [uid]))

    return make


def test_exact_pinned_hf_source_keeps_official_rubric_checks(reference_factory):
    fixture = reference_factory()
    evaluator = fixture.create()
    assert len(fixture.initialized) == 1
    category, image_id, annotation = evaluator.annotation(fixture.row)
    assert (category, image_id) == ("anomaly_correction", "60")
    assert annotation == fixture.annotation
    data = evaluator.data_provenance
    assert data["data_identity"] == fixture.manifest["data_identity"]
    assert data["manifest_sha256"] == references.file_sha256(fixture.prepared / "manifest.json")
    provenance = evaluator.reference_provenance(fixture.row)
    assert provenance["ground_truth"] is None
    source = provenance["source"]
    assert source["official_pixels_equal"] is True
    assert source["official_mae"] == source["official_max_abs_difference"] == 0
    assert source["generation_file_sha256"] == fixture.row["source_sha256"]


def test_official_source_difference_is_diagnostic_not_a_wrong_source_substitution(reference_factory):
    fixture = reference_factory(official=png_bytes((55, 77, 99)))
    before = fixture.source_path.read_bytes()
    evaluator = fixture.create()
    evaluator.annotation(fixture.row)
    source = evaluator.reference_provenance(fixture.row)["source"]
    assert source["official_pixels_equal"] is False
    assert source["official_mae"] == 7.0 and source["official_max_abs_difference"] == 9
    assert source["rgb_sha256"] != source["official_rgb_sha256"]
    assert fixture.source_path.read_bytes() == before


def test_different_official_dimensions_are_not_resized(reference_factory):
    fixture = reference_factory(official=png_bytes(size=(10, 8)))
    source = fixture.create().reference_provenance(fixture.row)["source"]
    assert source["size"] == [9, 7] and source["official_size"] == [10, 8]
    assert source["official_pixels_equal"] is False
    assert source["official_mae"] is None and source["official_max_abs_difference"] is None


def test_generation_source_must_match_prepared_even_if_new_logged_hash_is_valid(reference_factory):
    fixture = reference_factory()
    fixture.source_path.write_bytes(png_bytes((1, 2, 3)))
    fixture.row["source_sha256"] = references.file_sha256(fixture.source_path)
    with pytest.raises(ValueError, match="exact prepared HF image"):
        fixture.create().annotation(fixture.row)


@pytest.mark.parametrize("target", ["source", "output", "prepared", "manifest"])
def test_cached_comparisons_still_recheck_artifact_hashes(reference_factory, target):
    fixture = reference_factory()
    evaluator = fixture.create()
    evaluator.annotation(fixture.row)
    path = {"source": fixture.source_path, "output": fixture.output_path,
            "prepared": fixture.prepared / fixture.manifest["examples"][fixture.uid]["source_path"],
            "manifest": fixture.prepared / "manifest.json"}[target]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        evaluator.annotation(fixture.row)


def test_pixel_comparisons_are_cached_but_official_changes_are_recomputed(reference_factory, monkeypatch):
    fixture = reference_factory()
    evaluator = fixture.create()
    original_compare = references._compare
    calls = []

    def counted(*args):
        calls.append(args)
        return original_compare(*args)

    monkeypatch.setattr(references, "_compare", counted)
    evaluator.annotation(fixture.row)
    evaluator.reference_provenance(fixture.row)
    assert len(calls) == 1
    fixture.original.write_bytes(png_bytes((51, 72, 93)))
    assert evaluator.reference_provenance(fixture.row)["source"]["official_pixels_equal"] is False
    assert len(calls) == 2


@pytest.mark.parametrize("field", ["instruction", "category", "knowledge_type", "example_uid"])
def test_generation_metadata_must_match_exact_prepared_record(reference_factory, field):
    fixture = reference_factory()
    fixture.row[field] = "changed"
    with pytest.raises(ValueError):
        fixture.create().annotation(fixture.row)


def test_official_instruction_and_single_reference_checks_remain_strict(reference_factory):
    fixture = reference_factory()
    fixture.annotation["ins_en"] = "Different edit"
    write_json(fixture.annotation_path, {"60": fixture.annotation})
    with pytest.raises(ValueError, match="instruction mismatch"):
        fixture.create().annotation(fixture.row)
    fixture.annotation["ins_en"] = fixture.row["instruction"]
    fixture.annotation["ori_img"] = ["original.png"]
    write_json(fixture.annotation_path, {"60": fixture.annotation})
    with pytest.raises(ValueError, match="one relative image path"):
        fixture.create().annotation(fixture.row)


@pytest.mark.parametrize("bad_path", ["../../../../outside.png", "/tmp/absolute.png"])
def test_official_reference_paths_cannot_escape_category(reference_factory, bad_path):
    fixture = reference_factory()
    fixture.annotation["ori_img"] = bad_path
    write_json(fixture.annotation_path, {"60": fixture.annotation})
    with pytest.raises(ValueError, match="relative image path|outside its category"):
        fixture.create().annotation(fixture.row)


@pytest.mark.parametrize("prepared,official", [(None, "expected"), ("expected", None), ("first", "second")])
def test_knowledge_explanation_mismatch_rejected(reference_factory, prepared, official):
    fixture = reference_factory(category="biology", explanation=prepared, official_explanation=official)
    with pytest.raises(ValueError, match="knowledge explanation mismatch"):
        fixture.create().annotation(fixture.row)


@pytest.mark.parametrize("prepared,official", [(None, ""), ("", None), ("same explanation", "same explanation")])
def test_knowledge_explanation_exact_match_or_both_empty(reference_factory, prepared, official):
    fixture = reference_factory(category="biology", explanation=prepared, official_explanation=official)
    assert fixture.create().annotation(fixture.row)[2]["explain_en"] == (official or "")


def test_viewpoint_uses_same_record_hf_ground_truth_and_records_official_difference(reference_factory):
    fixture = reference_factory(category="viewpoint_change")
    evaluator = fixture.create()
    path, digest = evaluator.ground_truth(fixture.row)
    assert path == fixture.prepared / fixture.manifest["examples"][fixture.uid]["gt_path"]
    assert digest == references.file_sha256(path)
    gt = evaluator.reference_provenance(fixture.row)["ground_truth"]
    assert gt["prepared_file_sha256"] == digest and "generation_file_sha256" not in gt
    assert gt["official_pixels_equal"] is False and gt["official_mae"] == 2


def test_viewpoint_has_no_unverified_official_ground_truth_fallback(reference_factory):
    fixture = reference_factory(category="viewpoint_change", with_gt=False)
    with pytest.raises(ValueError, match="same frozen HF record"):
        fixture.create().annotation(fixture.row)


def test_archived_data_code_drift_and_prepared_identity_drift_rejected(reference_factory):
    fixture = reference_factory()
    changed = deepcopy(fixture.snapshot)
    changed["config"]["implementation_files_sha256"]["src/rq3/prepare.py"] = "0" * 64
    changed["protocol_hash"] = canonical_hash({name: changed[name] for name in ("config", "splits", "subset", "judge")})
    with pytest.raises(ValueError, match="data implementation differs"):
        references.PreparedReferenceEvaluator(changed["judge"], changed, fixture.prepared, [fixture.uid])
    fixture.manifest["identity"]["dataset"]["revision"] = "0" * 40
    write_json(fixture.prepared / "manifest.json", fixture.manifest)
    with pytest.raises(ValueError, match="Prepared dataset identity differs"):
        fixture.create()


def test_missing_prepared_data_has_no_unchecked_fallback(reference_factory):
    fixture = reference_factory()
    with pytest.raises(ValueError, match="requires --prepared-data"):
        references.PreparedReferenceEvaluator(fixture.snapshot["judge"], fixture.snapshot, None, [fixture.uid])
    (fixture.prepared / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest is missing"):
        fixture.create()


@pytest.mark.skipif(not (os.environ.get("RQ3_TEST_HF_SOURCE") and os.environ.get("RQ3_TEST_OFFICIAL_SOURCE")),
                    reason="Opt-in regression needs the two locally downloaded pinned anomaly_correction__60 JPEGs")
def test_real_anomaly60_hf_and_github_copies_remain_distinct_but_valid(reference_factory):
    import numpy as np

    hf = Path(os.environ["RQ3_TEST_HF_SOURCE"]).read_bytes()
    official = Path(os.environ["RQ3_TEST_OFFICIAL_SOURCE"]).read_bytes()
    assert references.hashlib.sha256(hf).hexdigest() == "53ab7e9807c502756d9cd5497763f04d8bfc2ebd61a698ad36821f00a444b9d3"
    assert references.hashlib.sha256(official).hexdigest() == "370de38cfda7990dcd016d2e9de50cbf808daa5c31a47ba1c17ca00133060682"
    fixture = reference_factory(source=hf, official=official)
    evaluator = fixture.create()
    evaluator.annotation(fixture.row)
    source = evaluator.reference_provenance(fixture.row)["source"]
    assert source["size"] == source["official_size"] == [1260, 1771]
    assert source["official_pixels_equal"] is False
    # JPEG decoding can vary slightly with the installed decoder. Compare to
    # independent int16 arithmetic under this same runtime, not another host's
    # exact floating-point diagnostic value.
    with Image.open(BytesIO(hf)) as image:
        first = np.asarray(image.convert("RGB"), dtype=np.int16)
    with Image.open(BytesIO(official)) as image:
        second = np.asarray(image.convert("RGB"), dtype=np.int16)
    differences = np.abs(first - second)
    assert source["official_mae"] == pytest.approx(float(differences.mean()), abs=1e-12)
    assert source["official_max_abs_difference"] == int(differences.max())
    assert 0 < source["official_mae"] < 1 and 0 < source["official_max_abs_difference"] < 20

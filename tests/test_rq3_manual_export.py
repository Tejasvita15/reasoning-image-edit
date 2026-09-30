"""Blinded export checks with local PNGs and mocked official annotations."""

import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest

from src.rq3.protocol import build_jobs, canonical_hash, load_protocol, protocol_manifest
from src.rq3_manual import common
from src.rq3_manual import export


def write_json(path, value):
    Path(path).write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def archive_factory(tmp_path, monkeypatch):
    counter = [0]

    def make(*, kind="calibration", viewpoint=False, frozen=True):
        counter[0] += 1
        root = tmp_path / f"fixture_{counter[0]}"
        root.mkdir()
        config = load_protocol()
        if kind == "main" and frozen:
            config["statistics"].update(minimum_meaningful_difference=2.0, equivalence_margin=5.0)
        snapshot = protocol_manifest(config)
        snapshot["protocol_hash"] = canonical_hash({key: snapshot[key] for key in ("config", "splits", "subset", "judge")})
        public = {**snapshot["config"], "protocol_hash": snapshot["protocol_hash"]}
        write_json(root / "protocol_snapshot.json", snapshot)
        write_json(root / "protocol.json", public)
        jobs = build_jobs(config, phase="pilot" if kind == "calibration" else "all", seeds=[42])
        if viewpoint:
            uid = next(uid for uid in config["_splits"]["k_dev"] if uid.startswith("viewpoint_change__"))
            first = jobs[0]["example_uid"]
            jobs = [{**job, "example_uid": uid} if job["example_uid"] == first else job for job in jobs]
        images = root / "images"
        images.mkdir()
        rows = []
        for index, job in enumerate(jobs):
            source, output = images / f"source_{index}.png", images / f"edited_{index}.png"
            Image.new("RGB", (8, 6), (index % 256, 20, 30)).save(source)
            Image.new("RGB", (8, 6), (index % 256, 40, 50)).save(output)
            rows.append({"example_uid": job["example_uid"], "condition": job["condition"], "base_seed": 42,
                         "phase": job["phase"], "repeat": 0, "status": "ok", "config": public,
                         "protocol_hash": snapshot["protocol_hash"], "source_path": str(source), "output_path": str(output),
                         "source_sha256": common.file_sha256(source), "output_sha256": common.file_sha256(output),
                         "instruction": f"Make the object brighter, example {index}.",
                         "prompt": "PRIVATE_REWRITE_MUST_NOT_BE_UPLOADED", "thought": {"text": "PRIVATE_REASONING"}})
        generations = root / "generations.jsonl"
        write_rows(generations, rows)
        official_root = root / "official"
        gt = official_root / "KRIS_Bench/viewpoint_change/target.jpeg"
        gt.parent.mkdir(parents=True)
        Image.new("RGB", (8, 6), (1, 2, 3)).save(gt)
        checked = []

        class FakeEvaluator:
            prompt_sha256 = "a" * 64

            def __init__(self, judge):
                self.root = official_root

            def annotation(self, row):
                checked.append(row["example_uid"])
                category, image_id = row["example_uid"].rsplit("__", 1)
                return category, image_id, {"ins_en": row["instruction"], "gt_img": "target.jpeg"}

            def family(self, category):
                return "view" if category == "viewpoint_change" else "common"

        monkeypatch.setattr(export, "OfficialEvaluator", FakeEvaluator)
        policy = {"interface": "chatgpt_manual", "aggregate": "mean_applicable_normalized",
                  "judge_label": "UI test model", "ui_settings": "fresh chats",
                  "adapter_files_sha256": common.adapter_hashes()}
        monkeypatch.setattr(common, "make_policy", lambda *args: copy.deepcopy(policy))

        def prompt(batch, items, evaluator):
            return "\n".join(f"{entry['blind_id']} {Path(entry['source_file']).name} "
                             f"{Path(entry['edited_file']).name}: {items[entry['item_key']]['row']['instruction']}"
                             for entry in batch["items"])

        monkeypatch.setattr(common, "build_prompt", prompt)
        report = root / "repeatability.json"
        write_json(report, {"evaluation_protocol_hash": canonical_hash({**policy, "batch_size": 2}),
                            "judge_interface": "chatgpt_manual", "outputs": 20,
                            "method": "percentile95_absolute_rescore_Q_difference",
                            "candidate_minimum_meaningful_difference": 1.0})
        args = SimpleNamespace(generations=str(generations), output_dir=str(root / "bundle"), kind=kind,
                               limit=20, seeds=[42], batch_size=2, judge_label="UI test model", ui_settings="fresh chats",
                               calibration_report=str(report) if kind == "main" else None)
        return SimpleNamespace(root=root, rows=rows, snapshot=snapshot, args=args, generations=generations,
                               policy=policy, report=report, checked=checked, gt=gt)

    return make


def test_calibration_exports_blinded_two_passes_without_changing_generation_identity(archive_factory):
    fixture = archive_factory()
    before = common.file_sha256(fixture.generations)
    result = export.export_bundle(fixture.args)
    directory = Path(result["output_dir"])
    manifest = common.load_bundle(directory)
    assert manifest["generation_protocol_hash"] == fixture.snapshot["protocol_hash"]
    assert manifest["generation_snapshot"]["judge"]["aggregate_confirmed"] is False
    assert len(manifest["items"]) == 20 and len(manifest["batches"]) == 20
    assert len(fixture.checked) == 20
    assert common.file_sha256(fixture.generations) == before
    ordered, aliases = {}, {}
    for batch in manifest["batches"]:
        ordered.setdefault(batch["pass"], []).extend(entry["item_key"] for entry in batch["items"])
        aliases.setdefault(batch["pass"], set()).update(entry["blind_id"] for entry in batch["items"])
        text = (directory / batch["prompt_path"]).read_text()
        assert "PRIVATE_" not in text and "__" not in text
        for entry in batch["items"]:
            assert "item_" not in Path(entry["source_file"]).name
            item = next(item for item in manifest["items"] if item["item_key"] == entry["item_key"])
            assert (directory / entry["source_file"]).read_bytes() == Path(item["row"]["source_path"]).read_bytes()
            assert (directory / entry["edited_file"]).read_bytes() == Path(item["row"]["output_path"]).read_bytes()
    assert ordered[1] != ordered[2] and set(ordered[1]) == set(ordered[2])
    assert aliases[1].isdisjoint(aliases[2])
    with zipfile.ZipFile(result["zip_path"]) as archive:
        assert "private/manifest.json" in archive.namelist() and "README.md" in archive.namelist()
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in archive.namelist())
    readme = (directory / "README.md").read_text()
    assert "Do not upload the ZIP" in readme and "new chats for pass 2" in readme
    sessions = json.loads((directory / "session_log_template.json").read_text())
    assert set(sessions) == {batch["batch_id"] for batch in manifest["batches"]}
    assert all(session["judged_at"] == "" for session in sessions.values())


def test_shuffling_and_aliases_are_reproducible(archive_factory):
    fixture = archive_factory()
    first = export.export_bundle(fixture.args)
    first_manifest = common.load_bundle(first["output_dir"])
    fixture.args.output_dir = str(fixture.root / "second_bundle")
    second = export.export_bundle(fixture.args)
    second_manifest = common.load_bundle(second["output_dir"])
    assert first_manifest == second_manifest


def test_bundle_rejects_importer_adapter_code_drift(archive_factory, monkeypatch):
    fixture = archive_factory()
    result = export.export_bundle(fixture.args)
    changed = dict(common.adapter_hashes())
    changed["src/rq3_manual/results.py"] = "0" * 64
    monkeypatch.setattr(common, "adapter_hashes", lambda: changed)
    with pytest.raises(ValueError, match="code|adapter|implementation"):
        common.load_bundle(result["output_dir"])


def test_failed_first_selected_key_is_not_replaced_by_later_output(archive_factory):
    fixture = archive_factory()
    fixture.rows[0]["status"] = "error"
    write_rows(fixture.generations, fixture.rows)
    with pytest.raises(ValueError, match="unresolved failure"):
        export.export_bundle(fixture.args)
    assert not Path(fixture.args.output_dir).exists()


def test_recovered_error_preserves_original_selection_order(archive_factory):
    fixture = archive_factory()
    recovered = copy.deepcopy(fixture.rows[0])
    fixture.rows[0]["status"] = "error"
    write_rows(fixture.generations, fixture.rows + [recovered])
    result = export.export_bundle(fixture.args)
    items = common.load_bundle(result["output_dir"])["items"]
    assert export._row_key(items[0]["row"]) == export._row_key(recovered)
    assert export._row_key(items[-1]["row"]) == export._row_key(fixture.rows[19])


@pytest.mark.parametrize("part", ["snapshot", "public_manifest", "row_hash", "row_config"])
def test_archived_provenance_mismatch_rejected(archive_factory, part):
    fixture = archive_factory()
    if part == "snapshot":
        fixture.snapshot["config"]["generation"]["num_inference_steps"] = 50
        write_json(fixture.root / "protocol_snapshot.json", fixture.snapshot)
    elif part == "public_manifest":
        write_json(fixture.root / "protocol.json", {})
    else:
        fixture.rows[0] = copy.deepcopy(fixture.rows[0])
        if part == "row_hash":
            fixture.rows[0]["protocol_hash"] = "0" * 64
        else:
            fixture.rows[0]["config"]["generation"]["num_inference_steps"] = 50
        write_rows(fixture.generations, fixture.rows)
    with pytest.raises(ValueError, match="snapshot|protocol|configuration"):
        export.export_bundle(fixture.args)


@pytest.mark.parametrize("image", ["source", "output"])
def test_changed_image_bytes_rejected(archive_factory, image):
    fixture = archive_factory()
    Path(fixture.rows[0][f"{image}_path"]).write_bytes(b"changed image")
    with pytest.raises(ValueError, match="image is missing or changed"):
        export.export_bundle(fixture.args)


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_phase", "outside_split", "bad_repeat"])
def test_invalid_rows_are_rejected(archive_factory, mutation):
    fixture = archive_factory()
    if mutation == "duplicate":
        fixture.rows.append(copy.deepcopy(fixture.rows[0]))
    elif mutation == "wrong_phase":
        fixture.rows[0]["phase"] = "rq3_1"
    elif mutation == "outside_split":
        fixture.rows[0]["example_uid"] = fixture.snapshot["splits"]["k_small"][0]
    else:
        fixture.rows[0]["repeat"] = True
    write_rows(fixture.generations, fixture.rows)
    with pytest.raises(ValueError):
        export.export_bundle(fixture.args)


def test_main_exports_all_200_once_with_matching_manual_calibration(archive_factory):
    fixture = archive_factory(kind="main")
    result = export.export_bundle(fixture.args)
    manifest = common.load_bundle(result["output_dir"])
    assert len(manifest["items"]) == 200 and len(manifest["batches"]) == 100
    assert {batch["pass"] for batch in manifest["batches"]} == {1}
    assert manifest["calibration_report"]["candidate_minimum_meaningful_difference"] == 1


def test_main_requires_complete_200_cells(archive_factory):
    fixture = archive_factory(kind="main")
    write_rows(fixture.generations, fixture.rows[:-1])
    with pytest.raises(ValueError, match="all 200"):
        export.export_bundle(fixture.args)


@pytest.mark.parametrize("mismatch", ["missing", "policy", "margin", "unfrozen", "incomplete", "api"])
def test_main_rejects_missing_or_incompatible_manual_calibration(archive_factory, mismatch):
    fixture = archive_factory(kind="main", frozen=mismatch != "unfrozen")
    if mismatch == "missing":
        fixture.args.calibration_report = None
    elif mismatch == "policy":
        fixture.args.batch_size = 1
    elif mismatch in {"margin", "incomplete", "api"}:
        report = json.loads(fixture.report.read_text())
        if mismatch == "margin":
            report["candidate_minimum_meaningful_difference"] = 3
        elif mismatch == "incomplete":
            report["outputs"] = 19
        else:
            report["judge_interface"] = "api"
        write_json(fixture.report, report)
    with pytest.raises(ValueError, match="calibration|evaluation policy|Freeze finite"):
        export.export_bundle(fixture.args)


def test_viewpoint_ground_truth_uses_unchanged_bytes_and_blind_reference_filename(archive_factory):
    fixture = archive_factory(viewpoint=True)
    result = export.export_bundle(fixture.args)
    directory = Path(result["output_dir"])
    manifest = common.load_bundle(directory)
    references = [entry for batch in manifest["batches"] for entry in batch["items"] if "ground_truth_file" in entry]
    assert len(references) == 12
    for entry in references:
        image = directory / entry["ground_truth_file"]
        assert image.name == entry["blind_id"] + "_reference.jpeg"
        assert image.read_bytes() == fixture.gt.read_bytes()


def test_partial_export_is_not_published_and_existing_bundle_is_not_overwritten(archive_factory, monkeypatch):
    fixture = archive_factory()
    with monkeypatch.context() as patch:
        patch.setattr(common, "build_prompt", lambda *args: (_ for _ in ()).throw(ValueError("mock prompt failure")))
        with pytest.raises(ValueError, match="mock prompt failure"):
            export.export_bundle(fixture.args)
    output = Path(fixture.args.output_dir)
    assert not output.exists() and not output.with_name(output.name + ".zip").exists()
    assert not list(output.parent.glob(".bundle.*"))
    export.export_bundle(fixture.args)
    before = common.file_sha256(output / "private/manifest.json")
    with pytest.raises(ValueError, match="already exists"):
        export.export_bundle(fixture.args)
    assert common.file_sha256(output / "private/manifest.json") == before


def test_running_generation_cannot_be_exported(archive_factory):
    fixture = archive_factory()
    with (fixture.root / ".run.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="still active"):
            export.export_bundle(fixture.args)

"""Complete manual-analysis fixtures with real imports and no judge calls."""
import copy
import csv
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from src.rq3.analyze import TIMING_FIELDS
from src.rq3.evaluate import file_sha256, read_jsonl
from src.rq3.protocol import build_jobs, canonical_hash, load_protocol, protocol_manifest
from src.rq3_manual import analysis, common, results
from src.rq3_manual.prompts import RUBRIC_NAMES
from src.taxonomy import get_reasoning_dimension


def write_json(path, value):
    Path(path).write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def main_fixture(tmp_path, monkeypatch):
    """Build 200 real imported manual scores and 600 measured repetitions."""
    count = [0]

    def create(*, thinking_gain=True, delta=5, mmd=2, candidate=1):
        count[0] += 1
        root = tmp_path / f"main_{count[0]}"
        bundle = root / "bundle"
        (bundle / "private").mkdir(parents=True)
        (bundle / "images").mkdir()
        responses = root / "responses"
        responses.mkdir()
        source, edited = bundle / "images/source.png", bundle / "images/edited.png"
        Image.new("RGB", (8, 6), "blue").save(source)
        Image.new("RGB", (8, 6), "green").save(edited)
        config = load_protocol()
        config["statistics"].update(equivalence_margin=delta, minimum_meaningful_difference=mmd)
        snapshot = protocol_manifest(config)
        snapshot["protocol_hash"] = canonical_hash({name: snapshot[name] for name in ("config", "splits", "subset", "judge")})
        generation_hash = snapshot["protocol_hash"]
        public = {**snapshot["config"], "protocol_hash": generation_hash}
        evaluator = SimpleNamespace(prompt_sha256="synthetic-native-prompt-hash", modules={
            "common": {name: "synthetic native rubric " + name for name in RUBRIC_NAMES}})
        policy = common.make_policy("Synthetic manual judge", "fresh isolated synthetic sessions", evaluator)
        policy["batch_size"] = 2
        policy_hash = canonical_hash(policy)
        items, generations = [], []
        source_hash, edited_hash = file_sha256(source), file_sha256(edited)
        for index, job in enumerate(build_jobs(config, phase="all", seeds=[42])):
            uid, condition = job["example_uid"], job["condition"]
            category = uid.rsplit("__", 1)[0]
            family = ("knowledge" if get_reasoning_dimension(category) in
                      {"natural_science", "social_science", "logical_reasoning"}
                      else "view" if category == "viewpoint_change" else "common")
            row = {**job, "config": public, "protocol_hash": generation_hash,
                   "status": "ok", "source_path": str(source), "source_sha256": source_hash,
                   "output_path": str(edited), "output_sha256": edited_hash,
                   "instruction": "edit " + uid, "prompt": ("edit " if condition in "AC" else "rewrite ") + uid,
                   "initial_noise_sha256": "initial-noise-" + uid, "peak_vram_bytes": 12345}
            for repeat in range(4):
                duration = 4 if condition in "AB" else 2
                duration *= (999, .5, 1, 25)[repeat]
                timing = {name: .1 for name in TIMING_FIELDS}
                timing.update(diffusion_seconds=duration, measured_replay_seconds=duration + .5,
                              charged_total_seconds=duration + 1)
                generation = {**row, "repeat": repeat, "timings": timing}
                generations.append(generation)
                if repeat == 0:
                    item = {"item_key": f"item_{index:04d}", "row": generation,
                            "official_family": family, "knowledge_required": family == "knowledge"}
                    if family == "view":
                        item["ground_truth_sha256"] = source_hash
                    items.append(item)
        manifest = {"schema_version": 1, "kind": "main", "seeds": [42], "generation_protocol_hash": generation_hash,
                    "generation_snapshot": snapshot, "evaluation_protocol": policy,
                    "evaluation_protocol_hash": policy_hash, "selection_hash": canonical_hash(common.selection_records(items)),
                    "items": items, "batches": [], "files": {"images/source.png": source_hash, "images/edited.png": edited_hash},
                    "calibration_report": {"outputs": 20, "method": "percentile95_absolute_rescore_Q_difference",
                                           "judge_interface": "chatgpt_manual", "judge_model": policy["judge_label"],
                                           "generation_protocol_hash": "synthetic-calibration-generation-protocol",
                                           "evaluation_protocol_hash": policy_hash, "selection_hash": "synthetic-calibration-selection",
                                           "candidate_minimum_meaningful_difference": candidate,
                                           "mean_absolute_difference": candidate}}
        sessions = {}
        for start in range(0, len(items), 2):
            batch_id = f"p1_b{start // 2 + 1:03d}"
            prompt_path = batch_id + "_prompt.md"
            (bundle / prompt_path).write_text("Synthetic prompt for " + batch_id, encoding="utf-8")
            manifest["files"][prompt_path] = file_sha256(bundle / prompt_path)
            batch = {"batch_id": batch_id, "pass": 1, "prompt_path": prompt_path, "items": []}
            ratings = []
            for index, item in enumerate(items[start:start+2], start):
                blind_id = f"blind_{index:04d}"
                entry = {"blind_id": blind_id, "item_key": item["item_key"],
                         "source_file": "images/source.png", "edited_file": "images/edited.png"}
                if item["official_family"] == "view":
                    entry["ground_truth_file"] = "images/source.png"
                batch["items"].append(entry)
                component = 4 if thinking_gain and item["row"]["condition"] in "BDEF" else 3
                ratings.append({"blind_id": blind_id, "status": "ok", "consistency_score": component,
                                "instruction_score": component, "quality_score": component,
                                "knowledge_score": component if item["knowledge_required"] else None,
                                "rationale": "Synthetic evidence for each applicable metric."})
            manifest["batches"].append(batch)
            write_json(responses / (batch_id + ".json"), {"batch_id": batch_id, "results": ratings})
            sessions[batch_id] = {"judge_label": policy["judge_label"], "ui_settings": policy["ui_settings"],
                                  "judged_at": "2026-09-30T12:00:00+08:00", "conversation_url_or_id": "synthetic-" + batch_id}
        manifest["manifest_hash"] = canonical_hash(manifest)
        manifest_path = bundle / "private/manifest.json"
        write_json(manifest_path, manifest)
        sessions_path = root / "sessions.json"
        write_json(sessions_path, sessions)
        score_path, generation_path = root / "scores.jsonl", root / "generations.jsonl"
        write_rows(generation_path, generations)
        results.import_results(SimpleNamespace(bundle=str(bundle), responses=str(responses), session_log=str(sessions_path),
                                               pass_number=1, output=str(score_path)))
        args = SimpleNamespace(bundle=str(bundle), scores=str(score_path), generations=str(generation_path),
                               output_dir=str(root / "analysis"), similarity=False)
        def resign_manifest():
            manifest["manifest_hash"] = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
            write_json(manifest_path, manifest)
            scores = read_jsonl(score_path)
            for score in scores:
                score["bundle_manifest_hash"] = manifest["manifest_hash"]
            write_rows(score_path, scores)
        return SimpleNamespace(root=root, bundle=bundle, manifest=manifest, manifest_path=manifest_path,
                               scores=score_path, generations=generation_path, generation_rows=generations,
                               args=args, resign_manifest=resign_manifest)

    for name in ("plot_quality_latency", "region_diagnostics", "similarity_diagnostics"):
        monkeypatch.setattr(analysis, name, lambda *args, **kwargs: None)
    return create


def test_complete_manual_main_analysis_uses_validated_200_scores_and_600_timings(main_fixture, monkeypatch):
    fixture = main_fixture()
    calls = []
    original = results.validate_score_row
    def record_validation(row):
        calls.append((row["example_uid"], row["condition"], row["base_seed"]))
        return original(row)
    monkeypatch.setattr(results, "validate_score_row", record_validation)
    analysis.analyze_manual(fixture.args)
    report = json.loads((Path(fixture.args.output_dir) / "report.json").read_text())
    assert len(calls) == len(set(calls)) == report["expected_quality_outputs"] == 200
    assert len(fixture.generation_rows) == 800
    assert report["judge_interface"] == "chatgpt_manual"
    assert report["evaluation_protocol_hash"] == fixture.manifest["evaluation_protocol_hash"]
    assert report["contrasts"]["G0_B_minus_A"]["mean"] == 25
    assert report["contrasts"]["G1_D_minus_C"]["mean"] == 25
    assert report["contrasts"]["interaction_D_minus_C_minus_B_plus_A"]["mean"] == 0
    assert report["contrasts"]["G0_B_minus_A"]["bootstrap_resamples"] == 10000
    assert report["ablations"]["contrasts"]["full_D_minus_vanilla_B"]["n_examples"] == 20
    assert report["retention"]["quality_retained_one_sided_rule"] is True
    assert report["retention"]["unaccelerated_thinking_benefit_established_above_MMD"] is True
    accelerated = report["efficiency_thinking_on"]["conditions"]["D"]
    assert accelerated["mean_diffusion_seconds"] == 2
    assert accelerated["generator_speedup_mean_paired_ratio"] == 2
    with (Path(fixture.args.output_dir) / "paired_effects.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 40


@pytest.mark.parametrize("mutation", ["api", "foreign_bundle", "wrong_item", "wrong_selection_index", "wrong_source", "altered_Q", "altered_raw"])
def test_manual_analysis_rejects_foreign_or_altered_scores(main_fixture, mutation):
    fixture = main_fixture()
    scores = read_jsonl(fixture.scores)
    row = scores[0]
    if mutation == "api":
        row["judge_interface"] = "openai_api"
    elif mutation == "foreign_bundle":
        row["bundle_manifest_hash"] = "different-bundle"
    elif mutation == "wrong_item":
        row["item_key"] = scores[1]["item_key"]
    elif mutation == "wrong_selection_index":
        row["selection_index"] = scores[1]["selection_index"]
    elif mutation == "wrong_source":
        row["source_sha256"] = "different-source"
    elif mutation == "altered_Q":
        row["Q"] += 1
    else:
        row["raw_response_json"] += " "
    write_rows(fixture.scores, scores)
    with pytest.raises(ValueError):
        analysis.analyze_manual(fixture.args)
    assert not Path(fixture.args.output_dir).exists()


@pytest.mark.parametrize("mutation", ["nan_candidate", "negative_candidate", "candidate_above_mmd", "wrong_method", "api_report", "too_few_outputs"])
def test_manual_analysis_requires_valid_calibration_before_retention_claim(main_fixture, mutation):
    fixture = main_fixture()
    calibration = fixture.manifest["calibration_report"]
    if mutation == "nan_candidate":
        calibration["candidate_minimum_meaningful_difference"] = float("nan")
    elif mutation == "negative_candidate":
        calibration["candidate_minimum_meaningful_difference"] = -1
    elif mutation == "candidate_above_mmd":
        calibration["candidate_minimum_meaningful_difference"] = 3
    elif mutation == "wrong_method":
        calibration["method"] = "unverified_guess"
    elif mutation == "api_report":
        calibration["judge_interface"] = "openai_api"
    else:
        calibration["outputs"] = 19
    fixture.resign_manifest()
    with pytest.raises(ValueError):
        analysis.analyze_manual(fixture.args)
    assert not Path(fixture.args.output_dir).exists()


@pytest.mark.parametrize("delta,mmd", [(None, 2), (0, 0), (1, 2)])
def test_manual_analysis_requires_frozen_valid_equivalence_margin(main_fixture, delta, mmd):
    fixture = main_fixture(delta=delta, mmd=mmd)
    with pytest.raises(ValueError):
        analysis.analyze_manual(fixture.args)
    assert not Path(fixture.args.output_dir).exists()


def test_equal_scores_do_not_establish_a_thinking_benefit(main_fixture):
    fixture = main_fixture(thinking_gain=False)
    analysis.analyze_manual(fixture.args)
    report = json.loads((Path(fixture.args.output_dir) / "report.json").read_text())
    assert report["retention"]["quality_retained_one_sided_rule"] is True
    assert report["retention"]["unaccelerated_thinking_benefit_established_above_MMD"] is False
    assert report["contrasts"]["G0_B_minus_A"]["mean"] == 0


@pytest.mark.parametrize("mutation", ["raw_response", "session", "reused_conversation"])
def test_individually_valid_rows_must_share_batch_evidence_and_use_independent_chats(main_fixture, mutation):
    fixture = main_fixture()
    scores = read_jsonl(fixture.scores)
    if mutation == "raw_response":
        scores[0]["raw_response_json"] += " "
        scores[0]["raw_response_sha256"] = sha256(scores[0]["raw_response_json"].encode()).hexdigest()
    elif mutation == "session":
        scores[0]["session_metadata"]["judged_at"] = "2026-09-30T12:01:00+08:00"
        scores[0]["session_metadata_sha256"] = canonical_hash(scores[0]["session_metadata"])
    else:
        second_batch = scores[2]["batch_id"]
        conversation = scores[0]["session_metadata"]["conversation_url_or_id"]
        for score in scores:
            if score["batch_id"] == second_batch:
                score["session_metadata"]["conversation_url_or_id"] = conversation
                score["session_metadata_sha256"] = canonical_hash(score["session_metadata"])
    for row in scores:
        results.validate_score_row(row)
    write_rows(fixture.scores, scores)
    with pytest.raises(ValueError, match="batch|conversation"):
        analysis.analyze_manual(fixture.args)


def test_analysis_rejects_evaluation_adapter_drift_since_export(main_fixture, monkeypatch):
    fixture = main_fixture()
    changed = dict(common.adapter_hashes())
    changed["src/rq3_manual/analysis.py"] = "changed-implementation"
    monkeypatch.setattr(common, "adapter_hashes", lambda: changed)
    with pytest.raises(ValueError, match="code changed since export"):
        analysis.analyze_manual(fixture.args)
    assert not Path(fixture.args.output_dir).exists()


@pytest.mark.parametrize("mutation", ["incomplete_scores", "missing_timing", "changed_noise", "changed_image", "tampered_bundle"])
def test_manual_analysis_rejects_incomplete_or_changed_experiment(main_fixture, mutation):
    fixture = main_fixture()
    if mutation == "incomplete_scores":
        write_rows(fixture.scores, read_jsonl(fixture.scores)[:-1])
    elif mutation == "missing_timing":
        write_rows(fixture.generations, fixture.generation_rows[:-1])
    elif mutation == "changed_noise":
        rows = copy.deepcopy(fixture.generation_rows)
        rows[1]["initial_noise_sha256"] = "changed-timing-noise"
        write_rows(fixture.generations, rows)
    elif mutation == "changed_image":
        Path(fixture.generation_rows[0]["output_path"]).write_bytes(b"changed")
    else:
        fixture.manifest["evaluation_protocol"]["judge_label"] = "another judge"
        write_json(fixture.manifest_path, fixture.manifest)
    with pytest.raises(ValueError):
        analysis.analyze_manual(fixture.args)
    assert not Path(fixture.args.output_dir).exists()

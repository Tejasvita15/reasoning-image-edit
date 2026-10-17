"""Offline manual-result integrity tests; no model, network, or API client."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rq3_manual.common import adapter_hashes, canonical_hash, selection_records
from src.rq3_manual.results import import_results, repeatability, validate_score_row


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def judging_fixture(tmp_path, count=2):
    bundle, responses = tmp_path / "bundle", tmp_path / "responses"
    responses.mkdir()
    policy = {"judge_label": "Recorded ChatGPT label", "ui_settings": "temporary chat; memory/tools off",
              "interface": "chatgpt_manual", "aggregate": "mean_applicable_normalized", "prompt_version": "fixture",
              "adapter_files_sha256": adapter_hashes()}
    snapshot = {"config": {"fixture": True}, "splits": {}, "subset": {}, "judge": {}}
    generation_hash = canonical_hash(snapshot)
    snapshot["protocol_hash"] = generation_hash
    items = []
    for index in range(count):
        source = f"source {index}".encode()
        output = f"output {index}".encode()
        row = {"example_uid": f"biology__{index}", "condition": "A", "base_seed": 42,
               "protocol_hash": generation_hash, "config": {**snapshot["config"], "protocol_hash": generation_hash},
               "source_sha256": hashlib.sha256(source).hexdigest(), "output_sha256": hashlib.sha256(output).hexdigest()}
        knowledge = index % 2 == 0
        items.append({"item_key": f"item_{index:04d}", "row": row,
                      "official_family": "knowledge" if knowledge else "common", "knowledge_required": knowledge})
    manifest = {"schema_version": 1, "kind": "calibration", "generation_protocol_hash": generation_hash,
                "generation_snapshot": snapshot, "evaluation_protocol": policy,
                "evaluation_protocol_hash": canonical_hash(policy), "selection_hash": canonical_hash(selection_records(items)),
                "seeds": [42], "items": items, "batches": [], "files": {}}
    sessions = {}
    for pass_number in (1, 2):
        for offset in range(0, count, 2):
            batch_id = f"p{pass_number}_b{offset // 2 + 1:03d}"
            batch = {"pass": pass_number, "batch_id": batch_id, "prompt_path": f"upload/{batch_id}/prompt.txt", "items": []}
            prompt_path = bundle / batch["prompt_path"]
            prompt_path.parent.mkdir(parents=True)
            prompt_path.write_text("Frozen prompt", encoding="utf-8")
            manifest["files"][batch["prompt_path"]] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
            results = []
            for index in range(offset, min(offset + 2, count)):
                blind_id = f"{batch_id}_image{index}"
                entry = {"blind_id": blind_id, "item_key": items[index]["item_key"],
                         "source_file": f"upload/{batch_id}/{blind_id}_source.png",
                         "edited_file": f"upload/{batch_id}/{blind_id}_edited.png"}
                for field, content in (("source_file", f"source {index}"), ("edited_file", f"output {index}")):
                    (bundle / entry[field]).write_bytes(content.encode())
                    manifest["files"][entry[field]] = hashlib.sha256(content.encode()).hexdigest()
                batch["items"].append(entry)
                value = 4 if pass_number == 1 else (1 if index == count - 1 else 5)
                results.append({"blind_id": blind_id, "status": "ok", "consistency_score": value,
                                "instruction_score": value, "quality_score": value,
                                "knowledge_score": value if items[index]["knowledge_required"] else None,
                                "rationale": "Visible evidence supports these ratings."})
            manifest["batches"].append(batch)
            write_json(responses / (batch_id + ".json"), {"batch_id": batch_id, "results": results})
            sessions[batch_id] = {"judge_label": policy["judge_label"], "ui_settings": policy["ui_settings"],
                                  "judged_at": f"2026-09-30T0{pass_number}:00:00+08:00",
                                  "conversation_url_or_id": "conversation-" + batch_id}
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(bundle / "private/manifest.json", manifest)
    session_log = tmp_path / "sessions.json"
    write_json(session_log, sessions)
    args = SimpleNamespace(bundle=bundle, responses=responses, session_log=session_log, pass_number=1,
                           output=tmp_path / "scores1.jsonl")
    return SimpleNamespace(args=args, manifest=manifest, sessions=sessions, root=tmp_path)


def edit_response(fixture, mutate, name="p1_b001"):
    path = fixture.args.responses / (name + ".json")
    value = json.loads(path.read_text())
    mutate(value)
    write_json(path, value)


def test_import_preserves_raw_json_hash_session_and_score_semantics(tmp_path):
    fixture = judging_fixture(tmp_path)
    report = import_results(fixture.args)
    assert report["outputs"] == 2
    rows = read_rows(fixture.args.output)
    assert [row["Q"] for row in rows] == [75, 75]
    assert rows[1]["knowledge_score"] is None
    assert rows[0]["knowledge_score"] == 4
    raw = (fixture.args.responses / "p1_b001.json").read_bytes()
    for row in rows:
        assert row["raw_response_json"].encode("utf-8") == raw
        assert row["raw_response_sha256"] == hashlib.sha256(raw).hexdigest()
        assert row["session_metadata"] == fixture.sessions["p1_b001"]
        assert validate_score_row(row) is row
    original = fixture.args.output.read_bytes()
    import_results(fixture.args)
    assert fixture.args.output.read_bytes() == original
    edit_response(fixture, lambda value: value["results"][0].update(rationale="Changed answer"))
    with pytest.raises(ValueError, match="conflicts"):
        import_results(fixture.args)
    assert fixture.args.output.read_bytes() == original


@pytest.mark.parametrize("invalid", [True, False, 0, 6, 3.0, "3", None])
def test_rejects_noninteger_or_out_of_range_scores_before_publication(tmp_path, invalid):
    fixture = judging_fixture(tmp_path)
    edit_response(fixture, lambda value: value["results"][0].update(consistency_score=invalid))
    with pytest.raises(ValueError, match="integer"):
        import_results(fixture.args)
    assert not fixture.args.output.exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value["results"].append(copy.deepcopy(value["results"][0])), "Duplicate blind"),
    (lambda value: value["results"].pop(), "Missing/extra"),
    (lambda value: value["results"][0].update(status="unreadable"), "Unreadable"),
    (lambda value: value["results"][0].update(status="uncertain"), "status"),
    (lambda value: value["results"][0].update(rationale=" "), "rationale"),
    (lambda value: value["results"][0].update(extra=1), "exactly"),
    (lambda value: value.update(batch_id="wrong"), "identify batch"),
    (lambda value: value["results"][0].update(knowledge_score=None), "knowledge_score"),
    (lambda value: value["results"][1].update(knowledge_score=4), "must be null"),
    (lambda value: value["results"][0].update(instruction_score=2, knowledge_score=4), "official rubric"),
])
def test_rejects_incomplete_unreadable_or_inconsistent_responses(tmp_path, mutation, match):
    fixture = judging_fixture(tmp_path)
    edit_response(fixture, mutation)
    with pytest.raises(ValueError, match=match):
        import_results(fixture.args)
    assert not fixture.args.output.exists()


def test_rejects_duplicate_json_keys_missing_batches_and_alternate_answers(tmp_path):
    fixture = judging_fixture(tmp_path)
    path = fixture.args.responses / "p1_b001.json"
    raw = path.read_text()
    path.write_text(raw.replace('"batch_id":', '"batch_id":"duplicate", "batch_id":', 1))
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        import_results(fixture.args)
    path.unlink()
    with pytest.raises(ValueError, match="Missing response"):
        import_results(fixture.args)
    path.write_text(raw)
    (fixture.args.responses / "p1_b001_alternative.json").write_text(raw)
    with pytest.raises(ValueError, match="alternative"):
        import_results(fixture.args)


@pytest.mark.parametrize("field,value,match", [
    ("judge_label", "Different model", "frozen judge"),
    ("ui_settings", "tools on", "frozen judge"),
    ("judged_at", "2026-09-30T00:00:00", "timezone"),
    ("judged_at", "yesterday", "ISO datetime"),
    ("conversation_url_or_id", "", "nonempty"),
    ("conversation_url_or_id", "conversation-p2_b001", "distinct conversation"),
])
def test_session_metadata_is_operator_recorded_and_frozen(tmp_path, field, value, match):
    fixture = judging_fixture(tmp_path)
    fixture.sessions["p1_b001"][field] = value
    write_json(fixture.args.session_log, fixture.sessions)
    with pytest.raises(ValueError, match=match):
        import_results(fixture.args)


def test_first_pass_allows_unstarted_second_pass_template_without_changing_rerun(tmp_path):
    fixture = judging_fixture(tmp_path)
    complete = copy.deepcopy(fixture.sessions)
    fixture.sessions["p2_b001"].update(judged_at="", conversation_url_or_id="")
    write_json(fixture.args.session_log, fixture.sessions)
    import_results(fixture.args)
    original = fixture.args.output.read_bytes()
    write_json(fixture.args.session_log, complete)
    import_results(fixture.args)
    assert fixture.args.output.read_bytes() == original


def test_one_bad_late_batch_never_publishes_partial_scores(tmp_path):
    fixture = judging_fixture(tmp_path, count=4)
    edit_response(fixture, lambda value: value["results"][0].update(status="unreadable"), name="p1_b002")
    with pytest.raises(ValueError, match="Unreadable"):
        import_results(fixture.args)
    assert not fixture.args.output.exists()


def test_atomic_publication_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    fixture = judging_fixture(tmp_path)
    original = Path.replace

    def fail_publication(path, target):
        if Path(target) == fixture.args.output:
            raise OSError("interrupted publication")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication)
    with pytest.raises(OSError, match="interrupted publication"):
        import_results(fixture.args)
    assert not fixture.args.output.exists()
    assert not list(tmp_path.glob(".manual-results-*"))


def two_passes(tmp_path, count=20):
    fixture = judging_fixture(tmp_path, count=count)
    import_results(fixture.args)
    first = fixture.args.output
    fixture.args.pass_number, fixture.args.output = 2, tmp_path / "scores2.jsonl"
    import_results(fixture.args)
    return fixture, SimpleNamespace(first=first, second=fixture.args.output, output=tmp_path / "repeatability.json")


def test_repeatability_matches_independent_passes_and_linear_p95(tmp_path):
    _, args = two_passes(tmp_path)
    report = repeatability(args)
    assert report["outputs"] == 20
    assert report["candidate_minimum_meaningful_difference"] == pytest.approx(27.5)
    assert report["mean_absolute_difference"] == pytest.approx(27.5)
    original = args.output.read_bytes()
    repeatability(args)
    assert args.output.read_bytes() == original


def test_repeatability_rejects_small_or_posthoc_subsets_and_same_pass(tmp_path):
    _, args = two_passes(tmp_path, count=2)
    with pytest.raises(ValueError, match="at least 20"):
        repeatability(args)
    assert not args.output.exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "entire frozen selection"),
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "unique output"),
    (lambda rows: rows[0].update(Q=0), "Q differs"),
    (lambda rows: rows[0].update(raw_response_json="{}"), "Raw response hash"),
    (lambda rows: rows[0].update(output_sha256="different"), "frozen selection"),
    (lambda rows: rows[0].update(selection_hash="different"), "mixes frozen"),
])
def test_repeatability_rejects_tampered_or_incomplete_scores(tmp_path, mutation, match):
    _, args = two_passes(tmp_path)
    rows = read_rows(args.second)
    mutation(rows)
    write_rows(args.second, rows)
    with pytest.raises(ValueError, match=match):
        repeatability(args)
    assert not args.output.exists()


def test_repeatability_rejects_same_pass_and_reused_conversations(tmp_path):
    _, args = two_passes(tmp_path)
    with pytest.raises(ValueError, match="pass 1 against pass 2"):
        repeatability(SimpleNamespace(first=args.first, second=args.first, output=args.output))
    rows = read_rows(args.second)
    for row in rows:
        row["session_metadata"]["conversation_url_or_id"] = row["session_metadata"]["conversation_url_or_id"].replace("p2_", "p1_")
        row["session_metadata_sha256"] = canonical_hash(row["session_metadata"])
    write_rows(args.second, rows)
    with pytest.raises(ValueError, match="independent conversations"):
        repeatability(args)


def test_public_row_validation_rejects_edited_components_even_with_recomputed_q(tmp_path):
    fixture = judging_fixture(tmp_path)
    import_results(fixture.args)
    row = read_rows(fixture.args.output)[0]
    row.update(consistency_score=5, Q=81.25)
    with pytest.raises(ValueError, match="preserved raw response"):
        validate_score_row(row)
    row.update(instruction_score=2, knowledge_score=4)
    with pytest.raises(ValueError, match="official rubric"):
        validate_score_row(row)


def test_public_row_validation_rejects_adapter_drift(tmp_path, monkeypatch):
    fixture = judging_fixture(tmp_path)
    import_results(fixture.args)
    row = read_rows(fixture.args.output)[0]
    monkeypatch.setattr("src.rq3_manual.results.adapter_hashes", lambda: {"changed.py": "different"})
    with pytest.raises(ValueError, match="implementation changed"):
        validate_score_row(row)

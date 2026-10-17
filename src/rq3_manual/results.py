"""Import complete, auditable manual ChatGPT ratings and compare two passes.

This module performs no API calls. Responses and operator-recorded session
metadata must match a frozen bundle; scores are published only as a full set.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from .common import adapter_hashes, canonical_hash, file_sha256, load_bundle, selection_records


SCORE_FIELDS = ("consistency_score", "instruction_score", "quality_score", "knowledge_score")
RESULT_FIELDS = {"blind_id", "status", "rationale", *SCORE_FIELDS}
SESSION_FIELDS = {"judge_label", "ui_settings", "judged_at", "conversation_url_or_id"}
AGGREGATE = "mean_applicable_normalized"
SCORE_ROW_FIELDS = {
    "schema_version", "status", "example_uid", "condition", "base_seed", "output_sha256", "source_sha256",
    "protocol_hash", "generation_protocol_hash", "generation_snapshot", "evaluation_protocol",
    "evaluation_protocol_hash", "selection_hash", "selection_size", "selection_index", "bundle_manifest_hash",
    "bundle_kind", "item_key", "judge_model", "judge_interface", "aggregate", "official_family",
    "knowledge_required", *SCORE_FIELDS, "Q", "rationale", "pass_number", "batch_id", "blind_id",
    "raw_response_json", "raw_response_sha256", "session_metadata", "session_metadata_sha256",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _parse_json(text, description):
    try:
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid JSON in {description}: {error}") from error


def _read_json(path):
    try:
        raw = Path(path).read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read UTF-8 JSON: {path}") from error
    return _parse_json(raw, str(path)), raw


def _exact_fields(value, fields, description):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{description} must contain exactly: {', '.join(sorted(fields))}")


def _nonempty(value, description):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a nonempty string")
    return value


def _session(value, policy, batch_id):
    _exact_fields(value, SESSION_FIELDS, f"Session {batch_id}")
    if value["judge_label"] != policy["judge_label"] or canonical_hash(value["ui_settings"]) != canonical_hash(policy["ui_settings"]):
        raise ValueError(f"Session {batch_id} does not match the frozen judge label/UI settings")
    _nonempty(value["conversation_url_or_id"], "conversation_url_or_id")
    timestamp = _nonempty(value["judged_at"], "judged_at")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Session {batch_id} judged_at must be an ISO datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Session {batch_id} judged_at must include a timezone")
    return value


def _quality(result, knowledge_required):
    applicable = SCORE_FIELDS if knowledge_required else SCORE_FIELDS[:3]
    for field in applicable:
        if type(result.get(field)) is not int or not 1 <= result[field] <= 5:
            raise ValueError(f"{field} must be an integer from 1 to 5 (booleans are invalid)")
    if not knowledge_required and result.get("knowledge_score") is not None:
        raise ValueError("knowledge_score must be null when knowledge is not applicable")
    if knowledge_required and result["instruction_score"] <= 2 and result["knowledge_score"] != 1:
        raise ValueError("The official rubric requires knowledge_score=1 when instruction_score<=2")
    return float(statistics.mean(25 * (result[field] - 1) for field in applicable))


def _response(value, batch):
    _exact_fields(value, {"batch_id", "results"}, "Response")
    if value["batch_id"] != batch["batch_id"] or not isinstance(value["results"], list):
        raise ValueError(f"Response must identify batch {batch['batch_id']} and contain a results list")
    expected = {item["blind_id"] for item in batch["items"]}
    results = {}
    for result in value["results"]:
        _exact_fields(result, RESULT_FIELDS, "Result")
        blind_id = _nonempty(result["blind_id"], "blind_id")
        if blind_id in results:
            raise ValueError(f"Duplicate blind_id in {batch['batch_id']}: {blind_id}")
        if result["status"] not in ("ok", "unreadable"):
            raise ValueError(f"Invalid result status for {blind_id}")
        if result["status"] == "unreadable":
            raise ValueError(f"Unreadable item {blind_id}; no score set was imported")
        _nonempty(result["rationale"], "rationale")
        results[blind_id] = result
    if set(results) != expected:
        raise ValueError(f"Missing/extra blind IDs in {batch['batch_id']}")
    return results


@contextmanager
def _output_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is publishing {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _publish(path, content):
    """Publish a full file atomically; never replace conflicting prior results."""
    path = Path(path)
    with _output_lock(path):
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Existing output conflicts with these results: {path}")
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".manual-results-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def import_results(args):
    manifest = load_bundle(args.bundle)
    if type(args.pass_number) is not int or args.pass_number < 1:
        raise ValueError("pass_number must be a positive integer")
    policy = manifest["evaluation_protocol"]
    if policy.get("aggregate") != AGGREGATE:
        raise ValueError("Unsupported manual score aggregation")
    all_batches = {batch["batch_id"]: batch for batch in manifest["batches"]}
    batches = [batch for batch in manifest["batches"] if batch["pass"] == args.pass_number]
    if not batches:
        raise ValueError("The bundle does not contain the requested pass")
    items = {item["item_key"]: item for item in manifest["items"]}
    positions = {item["item_key"]: index for index, item in enumerate(manifest["items"])}
    selected = [item["item_key"] for batch in batches for item in batch["items"]]
    if len(selected) != len(set(selected)) or set(selected) != set(items):
        raise ValueError("Every selected output must appear exactly once in this pass")
    responses = Path(args.responses)
    if not responses.is_dir():
        raise ValueError("Responses must be a directory of <batch_id>.json files")
    expected_names = {batch_id + ".json" for batch_id in all_batches}
    names = {path.name for path in responses.iterdir() if path.is_file() and path.suffix.lower() == ".json"}
    if names - expected_names:
        raise ValueError("Unexpected response JSON files; do not select among alternative/regenerated answers")
    if any(batch["batch_id"] + ".json" not in names for batch in batches):
        raise ValueError("Missing response files for the requested pass")
    sessions, _ = _read_json(args.session_log)
    if not isinstance(sessions, dict) or set(sessions) - set(all_batches):
        raise ValueError("Session log must map known batch IDs to operator-recorded metadata")
    if any(batch["batch_id"] not in sessions for batch in batches):
        raise ValueError("Session log is missing a requested batch")
    conversations = set()
    requested_batch_ids = {batch["batch_id"] for batch in batches}
    for batch_id, value in sessions.items():
        # The exported template includes both passes. A not-yet-started other
        # pass may retain its empty timestamp/conversation placeholders.
        if (batch_id not in requested_batch_ids and isinstance(value, dict)
                and value.get("judged_at") == "" and value.get("conversation_url_or_id") == ""):
            _exact_fields(value, SESSION_FIELDS, f"Session {batch_id}")
            if value["judge_label"] != policy["judge_label"] or canonical_hash(value["ui_settings"]) != canonical_hash(policy["ui_settings"]):
                raise ValueError(f"Session {batch_id} does not match the frozen judge label/UI settings")
            continue
        _session(value, policy, batch_id)
        conversation = value["conversation_url_or_id"].strip().rstrip("/")
        if conversation in conversations:
            raise ValueError("Use a distinct conversation for every batch and pass")
        conversations.add(conversation)
    rows = []
    for batch in batches:
        batch_id = batch["batch_id"]
        response, raw = _read_json(responses / (batch_id + ".json"))
        ratings = _response(response, batch)
        raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        for assignment in batch["items"]:
            item = items[assignment["item_key"]]
            generation, rating = item["row"], ratings[assignment["blind_id"]]
            q = _quality(rating, item["knowledge_required"])
            row = {key: generation[key] for key in ("example_uid", "condition", "base_seed", "output_sha256", "source_sha256")}
            row.update(schema_version=1, status="ok", protocol_hash=manifest["generation_protocol_hash"],
                       generation_protocol_hash=manifest["generation_protocol_hash"],
                       generation_snapshot=manifest["generation_snapshot"],
                       evaluation_protocol=policy, evaluation_protocol_hash=manifest["evaluation_protocol_hash"],
                       selection_hash=manifest["selection_hash"], selection_size=len(items),
                       selection_index=positions[item["item_key"]],
                       bundle_manifest_hash=manifest["manifest_hash"], bundle_kind=manifest["kind"],
                       item_key=item["item_key"], judge_model=policy["judge_label"], judge_interface="chatgpt_manual",
                       aggregate=AGGREGATE, official_family=item["official_family"],
                       knowledge_required=item["knowledge_required"], **{field: rating[field] for field in SCORE_FIELDS},
                       Q=q, rationale=rating["rationale"], pass_number=args.pass_number,
                       batch_id=batch_id, blind_id=assignment["blind_id"],
                       raw_response_json=raw, raw_response_sha256=raw_hash,
                       session_metadata=sessions[batch_id], session_metadata_sha256=canonical_hash(sessions[batch_id]))
            validate_score_row(row)
            rows.append(row)
    rows.sort(key=lambda row: row["item_key"])
    content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")
    _publish(args.output, content)
    report = {"outputs": len(rows), "pass_number": args.pass_number, "output": str(args.output),
              "evaluation_protocol_hash": manifest["evaluation_protocol_hash"], "selection_hash": manifest["selection_hash"]}
    print(json.dumps(report, indent=2))
    return report


def validate_score_row(row):
    """Validate an imported row, including its preserved response and operator log."""
    _exact_fields(row, SCORE_ROW_FIELDS, "Manual score row")
    if (type(row["schema_version"]) is not int or row["schema_version"] != 1 or row["status"] != "ok"
            or row["judge_interface"] != "chatgpt_manual" or row["bundle_kind"] not in ("calibration", "main")):
        raise ValueError("Unsupported manual score schema/status/interface")
    if type(row["base_seed"]) is not int or not isinstance(row["condition"], str) or row["condition"] not in tuple("ABCDEF"):
        raise ValueError("Invalid generation output key")
    for field in ("example_uid", "item_key", "batch_id", "blind_id", "rationale", "judge_model",
                  "output_sha256", "source_sha256", "bundle_manifest_hash", "selection_hash", "official_family"):
        _nonempty(row[field], field)
    if (type(row["selection_size"]) is not int or type(row["selection_index"]) is not int
            or not 0 <= row["selection_index"] < row["selection_size"]
            or type(row["pass_number"]) is not int or row["pass_number"] not in (1, 2)
            or row["bundle_kind"] == "main" and row["pass_number"] != 1):
        raise ValueError("Invalid frozen selection or judging pass")
    if type(row["knowledge_required"]) is not bool or row["knowledge_required"] != (row["official_family"] == "knowledge"):
        raise ValueError("Invalid knowledge applicability")
    q = _quality(row, row["knowledge_required"])
    if (isinstance(row["Q"], bool) or not isinstance(row["Q"], (int, float))
            or not math.isfinite(row["Q"]) or not math.isclose(row["Q"], q, rel_tol=0, abs_tol=1e-10)):
        raise ValueError("Q differs from the recorded component scores")
    policy = row["evaluation_protocol"]
    if (not isinstance(policy, dict) or canonical_hash(policy) != row["evaluation_protocol_hash"]
            or row["aggregate"] != AGGREGATE or policy.get("aggregate") != AGGREGATE
            or policy.get("interface") != "chatgpt_manual"):
        raise ValueError("Manual evaluation policy/hash/aggregation is inconsistent")
    if policy.get("adapter_files_sha256") != adapter_hashes():
        raise ValueError("Manual adapter implementation changed after the judging policy was frozen")
    _nonempty(policy.get("judge_label"), "judge_label")
    _nonempty(policy.get("ui_settings"), "ui_settings")
    if row["judge_model"] != policy["judge_label"] or row["protocol_hash"] != row["generation_protocol_hash"]:
        raise ValueError("Manual score identity is inconsistent")
    snapshot = row["generation_snapshot"]
    if (not isinstance(snapshot, dict) or snapshot.get("protocol_hash") != row["generation_protocol_hash"]
            or not {"config", "splits", "subset", "judge"} <= set(snapshot)
            or canonical_hash({key: snapshot[key] for key in ("config", "splits", "subset", "judge")}) != row["generation_protocol_hash"]):
        raise ValueError("Archived generation protocol checksum mismatch")
    session = _session(row["session_metadata"], policy, row["batch_id"])
    if canonical_hash(session) != row["session_metadata_sha256"]:
        raise ValueError("Session metadata hash is inconsistent")
    raw = row["raw_response_json"]
    if not isinstance(raw, str) or hashlib.sha256(raw.encode("utf-8")).hexdigest() != row["raw_response_sha256"]:
        raise ValueError("Raw response hash is inconsistent")
    response = _parse_json(raw, "preserved raw response")
    _exact_fields(response, {"batch_id", "results"}, "Preserved response")
    if response["batch_id"] != row["batch_id"] or not isinstance(response["results"], list):
        raise ValueError("Preserved response belongs to a different batch")
    identifiers, matched = set(), []
    for result in response["results"]:
        _exact_fields(result, RESULT_FIELDS, "Preserved result")
        blind_id = _nonempty(result["blind_id"], "blind_id")
        if blind_id in identifiers:
            raise ValueError("Duplicate blind ID in preserved response")
        identifiers.add(blind_id)
        if blind_id == row["blind_id"]:
            matched.append(result)
    if len(matched) != 1 or any(canonical_hash(matched[0][field]) != canonical_hash(row[field]) for field in (*SCORE_FIELDS, "rationale", "status")):
        raise ValueError("Score differs from its preserved raw response")
    return row


def _score_rows(path):
    try:
        lines = Path(path).read_bytes().decode("utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read score JSONL: {path}") from error
    rows = [_parse_json(line, str(path)) for line in lines if line.strip()]
    mapping, batches = {}, {}
    frozen = None
    shared_fields = ("generation_protocol_hash", "generation_snapshot", "evaluation_protocol_hash", "selection_hash",
                     "selection_size", "bundle_manifest_hash", "judge_model", "pass_number")
    for row in rows:
        validate_score_row(row)
        if row["bundle_kind"] != "calibration":
            raise ValueError("Repeatability requires imported manual calibration scores")
        key = (row["example_uid"], row["condition"], row["base_seed"])
        if key in mapping:
            raise ValueError("Repeatability scores must have unique output keys")
        identity = canonical_hash({field: row[field] for field in shared_fields})
        if frozen is None:
            frozen = identity
        elif identity != frozen:
            raise ValueError("Score file mixes frozen selections, policies or passes")
        batch_identity = (row["raw_response_sha256"], row["session_metadata_sha256"])
        if row["batch_id"] in batches and batches[row["batch_id"]] != batch_identity:
            raise ValueError("Rows from a batch have inconsistent raw responses/session metadata")
        batches[row["batch_id"]] = batch_identity
        mapping[key] = row
    if not mapping or any(type(row.get("selection_size")) is not int or row["selection_size"] != len(mapping) for row in mapping.values()):
        raise ValueError("Score file must contain the entire frozen selection")
    ordered = sorted(mapping.values(), key=lambda row: row["selection_index"])
    if [row["selection_index"] for row in ordered] != list(range(len(ordered))):
        raise ValueError("Invalid or duplicate frozen selection indices")
    records = selection_records([{"item_key": row["item_key"], "row": row} for row in ordered])
    if canonical_hash(records) != ordered[0]["selection_hash"]:
        raise ValueError("Selected output keys/hashes differ from the frozen selection")
    conversation_owner = {}
    for row in ordered:
        conversation = row["session_metadata"]["conversation_url_or_id"].strip().rstrip("/")
        if conversation in conversation_owner and conversation_owner[conversation] != row["batch_id"]:
            raise ValueError("Distinct batches must use independent conversations")
        conversation_owner[conversation] = row["batch_id"]
    return mapping


def repeatability(args):
    import numpy as np
    first, second = _score_rows(args.first), _score_rows(args.second)
    if set(first) != set(second) or len(first) < 20:
        raise ValueError("Re-score the same predetermined set of at least 20 outputs")
    compared = ("generation_protocol_hash", "generation_snapshot", "output_sha256", "source_sha256",
                "evaluation_protocol", "evaluation_protocol_hash", "selection_hash", "selection_size",
                "bundle_manifest_hash", "judge_model", "judge_interface", "aggregate", "official_family",
                "knowledge_required", "item_key")
    differences = []
    conversations = [set(), set()]
    for key in sorted(first):
        a, b = first[key], second[key]
        if type(a.get("pass_number")) is not int or type(b.get("pass_number")) is not int or (a["pass_number"], b["pass_number"]) != (1, 2):
            raise ValueError("Repeatability compares independent pass 1 against pass 2")
        for field in compared:
            if a.get(field) is None or canonical_hash(a[field]) != canonical_hash(b.get(field)):
                raise ValueError(f"Repeatability mismatch in {field}: {key}")
        for index, row in enumerate((a, b)):
            conversations[index].add(row["session_metadata"]["conversation_url_or_id"].strip().rstrip("/"))
        differences.append(abs(a["Q"] - b["Q"]))
    if conversations[0] & conversations[1]:
        raise ValueError("The two passes must use independent conversations")
    example = next(iter(first.values()))
    report = {"outputs": len(first), "method": "percentile95_absolute_rescore_Q_difference",
              "judge_interface": "chatgpt_manual", "judge_model": example["judge_model"],
              "generation_protocol_hash": example["generation_protocol_hash"],
              "bundle_manifest_hash": example["bundle_manifest_hash"],
              "first_scores_sha256": file_sha256(args.first), "second_scores_sha256": file_sha256(args.second),
              "evaluation_protocol": example["evaluation_protocol"],
              "evaluation_protocol_hash": example["evaluation_protocol_hash"], "selection_hash": example["selection_hash"],
              "candidate_minimum_meaningful_difference": float(np.quantile(differences, .95)),
              "mean_absolute_difference": float(np.mean(differences)),
              "note": "Manual ChatGPT UI repeatability; choose and freeze delta >= this MMD before test scoring. UI labels do not identify immutable API model snapshots."}
    _publish(args.output, (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps(report, indent=2))
    return report

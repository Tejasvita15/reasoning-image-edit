"""Export frozen RQ3 images for blinded, manual ChatGPT UI assessment.

This module deliberately lives outside ``src/rq3``: exporting existing images
must not alter the implementation hash that identifies their generation run.
"""

from __future__ import annotations

from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import zipfile

from src.rq3.evaluate import OFFICIAL_REVISION
from src.rq3.protocol import REPO_ROOT, build_jobs, validate_config, validate_splits
from .references import PreparedReferenceEvaluator


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing required archived artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid archived JSON: {path}") from exc


def _row_key(row):
    uid, condition, seed = (row.get(name) for name in ("example_uid", "condition", "base_seed"))
    if not isinstance(uid, str) or condition not in tuple("ABCDEF") or type(seed) is not int:
        raise ValueError("Generation has an invalid example/condition/seed identity")
    return uid, condition, seed


def _load_archive(generations):
    """Verify the archived payload against its original hash, never a new one."""
    from .common import canonical_hash

    generations = Path(generations).resolve()
    snapshot = _read_json(generations.parent / "protocol_snapshot.json")
    fields = {"protocol_hash", "config", "splits", "subset", "judge"}
    if not isinstance(snapshot, dict) or set(snapshot) != fields:
        raise ValueError("Archived protocol snapshot has an unexpected schema")
    payload = {name: snapshot[name] for name in fields - {"protocol_hash"}}
    if canonical_hash(payload) != snapshot["protocol_hash"]:
        raise ValueError("Archived protocol snapshot hash does not match its contents")
    if any(not isinstance(snapshot[name], dict) for name in fields - {"protocol_hash"}):
        raise ValueError("Archived protocol snapshot fields must be objects")
    public = {**snapshot["config"], "protocol_hash": snapshot["protocol_hash"]}
    if _read_json(generations.parent / "protocol.json") != public:
        raise ValueError("Archived protocol.json differs from the frozen snapshot")
    validate_config(snapshot["config"])
    validate_splits(snapshot["splits"], snapshot["subset"])
    if snapshot["subset"].get("source_splits_sha256") != snapshot["config"]["data"].get("splits_sha256"):
        raise ValueError("Archived subset belongs to a different frozen split")
    implementation = snapshot["config"].get("implementation_files_sha256")
    required = {"src/rq3/backend.py", "src/rq3/run.py", "src/rq3/protocol.py", "src/rq3/regione.py"}
    if (not isinstance(implementation, dict) or not required <= implementation.keys()
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in implementation.values())):
        raise ValueError("Archived generation implementation hashes are incomplete or invalid")
    if snapshot["judge"].get("official_revision") != OFFICIAL_REVISION:
        raise ValueError("The archived official evaluator revision is not supported")
    if snapshot["judge"].get("prompt_file") != "utils/prompts.py":
        raise ValueError("The archived official prompt path is not supported")
    rows = []
    try:
        with generations.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid generation JSON at line {number}") from exc
                if (not isinstance(row, dict) or row.get("protocol_hash") != snapshot["protocol_hash"]
                        or row.get("config") != public):
                    raise ValueError(f"Generation protocol/configuration differs from its archive at line {number}")
                _row_key(row)
                if (type(row.get("repeat")) is not int or row["repeat"] < 0
                        or row.get("phase") not in {"pilot", "all"}
                        or row.get("status") not in {"ok", "error"}):
                    raise ValueError(f"Invalid generation repeat/phase/status at line {number}")
                split = snapshot["splits"]["k_dev" if row["phase"] == "pilot" else "k_small"]
                if row["example_uid"] not in split:
                    raise ValueError("Generation example is outside its archived split")
                if (row["phase"] == "all" and row["condition"] in "EF"
                        and row["example_uid"] not in snapshot["subset"]["example_uids"]):
                    raise ValueError("Generation ablation is outside its archived subset")
                rows.append(row)
    except (FileNotFoundError, UnicodeError) as exc:
        raise ValueError(f"Missing or unreadable generation log: {generations}") from exc
    return snapshot, rows


def _select_rows(snapshot, rows, *, kind, seeds, limit):
    config = {**snapshot["config"], "_splits": snapshot["splits"], "_subset": snapshot["subset"]}
    expected = {_row_key(job) for job in build_jobs(config, phase="all", seeds=seeds)}
    phase = "pilot" if kind == "calibration" else "all"
    quality = [row for row in rows if row["repeat"] == 0 and row["base_seed"] in seeds]
    if any(row["phase"] != phase for row in quality):
        raise ValueError(f"{kind} export requires only {phase} quality generations")
    # Choose before resolving errors so failed attempts cannot shift the sample.
    keys = list(dict.fromkeys(_row_key(row) for row in quality))
    if kind == "calibration":
        if type(limit) is not int or limit < 20:
            raise ValueError("Calibration needs a predetermined sample of at least 20 outputs")
        if len(keys) < limit:
            raise ValueError(f"Calibration has fewer than the requested {limit} distinct outputs")
        keys = keys[:limit]
    elif set(keys) != expected:
        raise ValueError(f"Main export requires all 200 quality outputs per seed: missing={len(expected-set(keys))}, extra={len(set(keys)-expected)}")
    selected = set(keys)
    complete, latest = {}, {}
    for row in quality:
        key = _row_key(row)
        if key not in selected:
            continue
        latest[key] = row["status"]
        if row["status"] == "ok":
            if key in complete:
                raise ValueError("Duplicate successful generation in the selected batch")
            complete[key] = row
    if set(complete) != selected or any(value != "ok" for value in latest.values()):
        raise ValueError("Selected generation has an unresolved failure; selection will not skip it")
    return [complete[key] for key in keys]


def _verify_images(row):
    from PIL import Image
    from .common import file_sha256

    for name in ("source", "output"):
        value, expected = row.get(f"{name}_path"), row.get(f"{name}_sha256")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Generation images must retain their original absolute paths")
        path = Path(value)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Generation {name} image is missing or changed")
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError("Frozen RQ3 images must be PNG")
                image.verify()
        except OSError as exc:
            raise ValueError(f"Generation {name} image is not a valid PNG") from exc


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _calibration_report(args, policy_hash, config):
    path = getattr(args, "calibration_report", None)
    if not path:
        raise ValueError("Main export requires a matching manual calibration repeatability report")
    report = _read_json(path)
    if not isinstance(report, dict) or report.get("evaluation_protocol_hash") != policy_hash:
        raise ValueError("Manual calibration report uses a different evaluation policy")
    if (report.get("judge_interface") != "chatgpt_manual"
            or report.get("method") != "percentile95_absolute_rescore_Q_difference"
            or type(report.get("outputs")) is not int or report["outputs"] < 20):
        raise ValueError("Main export needs a completed manual calibration report for at least 20 outputs")
    candidate = report.get("candidate_minimum_meaningful_difference")
    stats = config.get("statistics", {})
    mmd, margin = stats.get("minimum_meaningful_difference"), stats.get("equivalence_margin")
    values = (candidate, mmd, margin)
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values) or not 0 <= candidate <= mmd <= margin):
        raise ValueError("Freeze finite delta >= MMD >= the manual repeatability candidate before main generation/export")
    return report


def _readme(kind):
    repeats = ("There are two independent calibration passes. Finish pass 1, then use new chats for pass 2. "
               "Do not show pass-1 answers to pass 2.\n\n") if kind == "calibration" else ""
    return ("# Manual ChatGPT UI judging bundle\n\n"
            "Keep this ZIP on your computer. **Do not upload the ZIP, private/ directory, this README, "
            "or any experiment logs to ChatGPT.** The private manifest reveals condition assignments.\n\n"
            "For each batch under upload/, start a fresh chat using the recorded judge label and UI settings. "
            "Attach all image files in that batch (source, edited, and any reference), then paste its prompt.md. "
            "Keep each filename visible so the source and edited image can be paired correctly. "
            "Do not add model prompts, thinking traces, conditions, seeds or previous ratings.\n\n"
            + repeats +
            "Save the complete JSON answer, unchanged, to a separate responses directory using the batch ID "
            "as its filename (for example p1_b001.json). Preserve the chat transcript separately for audit. "
            "Record the actual UI model/settings and evaluation date when importing responses. "
            "Do not edit upload files or the private manifest after export.\n\n"
            "These are manual ChatGPT UI ratings using adapted official rubrics, not the pinned API judge. "
            "The bundle preserves the original generation protocol and records a separate evaluation policy. "
            "Saved source images and any ground-truth references are verified against the pinned prepared "
            "Hugging Face dataset used by the experiment. Differences from the official GitHub image copy "
            "are recorded in private/manifest.json as reference_provenance; they are not hidden by resizing.\n")


def export_bundle(args):
    """Create a complete directory and transport ZIP without replacing outputs."""
    from .common import (build_prompt, canonical_hash, file_sha256,
                         make_policy, response_template)

    kind = args.kind
    if kind not in {"calibration", "main"}:
        raise ValueError("kind must be calibration or main")
    if type(args.batch_size) is not int or not 1 <= args.batch_size <= 4:
        raise ValueError("Use batches of 1..4 outputs (two or three images per output)")
    if not isinstance(args.judge_label, str) or not args.judge_label.strip():
        raise ValueError("Record the ChatGPT UI judge label before exporting")
    if not isinstance(args.ui_settings, str) or not args.ui_settings.strip():
        raise ValueError("Record the ChatGPT UI settings before exporting")
    prepared_data = getattr(args, "prepared_data", None) or os.environ.get("RQ3_PREPARED_DATA")
    if not prepared_data:
        raise ValueError("Set RQ3_PREPARED_DATA or pass --prepared-data to verify the pinned generation reference")
    generations = Path(args.generations).resolve()
    output = Path(args.output_dir).resolve()
    zip_path = output.with_name(output.name + ".zip")
    if output.exists() or zip_path.exists():
        raise ValueError("Export directory or ZIP already exists; choose a fresh output directory")
    if output == generations.parent or output in generations.parent.parents:
        raise ValueError("Export must not replace the generation directory or its parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name("." + output.name + ".export.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("An export lock exists; inspect any active exporter before retrying") from exc
    os.close(lock_fd)
    try:
        with ExitStack() as stack:
            run_lock = generations.parent / ".run.lock"
            if run_lock.exists():
                handle = stack.enter_context(run_lock.open("r"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError("Generation run is still active; export only completed work") from exc
            snapshot, rows = _load_archive(generations)
            log_hash = file_sha256(generations)
            selected = _select_rows(snapshot, rows, kind=kind, seeds=args.seeds, limit=args.limit)
            judge = dict(snapshot["judge"])
            judge["official_checkout"] = str((REPO_ROOT / judge["official_checkout"]).resolve())
            evaluator = PreparedReferenceEvaluator(judge, snapshot, prepared_data,
                                                   [row["example_uid"] for row in selected])
            items, ground_truth_paths = [], {}
            for index, row in enumerate(selected, 1):
                _verify_images(row)
                category, _, annotation = evaluator.annotation(row)
                family = evaluator.family(category)
                item = {"item_key": f"item_{index:04d}", "row": row,
                        "official_family": family, "knowledge_required": family == "knowledge",
                        "reference_provenance": evaluator.reference_provenance(row)}
                if family == "view":
                    path, digest = evaluator.ground_truth(row)
                    ground_truth_paths[item["item_key"]] = path
                    item["ground_truth_sha256"] = digest
                items.append(item)
            policy = make_policy(args.judge_label, args.ui_settings, evaluator)
            policy["batch_size"] = args.batch_size
            selection = [{"item_key": item["item_key"], **{name: item["row"][name]
                          for name in ("example_uid", "condition", "base_seed", "output_sha256")}}
                         for item in items]
            selection_hash = canonical_hash(selection)
            manifest = {"schema_version": 1, "kind": kind,
                        "generation_protocol_hash": snapshot["protocol_hash"], "generation_snapshot": snapshot,
                        "evaluation_protocol": policy, "evaluation_protocol_hash": canonical_hash(policy),
                        "selection_hash": selection_hash, "seeds": list(args.seeds),
                        "items": items, "batches": [], "files": {}, "generations_sha256": log_hash,
                        "prepared_data_provenance": evaluator.data_provenance}
            if kind == "main":
                manifest["calibration_report"] = _calibration_report(args, manifest["evaluation_protocol_hash"], snapshot["config"])
            items_by_key = {item["item_key"]: item for item in items}
            with tempfile.TemporaryDirectory(prefix="." + output.name + ".staging-", dir=output.parent) as work:
                work = Path(work)
                staging = work / "bundle"
                (staging / "private").mkdir(parents=True, mode=0o700)
                for pass_number in range(1, 3 if kind == "calibration" else 2):
                    ordering = list(items)
                    random.Random(int(canonical_hash({"selection": selection_hash, "pass": pass_number}), 16)).shuffle(ordering)
                    for start in range(0, len(ordering), args.batch_size):
                        batch_id = f"p{pass_number}_b{start // args.batch_size + 1:03d}"
                        directory = Path("upload") / f"pass_{pass_number}" / batch_id
                        (staging / directory).mkdir(parents=True)
                        batch = {"pass": pass_number, "batch_id": batch_id,
                                 "prompt_path": (directory / "prompt.md").as_posix(), "items": []}
                        for item in ordering[start:start + args.batch_size]:
                            blind_id = "case_" + canonical_hash({"selection": selection_hash, "pass": pass_number,
                                                               "item": item["item_key"]})[:16]
                            entry = {"blind_id": blind_id, "item_key": item["item_key"]}
                            for name, source_name in (("source", "source"), ("edited", "output")):
                                relative = directory / f"{blind_id}_{name}.png"
                                shutil.copyfile(item["row"][f"{source_name}_path"], staging / relative)
                                digest = file_sha256(staging / relative)
                                if digest != item["row"][f"{source_name}_sha256"]:
                                    raise ValueError("Image changed while constructing the export")
                                entry[f"{name}_file"] = relative.as_posix()
                                manifest["files"][relative.as_posix()] = digest
                            if item["item_key"] in ground_truth_paths:
                                path = ground_truth_paths[item["item_key"]]
                                relative = directory / f"{blind_id}_reference{path.suffix.lower()}"
                                shutil.copyfile(path, staging / relative)
                                digest = file_sha256(staging / relative)
                                if digest != item["ground_truth_sha256"]:
                                    raise ValueError("Official ground-truth image changed while constructing the export")
                                entry["ground_truth_file"] = relative.as_posix()
                                manifest["files"][relative.as_posix()] = digest
                            batch["items"].append(entry)
                        prompt = build_prompt(batch, items_by_key, evaluator)
                        (staging / batch["prompt_path"]).write_text(prompt, encoding="utf-8")
                        manifest["files"][batch["prompt_path"]] = file_sha256(staging / batch["prompt_path"])
                        template = directory / "response_template.json"
                        _write_json(staging / template, response_template(batch))
                        manifest["files"][template.as_posix()] = file_sha256(staging / template)
                        manifest["batches"].append(batch)
                if file_sha256(generations) != log_hash:
                    raise ValueError("Generation log changed while constructing the export")
                manifest["manifest_hash"] = canonical_hash(manifest)
                _write_json(staging / "private/manifest.json", manifest)
                _write_json(staging / "private/generation_protocol_snapshot.json", snapshot)
                _write_json(staging / "session_log_template.json", {
                    batch["batch_id"]: {"judge_label": policy["judge_label"], "ui_settings": policy["ui_settings"],
                                        "judged_at": "", "conversation_url_or_id": ""}
                    for batch in manifest["batches"]})
                (staging / "README.md").write_text(_readme(kind), encoding="utf-8")
                staged_zip = work / "bundle.zip"
                with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for path in sorted(staging.rglob("*")):
                        if path.is_file():
                            archive.write(path, path.relative_to(staging).as_posix())
                staged_zip.chmod(0o600)
                # Hard-link publication fails atomically if a ZIP already exists.
                # Both temporary paths live on the output filesystem.
                if output.exists():
                    raise ValueError("Export destination appeared while constructing the bundle")
                os.link(staged_zip, zip_path)
                try:
                    staging.rename(output)
                except Exception:
                    zip_path.unlink()
                    raise
            result = {"output_dir": str(output), "zip_path": str(zip_path), "kind": kind,
                      "selected_outputs": len(items), "batches": len(manifest["batches"]),
                      "generation_protocol_hash": snapshot["protocol_hash"],
                      "evaluation_protocol_hash": manifest["evaluation_protocol_hash"],
                      "manifest_hash": manifest["manifest_hash"],
                      "official_source_pixel_mismatches": sum(
                          not item["reference_provenance"]["source"]["official_pixels_equal"] for item in items)}
            print(json.dumps(result, indent=2))
            return result
    finally:
        lock_path.unlink(missing_ok=True)

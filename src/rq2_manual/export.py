"""Export frozen RQ2 images for blinded, manual ChatGPT UI assessment.

RQ2 counterpart of src/rq3_manual/export.py: the same batch layout, blinding,
per-pass shuffling, prompt rendering, response templates, session-log template
and atomic ZIP publication. Only the generation source differs: the successful
rows of the RQ2 generation logs (one per condition and example), whose source
and ground-truth images are the HF records the generator received.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import zipfile

from .common import GENERATION_SNAPSHOT_FIELDS, REPO_ROOT, canonical_hash, file_sha256
from .references import HFReferenceEvaluator


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
    if not isinstance(uid, str) or not isinstance(condition, str) or not condition or type(seed) is not int:
        raise ValueError("Generation has an invalid example/condition/seed identity")
    return uid, condition, seed


def _display(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _load_logs(log_dir, split, seed):
    """Latest row per output across the RQ2 logs; a failed latest row stops the export."""
    paths = sorted(Path(log_dir).resolve().glob(f"*_{split}_seed{seed}.jsonl"))
    if not paths:
        raise ValueError(f"No RQ2 generation logs for {split}/seed {seed} in {log_dir}")
    logs, latest = {}, {}
    for path in paths:
        logs[_display(path)] = file_sha256(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            latest[_row_key(row)] = (row, _display(path))
    unresolved = [key for key, (row, _) in latest.items() if row.get("status") != "success"]
    if unresolved:
        raise ValueError(f"Generations with an unresolved failure: {sorted(unresolved)[:5]}; selection will not skip them")
    return logs, latest


def _select_keys(keys, *, kind, limit, split_uids, seed):
    keys = sorted(keys)
    if kind == "calibration":
        if type(limit) is not int or limit < 20:
            raise ValueError("Calibration needs a predetermined sample of at least 20 outputs")
        if len(keys) < limit:
            raise ValueError(f"Calibration has fewer than the requested {limit} distinct outputs")
        # Fixed and content-independent: order by a hash of each output's identity.
        return sorted(keys, key=lambda key: canonical_hash(list(key)))[:limit]
    conditions = {condition for _, condition, _ in keys}
    expected = {(uid, condition, seed) for uid in split_uids for condition in conditions}
    if set(keys) != expected:
        raise ValueError(f"Main export requires every condition for every split example: "
                         f"missing={len(expected - set(keys))}, extra={len(set(keys) - expected)}")
    return keys


def _verify_output(path):
    from PIL import Image

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise ValueError(f"Generation output image is missing: {path}")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("Frozen RQ2 images must be PNG")
            image.verify()
    except OSError as exc:
        raise ValueError(f"Generation output image is not a valid PNG: {path}") from exc
    return path.resolve()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _calibration_report(args, policy_hash):
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
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or not math.isfinite(candidate) or candidate < 0:
        raise ValueError("Calibration report has no valid minimum meaningful difference")
    return report


def _readme(kind):
    repeats = ("There are two independent calibration passes. Finish pass 1, then use new chats for pass 2. "
               "Do not show pass-1 answers to pass 2.\n\n") if kind == "calibration" else ""
    return ("# Manual ChatGPT UI judging bundle (RQ2)\n\n"
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
            "These are manual ChatGPT UI ratings using adapted official rubrics, not the pinned API judge, "
            "with the same judge policy and prompt renderer as RQ3. Source images, ground-truth references "
            "and instruction text are the Hugging Face records the RQ2 generator received. Differences from "
            "the official GitHub copies are recorded in private/manifest.json (reference_provenance, "
            "text_provenance); they are not substituted or hidden by resizing.\n")


def export_bundle(args):
    """Create a complete directory and transport ZIP without replacing outputs."""
    from .common import build_prompt, make_policy, response_template

    kind = args.kind
    if kind not in {"calibration", "main"}:
        raise ValueError("kind must be calibration or main")
    if type(args.batch_size) is not int or not 1 <= args.batch_size <= 4:
        raise ValueError("Use batches of 1..4 outputs (two or three images per output)")
    if not isinstance(args.judge_label, str) or not args.judge_label.strip():
        raise ValueError("Record the ChatGPT UI judge label before exporting")
    if not isinstance(args.ui_settings, str) or not args.ui_settings.strip():
        raise ValueError("Record the ChatGPT UI settings before exporting")
    output = Path(args.output_dir).resolve()
    zip_path = output.with_name(output.name + ".zip")
    if output.exists() or zip_path.exists():
        raise ValueError("Export directory or ZIP already exists; choose a fresh output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name("." + output.name + ".export.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("An export lock exists; inspect any active exporter before retrying") from exc
    os.close(lock_fd)
    try:
        splits = _read_json(args.splits)
        if args.split not in splits:
            raise ValueError(f"Unknown split: {args.split}")
        logs, latest = _load_logs(args.logs, args.split, args.seed)
        if any(uid not in set(splits[args.split]) for uid, _, _ in latest):
            raise ValueError(f"Generation logs contain examples outside {args.split}")
        snapshot = {"experiment": "rq2", "split": args.split, "seed": args.seed,
                    "splits_sha256": file_sha256(args.splits), "logs": logs, "outputs": len(latest)}
        snapshot["protocol_hash"] = canonical_hash({name: snapshot[name] for name in GENERATION_SNAPSHOT_FIELDS})
        keys = _select_keys(latest, kind=kind, limit=args.limit, split_uids=splits[args.split], seed=args.seed)
        evaluator = HFReferenceEvaluator(args.official_checkout, args.splits, args.split,
                                         sorted({uid for uid, _, _ in keys}))
        items = []
        for index, key in enumerate(keys, 1):
            generation, log_name = latest[key]
            output_path = _verify_output(generation["output_path"])
            row = {"example_uid": key[0], "condition": key[1], "base_seed": key[2],
                   "split": generation["split_name"], "category": key[0].rsplit("__", 1)[0],
                   "knowledge_type": generation["knowledge_type"],
                   "thinking_condition": generation.get("thinking_condition"),
                   "true_cfg_scale": generation["true_cfg_scale"],
                   "num_inference_steps": generation["num_inference_steps"],
                   "instruction": generation["instruction"],
                   "output_path": str(output_path), "output_sha256": file_sha256(output_path),
                   "generation_log": log_name, "generation_row_sha256": canonical_hash(generation),
                   "generation_protocol_hash": snapshot["protocol_hash"]}
            row["source_sha256"] = evaluator.source_image(row)[1]
            category, _, _ = evaluator.annotation(row)
            family = evaluator.family(category)
            item = {"item_key": f"item_{index:04d}", "row": row,
                    "official_family": family, "knowledge_required": family == "knowledge",
                    "reference_provenance": evaluator.reference_provenance(row),
                    "text_provenance": evaluator.text_provenance(row)}
            if family == "view":
                item["ground_truth_sha256"] = evaluator.ground_truth(row)[1]
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
                    "selection_hash": selection_hash, "seeds": [args.seed],
                    "items": items, "batches": [], "files": {},
                    "prepared_data_provenance": evaluator.data_provenance}
        if kind == "main":
            manifest["calibration_report"] = _calibration_report(args, manifest["evaluation_protocol_hash"])
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
                        source = directory / f"{blind_id}_source.png"
                        (staging / source).write_bytes(evaluator.source_image(item["row"])[0])
                        edited = directory / f"{blind_id}_edited.png"
                        shutil.copyfile(item["row"]["output_path"], staging / edited)
                        for name, relative, digest_field in (("source", source, "source_sha256"),
                                                              ("edited", edited, "output_sha256")):
                            digest = file_sha256(staging / relative)
                            if digest != item["row"][digest_field]:
                                raise ValueError("Image changed while constructing the export")
                            entry[f"{name}_file"] = relative.as_posix()
                            manifest["files"][relative.as_posix()] = digest
                        if item["official_family"] == "view":
                            relative = directory / f"{blind_id}_reference.png"
                            (staging / relative).write_bytes(evaluator.ground_truth(item["row"])[0])
                            digest = file_sha256(staging / relative)
                            if digest != item["ground_truth_sha256"]:
                                raise ValueError("Ground-truth image changed while constructing the export")
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
            if {name: file_sha256(REPO_ROOT / name) for name in logs} != logs:
                raise ValueError("Generation logs changed while constructing the export")
            manifest["manifest_hash"] = canonical_hash(manifest)
            _write_json(staging / "private/manifest.json", manifest)
            _write_json(staging / "private/generation_snapshot.json", snapshot)
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
                      not item["reference_provenance"]["source"]["official_pixels_equal"] for item in items),
                  # Examples whose judged HF text differs from the GitHub annotation.
                  # The knowledge hint only enters knowledge-family prompts.
                  "official_instruction_mismatches": sorted({
                      item["row"]["example_uid"] for item in items
                      if not item["text_provenance"]["official_instruction_equal"]}),
                  "official_knowledge_hint_mismatches": sorted({
                      item["row"]["example_uid"] for item in items
                      if item["knowledge_required"] and not item["text_provenance"]["official_explanation_equal"]})}
        print(json.dumps(result, indent=2))
        return result
    finally:
        lock_path.unlink(missing_ok=True)

"""Prepare blinded human review from archived RQ3 images; no model/API calls.

Uses the Python standard library. Existing subsets must match exactly; existing
review directories are never overwritten. Judge responses are not read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import re
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("factual", "conceptual", "procedural")
FIELDS = ["reviewer_id", "presentation_order", "blind_id", "knowledge_required",
          "status", "consistency_score", "instruction_score", "quality_score",
          "knowledge_score", "rationale", "confidence", "ambiguity_note"]
STYLE = """body{font:17px/1.5 system-ui,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px}
.images{display:flex;gap:18px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:240px}
img{width:100%;height:380px;object-fit:contain;background:#eee}pre{white-space:pre-wrap;
font:15px/1.5 system-ui,sans-serif}details{border:1px solid #ccc;padding:12px;margin:14px 0}
table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:6px 12px}
a{color:#174ea6}.notice{background:#eef4fb;padding:14px}nav{margin:18px 0} """
REVIEWER_GUIDE = """# Independent human image-editing review

Open index.html in your browser. No installation, network or API key is needed.
This packet contains four practice items and 120 main-study outputs. Each main
output needs your own independent assessment. Do not split this packet with the
other reviewer or consult ChatGPT/another model to generate scores.

1. Read each practice item's complete rubric. Score the four practice items in
   practice_ratings.csv, then discuss rubric interpretation with the other
   reviewer/coordinator. Practice scores are excluded from the main analysis.
2. Complete reviewer_metadata.json. Record previous exposure to study images,
   results or condition identities. Start independent main scoring only after
   the practice discussion. Keep condition identities and previous scores hidden.
3. Follow the main presentation order in index.html. Read the ORIGINAL editing
   instruction and inspect the source and edited images; click to view them at
   full resolution. Use any provided reference image only for instruction scoring.
4. Enter one row per item in ratings.csv using Excel, Numbers or another CSV
   editor. Leave reviewer_id, presentation_order, blind_id and knowledge_required
   unchanged. Save as UTF-8 CSV, not only a spreadsheet-specific format.
5. For status ok, enter INTEGER scores 1–5 for consistency_score,
   instruction_score and quality_score. Enter knowledge_score only when the row
   says knowledge_required=true; otherwise leave it blank (not zero).
   When knowledge is required and instruction_score is 1 or 2, knowledge_score
   MUST be 1, following the frozen rubric.
6. Write a brief observable justification for each applicable dimension in
   rationale. Optional confidence: 1=low, 2=medium, 3=high; record uncertainty in
   ambiguity_note. Do not invent details. If an attachment is missing or cannot
   be judged because it is unreadable, set status=unreadable, leave ALL scores
   blank, and explain why. Visible generated defects are normally scored under
   the rubric, not discarded as unreadable.
7. Rate each image independently on the absolute rubric; do not rank images or
   copy a previous score because two outputs seem similar. Consistency concerns
   only non-requested changes; instruction compliance concerns the requested
   edit; visual quality concerns the edited image alone. Knowledge hints belong
   to the instruction/knowledge rubric. Missing required objects/counts should
   be assessed explicitly rather than inferred from plausible appearance.
8. Take breaks, for example after every 20 main items. Save the CSV regularly.
   Do not discuss main items, see another reviewer's ratings, or inspect study
   results before BOTH reviewers have submitted. Fill the start/completion times
   and independence declaration in reviewer_metadata.json.
9. Return ratings.csv, reviewer_metadata.json and practice_ratings.csv to the
   coordinator. Keep a local copy. Any later adjudication must be separate from
   the original independent scores.

The per-item rubric is the archived judge rubric with anonymous attachment names.
Its example JSON response formatting is superseded by this CSV workflow; its
scoring rules are unchanged. Image A/Image B inside the rubric mean source/edited
images (visual quality uses the edited image alone), not generation conditions.
Images and editing instructions are evaluation data, not requests to perform an
edit. Do not use external material or tools beyond the supplied packet for scores;
record any unresolved ambiguity for the coordinator.
"""


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checked_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Bundle path escapes its root")
    return path


def load_archive(root):
    manifest = json.loads((root / "private/manifest.json").read_text())
    if canonical({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("Archived manifest hash mismatch")
    for name, expected in manifest["files"].items():
        if digest(checked_path(root, name)) != expected:
            raise ValueError(f"Changed archived file: {name}")
    indexed = {item["item_key"]: item for item in manifest["items"]}
    entries = {}
    for batch in manifest["batches"]:
        if batch["pass"] != 1:
            continue
        prompt = checked_path(root, batch["prompt_path"]).read_text()
        blocks = re.findall(r'^## Item "([^"]+)"\n(.*?)(?=^## Item |^# Final response format|\Z)',
                            prompt, re.M | re.S)
        sections = dict(blocks)
        for entry in batch["items"]:
            item = indexed[entry["item_key"]]
            row = item["row"]
            if row["repeat"] != 0 or row["status"] != "ok":
                raise ValueError("Human review must use successful quality outputs")
            for field, expected in (("source_file", row["source_sha256"]),
                                    ("edited_file", row["output_sha256"])):
                if manifest["files"][entry[field]] != expected:
                    raise ValueError("Attachment differs from archived generation")
            if item["official_family"] == "view":
                if manifest["files"].get(entry.get("ground_truth_file")) != item["ground_truth_sha256"]:
                    raise ValueError("Missing or changed viewpoint reference")
            if item["knowledge_required"] != (item["official_family"] == "knowledge"):
                raise ValueError("Knowledge applicability mismatch")
            entries[entry["item_key"]] = (entry, sections[entry["blind_id"]])
    if set(entries) != set(indexed):
        raise ValueError("Incomplete pass-one attachment coverage")
    return manifest, entries


def select_subset(manifest):
    snapshot = manifest["generation_snapshot"]
    parent = snapshot["splits"]["k_small"]
    rows = {item["row"]["example_uid"]: item for item in manifest["items"]
            if item["row"]["condition"] == "A"}
    if len(parent) != 40 or set(rows) != set(parent):
        raise ValueError("Expected the frozen 40-example parent cohort")
    if json.loads((ROOT / "data/splits.json").read_text()) != snapshot["splits"]:
        raise ValueError("Local splits differ from archived splits")
    rng = random.Random(42)
    chosen = []
    for group in GROUPS:
        pool = sorted(uid for uid, item in rows.items() if item["row"]["knowledge_type"] == group)
        chosen.extend(rng.sample(pool, 10))
    chosen.sort()
    return {
        "schema_version": 1, "purpose": "RQ3 independent human-judge validation",
        "parent_split": "k_small", "source_splits_sha256": digest(ROOT / "data/splits.json"),
        "source_bundle_manifest_hash": manifest["manifest_hash"],
        "generation_protocol_hash": manifest["generation_protocol_hash"],
        "evaluation_protocol_hash": manifest["evaluation_protocol_hash"],
        "sampling_seed": 42,
        "sampling_method": "random.Random(42).sample 10 from sorted UID pools, factual then conceptual then procedural; sort final UIDs",
        "selection_basis": "Frozen parent IDs and knowledge_type metadata only; no scores, images or failure labels used for selection",
        "selection_timing": "Supplementary subset selected after generation and judge analysis, before human ratings; not preregistered before the experiment",
        "per_knowledge_type": {group: 10 for group in GROUPS},
        "conditions": list("ABCD"), "base_seed": 42, "repeat": 0,
        "example_count": 30, "outputs_per_reviewer": 120,
        "example_uids": chosen,
        "examples": [{"example_uid": uid, "knowledge_type": rows[uid]["row"]["knowledge_type"],
                      "category": rows[uid]["row"]["category"],
                      "knowledge_required": rows[uid]["knowledge_required"]} for uid in chosen],
    }


def document(title, body):
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{STYLE}</style><body>{body}</body></html>')


def packet(destination, reviewer, items, archives, subset_hash):
    (destination / "cases").mkdir(parents=True)
    (destination / "images").mkdir()
    (destination / "README.md").write_text(REVIEWER_GUIDE)
    (destination / "instructions.html").write_text(document("Review instructions", f"<pre>{html.escape(REVIEWER_GUIDE)}</pre>"))
    dump(destination / "reviewer_metadata.json", {
        "reviewer_id": reviewer, "prior_exposure": "", "background": "",
        "practice_completed_at": "", "started_at": "", "completed_at": "",
        "independent_rating_confirmed": None,
    })
    mapping, index = [], [f"<h1>{reviewer.replace('_', ' ').title()}</h1>",
        '<p><a href="instructions.html">Read the instructions first</a>. '
        'Enter scores in the CSV files; this browser page does not collect responses.</p>']
    for phase in ("practice", "main"):
        ordered = list(items[phase])
        random.Random(int(canonical([subset_hash, reviewer, phase, "order"]), 16)).shuffle(ordered)
        blind_ids = ["review_" + canonical([subset_hash, reviewer, phase, item["item_key"]])[:16]
                     for item in ordered]
        rating_rows = []
        index.append(f"<h2>{phase.title()} — {len(ordered)} outputs</h2><ol>")
        for position, (item, blind) in enumerate(zip(ordered, blind_ids), 1):
            bundle_root, manifest, entries = archives[phase]
            entry, original_rubric = entries[item["item_key"]]
            row = item["row"]
            rubric = original_rubric.replace(entry["blind_id"], blind)
            assets, hashes = {}, {}
            for key, label in (("source_file", "Source"), ("edited_file", "Edited"),
                               ("ground_truth_file", "Reference (instruction only)")):
                if key not in entry:
                    continue
                old = checked_path(bundle_root, entry[key])
                new_name = blind + "_" + key.replace("_file", "") + old.suffix
                target = destination / "images" / new_name
                shutil.copyfile(old, target)
                if digest(target) != manifest["files"][entry[key]]:
                    raise ValueError("Image changed during packet creation")
                rubric = rubric.replace(Path(entry[key]).name.replace(entry["blind_id"], blind), new_name)
                assets[label] = new_name
                hashes[new_name] = digest(target)
            if any(secret in rubric for secret in (row["example_uid"], item["item_key"], entry["blind_id"])):
                raise ValueError("An identifying label remains in the public rubric")
            figs = ''.join(f'<figure><figcaption>{label}</figcaption><a href="../images/{name}">'
                           f'<img src="../images/{name}" alt="{label}"></a></figure>' for label, name in assets.items())
            nav = '<a href="../index.html">Index</a>'
            if position > 1:
                nav += f' · <a href="{blind_ids[position-2]}.html">Previous</a>'
            if position < len(blind_ids):
                nav += f' · <a href="{blind_ids[position]}.html">Next</a>'
            parts = re.split(r'(?m)^### ', rubric)
            rendered = '<pre>' + html.escape(parts[0]) + '</pre>'
            for part in parts[1:]:
                title, text = part.split('\n', 1)
                rendered += f'<details open><summary>{html.escape(title)}</summary><pre>{html.escape(text)}</pre></details>'
            body = (f'<nav>{nav}</nav><h1>{phase.title()} item {position}: {blind}</h1>'
                    f'<p><strong>Original instruction:</strong> {html.escape(row["instruction"])}</p>'
                    '<p>Click images for full resolution. Image A/B in the rubric denotes source/edited, not generation conditions.</p>'
                    f'<div class="images">{figs}</div><p class="notice">Knowledge score: '
                    + ('required; if instruction score ≤2, knowledge score must be 1.' if item["knowledge_required"]
                       else 'not applicable; leave its CSV field blank.')
                    + f' Enter ratings in {"practice_ratings.csv" if phase == "practice" else "ratings.csv"}, row {position}.</p>'
                    + '<p>The following archived scoring rubric is unchanged except anonymous attachment names. '
                      'Use the CSV instead of its example JSON output formats.</p>' + rendered + f'<nav>{nav}</nav>')
            (destination / "cases" / f"{blind}.html").write_text(document(blind, body))
            index.append(f'<li><a href="cases/{blind}.html">{blind}</a></li>')
            rating_rows.append({"reviewer_id": reviewer, "presentation_order": position,
                                "blind_id": blind, "knowledge_required": str(item["knowledge_required"]).lower()})
            mapping.append({"reviewer_id": reviewer, "phase": phase, "presentation_order": position,
                            "blind_id": blind, "item_key": item["item_key"],
                            "example_uid": row["example_uid"], "condition": row["condition"],
                            "base_seed": row["base_seed"], "knowledge_required": item["knowledge_required"],
                            "official_family": item["official_family"], "output_sha256": row["output_sha256"],
                            "source_sha256": row["source_sha256"], "source_bundle_manifest_hash": manifest["manifest_hash"],
                            "rubric_original_sha256": hashlib.sha256(original_rubric.encode()).hexdigest(),
                            "rubric_public_sha256": hashlib.sha256(rubric.encode()).hexdigest(), "images": hashes})
        index.append("</ol>")
        csv_name = "practice_ratings.csv" if phase == "practice" else "ratings.csv"
        with (destination / csv_name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rating_rows)
    (destination / "index.html").write_text(document(reviewer, ''.join(index)))
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/rq3/evidence/chatgpt_main")
    parser.add_argument("--practice-bundle", type=Path, default=ROOT / "outputs/rq3/evidence/calibration")
    parser.add_argument("--subset", type=Path, default=ROOT / "data/rq3_human_validation_subset.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rq3/human_validation")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output directory already exists; preserve ratings and choose a fresh --output-dir")
    manifest, entries = load_archive(args.bundle)
    calibration, practice_entries = load_archive(args.practice_bundle)
    subset = select_subset(manifest)
    if args.subset.exists() and json.loads(args.subset.read_text()) != subset:
        raise ValueError("Existing human subset differs; refusing to change selected examples")
    chosen = set(subset["example_uids"])
    main_items = [item for item in manifest["items"]
                  if item["row"]["example_uid"] in chosen and item["row"]["condition"] in "ABCD"]
    expected = {(uid, cond) for uid in chosen for cond in "ABCD"}
    if len(main_items) != 120 or {(i["row"]["example_uid"], i["row"]["condition"]) for i in main_items} != expected:
        raise ValueError("Expected exactly 120 matched quality outputs")
    practice_items = [item for item in calibration["items"] if item["row"]["condition"] == "A"]
    if len(practice_items) != 4 or chosen & {item["row"]["example_uid"] for item in practice_items}:
        raise ValueError("Expected four disjoint calibration practice examples")
    if not {item["row"]["example_uid"] for item in practice_items} <= set(manifest["generation_snapshot"]["splits"]["k_dev"]):
        raise ValueError("Practice examples must belong to the development split")
    archives = {"main": (args.bundle, manifest, entries),
                "practice": (args.practice_bundle, calibration, practice_entries)}
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".human_validation_", dir=args.output_dir.parent) as temporary:
        staging = Path(temporary) / "packet"
        staging.mkdir()
        (staging / "private").mkdir(mode=0o700)
        mappings = []
        for reviewer in ("reviewer_1", "reviewer_2"):
            mappings.extend(packet(staging / reviewer, reviewer,
                                   {"main": main_items, "practice": practice_items}, archives, canonical(subset)))
        dump(staging / "private/mapping.json", mappings)
        dump(staging / "private/subset.json", subset)
        inventory = {str(p.relative_to(staging)): digest(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        dump(staging / "private/export_manifest.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "subset_hash": canonical(subset),
            "generator_sha256": digest(__file__), "files": inventory,
            "human_ratings_status": "not_started; blank templates only",
            "main_examples": 30, "main_outputs_per_reviewer": 120, "practice_outputs_per_reviewer": 4,
            "knowledge_required_main_outputs": sum(item["knowledge_required"] for item in main_items),
            "viewpoint_main_outputs": sum(item["official_family"] == "view" for item in main_items),
            "selection_timing": subset["selection_timing"],
        })
        (staging / "README.md").write_text(
            "# Coordinator only\n\nSee RQ3_human_validation.md in the repository.\n"
            "Send ONLY reviewer_1/ to reviewer 1 and ONLY reviewer_2/ to reviewer 2.\n"
            "Do not share private/, the subset list, original judge responses or results.\n"
            "Both reviewers score all 120 main outputs; practice outputs are excluded.\n")
        staging.rename(args.output_dir)
    args.subset.parent.mkdir(parents=True, exist_ok=True)
    if not args.subset.exists():
        dump(args.subset, subset)
    print(json.dumps({"subset": str(args.subset), "output_dir": str(args.output_dir),
                      "examples": 30, "main_outputs_per_reviewer": 120,
                      "practice_outputs_per_reviewer": 4,
                      "knowledge_required_outputs": sum(item["knowledge_required"] for item in main_items),
                      "status": "ready_for_independent_human_review"}, indent=2))


if __name__ == "__main__":
    main()

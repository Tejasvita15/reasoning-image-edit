"""Build the offline RQ3 output browser using only the Python standard library.

Run from any working directory: python scripts/build_rq3_gallery.py
Images and immutable experiment records remain in the existing evidence bundle.
The generated page contains relative links and never imports model dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "A": "Thinking off · RegionE off",
    "B": "Thinking on · RegionE off",
    "C": "Thinking off · RegionE on",
    "D": "Thinking on · RegionE on",
    "E": "Thinking on · RegionE without RIKV injection",
    "F": "Thinking on · RegionE without AVD reuse",
}
DIMENSIONS = ("consistency_score", "instruction_score", "quality_score", "knowledge_score")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def checked_path(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected a relative evidence path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"Missing or unsafe evidence path: {relative}")
    return path


def load_evidence(bundle):
    manifest = json.loads((bundle / "private/manifest.json").read_text(encoding="utf-8"))
    require(manifest["kind"] == "main" and manifest["schema_version"] == 1, "Expected the main RQ3 bundle")
    require(canonical({k: v for k, v in manifest.items() if k != "manifest_hash"}) == manifest["manifest_hash"],
            "Bundle manifest checksum mismatch")
    snapshot = manifest["generation_snapshot"]
    require(canonical({k: snapshot[k] for k in ("config", "splits", "subset", "judge")}) ==
            manifest["generation_protocol_hash"] == snapshot["protocol_hash"], "Generation protocol mismatch")
    require(canonical(manifest["evaluation_protocol"]) == manifest["evaluation_protocol_hash"],
            "Evaluation protocol mismatch")
    selection = [{"item_key": item["item_key"], **{key: item["row"][key] for key in
                 ("example_uid", "condition", "base_seed", "output_sha256")}} for item in manifest["items"]]
    require(canonical(selection) == manifest["selection_hash"], "Selected outputs checksum mismatch")
    items = {item["item_key"]: item for item in manifest["items"]}
    require(len(items) == len(manifest["items"]) == 200, "Expected 200 distinct quality outputs")
    images, aliases, batch_ids = {}, set(), set()
    for batch in manifest["batches"]:
        require(batch["pass"] == 1 and batch["batch_id"] not in batch_ids, "Unexpected or duplicate judging batch")
        batch_ids.add(batch["batch_id"])
        for entry in batch["items"]:
            key = entry["item_key"]
            require(key in items and key not in images and entry["blind_id"] not in aliases,
                    "Unknown or duplicate image identity")
            aliases.add(entry["blind_id"])
            row = items[key]["row"]
            resolved = {}
            for field, digest_field in (("source_file", "source_sha256"), ("edited_file", "output_sha256")):
                path = checked_path(bundle, entry[field])
                require(path.is_relative_to((bundle / "upload").resolve()), "Image is outside the upload directory")
                require(digest(path) == manifest["files"].get(entry[field]) == row[digest_field],
                        f"Image checksum mismatch for {key}: {field}")
                resolved[field] = path
            if items[key]["official_family"] == "view":
                path = checked_path(bundle, entry.get("ground_truth_file"))
                require(digest(path) == manifest["files"].get(entry["ground_truth_file"]) ==
                        items[key]["ground_truth_sha256"], f"Reference image mismatch: {key}")
                resolved["ground_truth_file"] = path
            images[key] = {**resolved, "blind_id": entry["blind_id"], "batch_id": batch["batch_id"]}
    require(set(images) == set(items), "Judging batches do not cover the 200 outputs")

    scores_path = bundle / "scores.jsonl"
    scores, rows = {}, []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        score = json.loads(line)
        key = score["item_key"]
        require(key in items and key not in scores, "Unknown or duplicate scored output")
        item, entry = items[key], images[key]
        row = item["row"]
        require(score["status"] == "ok" and score["pass_number"] == 1, "Incomplete or unexpected judging pass")
        for field in ("example_uid", "condition", "base_seed", "source_sha256", "output_sha256"):
            require(score[field] == row[field], f"Score/generation identity differs: {key}, {field}")
        require(row["repeat"] == 0 and row["base_seed"] == 42, "Expected quality repeat zero, seed 42")
        for field in ("blind_id", "batch_id"):
            require(score[field] == entry[field], f"Score/batch identity differs: {key}")
        for field in ("generation_protocol_hash", "evaluation_protocol_hash", "selection_hash"):
            require(score[field] == manifest[field], f"Score protocol differs: {key}, {field}")
        require(score["bundle_manifest_hash"] == manifest["manifest_hash"], "Score bundle differs")
        require(score["knowledge_required"] == item["knowledge_required"], "Knowledge applicability differs")
        applicable = DIMENSIONS if item["knowledge_required"] else DIMENSIONS[:3]
        require(all(type(score[field]) is int and 1 <= score[field] <= 5 for field in applicable),
                f"Invalid component score: {key}")
        if not item["knowledge_required"]:
            require(score["knowledge_score"] is None, "Inapplicable knowledge score must be null")
        calculated = sum(25 * (score[field] - 1) for field in applicable) / len(applicable)
        require(math.isclose(calculated, score["Q"], abs_tol=1e-9), f"Aggregate Q mismatch: {key}")
        scores[key] = score
        rows.append({"row": row, "score": score, "images": entry})
    require(set(scores) == set(items), "Scored outputs do not cover the complete bundle")
    expected = {(uid, condition) for uid in snapshot["splits"]["k_small"] for condition in "ABCD"}
    expected |= {(uid, condition) for uid in snapshot["subset"]["example_uids"] for condition in "EF"}
    observed = {(record["row"]["example_uid"], record["row"]["condition"]) for record in rows}
    require(len(expected) == len(observed) == len(rows) == 200 and observed == expected,
            "Condition/example coverage differs from the frozen 40/20 design")
    return manifest, rows, scores_path


def load_humans(directory, manifest, records, scores_path):
    if not directory.exists():
        return {}
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    inventory = json.loads((directory / "artifact_inventory.json").read_text(encoding="utf-8"))
    for name in ("report.json", "tables/joined_scores.csv"):
        require(digest(directory / name) == inventory[name], f"Human analysis artifact checksum mismatch: {name}")
    provenance = report["provenance"]
    require(provenance["judge_scores_sha256"] == digest(scores_path), "Human analysis used different judge scores")
    for field in ("generation_protocol_hash", "evaluation_protocol_hash"):
        require(provenance[field] == manifest[field], "Human analysis protocol differs")
    original = {(item["row"]["example_uid"], item["row"]["condition"]): item for item in records}
    humans = {}
    with (directory / "tables/joined_scores.csv").open(encoding="utf-8", newline="") as handle:
        for human in csv.DictReader(handle):
            key = (human["example_uid"], human["condition"])
            require(key in original and key not in humans and key[1] in "ABCD", "Unknown or duplicate human output")
            record = original[key]
            require(human["instruction"] == record["row"]["instruction"], "Human instruction differs")
            require(math.isclose(float(human["chatgpt_Q"]), record["score"]["Q"], abs_tol=1e-9),
                    "Human table has a different judge score")
            values = [float(human[f"reviewer_{reviewer}_Q"]) for reviewer in (1, 2)]
            require(all(math.isfinite(value) and 0 <= value <= 100 for value in values), "Invalid human Q")
            require(math.isclose(sum(values) / 2, float(human["human_mean_Q"]), abs_tol=1e-9), "Human mean Q differs")
            humans[key] = human
    uids = {uid for uid, _ in humans}
    require(len(uids) == 30 and set(humans) == {(uid, condition) for uid in uids for condition in "ABCD"},
            "Expected all A–D outputs for the 30-example human subset")
    return humans


def render(manifest, records, humans, destination, scores_path, human_directory):
    escape = html.escape

    def link(path):
        return escape(quote(Path(os.path.relpath(path, destination.parent)).as_posix(), safe="/"), quote=True)

    def image_card(path, title, extra="", condition=None):
        attributes = f' class="output" data-condition="{condition}"' if condition else ' class="input"'
        return (f"<figure{attributes}><figcaption><strong>{escape(title)}</strong></figcaption>"
                f'<a href="{link(path)}"><img src="{link(path)}" loading="lazy" decoding="async" '
                f'alt="{escape(title, quote=True)}"></a>{extra}</figure>')

    grouped = defaultdict(list)
    for record in records:
        grouped[record["row"]["example_uid"]].append(record)
    examples = []
    for uid, items in sorted(grouped.items()):
        items.sort(key=lambda item: item["row"]["condition"])
        first = items[0]["row"]
        require(all(item["row"]["instruction"] == first["instruction"] and
                    item["row"]["source_sha256"] == first["source_sha256"] for item in items),
                f"Example source/instruction differs across conditions: {uid}")
        source = image_card(items[0]["images"]["source_file"], "Original source")
        if "ground_truth_file" in items[0]["images"]:
            source += image_card(items[0]["images"]["ground_truth_file"], "Viewpoint reference (instruction scoring only)")
        cards = []
        for item in items:
            condition, score = item["row"]["condition"], item["score"]
            extra = f'<p class="scores">ChatGPT Q: <b>{score["Q"]:.2f}</b>'
            human = humans.get((uid, condition))
            if human:
                extra += (f'<br>Human 1: <b>{float(human["reviewer_1_Q"]):.2f}</b> · '
                          f'Human 2: <b>{float(human["reviewer_2_Q"]):.2f}</b>'
                          f'<br>Human mean (secondary): <b>{float(human["human_mean_Q"]):.2f}</b>')
            elif humans:
                extra += '<br><span class="muted">Not included in human validation</span>'
            extra += '</p><details><summary>Score details and rationales</summary>'
            labels = (("ChatGPT", score, ""),)
            if human:
                labels += (("Human 1", human, "reviewer_1_"), ("Human 2", human, "reviewer_2_"))
            for label, values, prefix in labels:
                components = [str(values[prefix + dimension]) if values[prefix + dimension] not in (None, "") else "n/a"
                              for dimension in DIMENSIONS]
                extra += (f'<p><b>{label}</b> — consistency / instruction / visual quality / knowledge: '
                          f'{escape(" / ".join(components))}</p><p>{escape(values[prefix + "rationale"])}</p>')
            extra += "</details>"
            cards.append(image_card(item["images"]["edited_file"], f"{condition} · {CONDITIONS[condition]}", extra, condition))
        search = escape(f'{uid} {first["knowledge_type"]} {first["instruction"]}'.lower(), quote=True)
        has_human = "true" if (uid, "A") in humans else "false"
        examples.append(f'<section class="example" id="{escape(uid)}" data-search="{search}" data-human="{has_human}">'
                        f'<h2>{escape(uid)}</h2><p class="muted">{escape(first["knowledge_type"].capitalize())} knowledge</p>'
                        f'<p class="instruction">{escape(first["instruction"])}</p><div class="sources">{source}</div>'
                        f'<div class="outputs">{"".join(cards)}</div></section>')
    human_nav = f'<a href="{link(human_directory / "index.html")}">Human-validation charts</a> · ' if humans else ""
    human_note = ("Two independent human reviewers scored A–D on a supplementary 30-example subset (120 outputs each). "
                  "Their individual scores are shown; their mean is a secondary summary, not an adjudicated ground truth. "
                  "Human–ChatGPT agreement was weak, so interpretation should compare the rating sources."
                  if humans else "No human-validation scores are included in this build.")
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RQ3 · Complete output gallery</title><style>
:root{color-scheme:light;--ink:#192c37;--muted:#556975;--border:#d6e0e5;--accent:#075e7c}
*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;color:var(--ink);background:#f6f8fa;margin:0}
header,main,footer{max-width:1440px;margin:auto;padding:24px}h1{font-size:2rem;margin:.3em 0}h2{overflow-wrap:anywhere}
a{color:var(--accent)}nav{margin:16px 0}.intro{max-width:1000px}.muted{color:var(--muted)}
.legend,.filters{display:flex;flex-wrap:wrap;gap:12px}.legend span{border:1px solid var(--border);border-radius:6px;padding:6px 10px;background:white}
.filters{align-items:center;border:1px solid var(--border);padding:16px;background:white;border-radius:10px;margin-top:20px}
input[type=search],select{font:inherit;padding:8px;border:1px solid #889ca7;border-radius:5px}input[type=search]{width:min(350px,100%)}
.example{padding:20px;background:white;border:1px solid var(--border);border-radius:12px;margin-bottom:28px}
.example h2{margin:0}.instruction{padding:12px;background:#edf5f8;border-left:4px solid #5598ac;white-space:pre-wrap}
.sources,.outputs{display:grid;gap:16px}.sources{grid-template-columns:repeat(auto-fit,minmax(240px,360px));margin-bottom:22px}
.outputs{grid-template-columns:repeat(3,minmax(0,1fr))}figure{margin:0;border:1px solid var(--border);border-radius:8px;overflow:hidden}
figcaption{padding:10px;background:#edf2f5;min-height:70px}img{display:block;width:100%;height:320px;object-fit:contain;background:#fafafa}
.sources img{height:240px}.scores,details{padding:0 12px}.scores{font-variant-numeric:tabular-nums}details{margin-bottom:12px;font-size:.92rem}
summary{cursor:pointer}details p{white-space:pre-wrap;overflow-wrap:anywhere}[hidden]{display:none!important}code{overflow-wrap:anywhere}
@media(max-width:1000px){.outputs{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:640px){header,main,footer{padding:14px}.example{padding:12px}.outputs{grid-template-columns:1fr}}
</style></head><body><header><h1>RQ3 · Complete output gallery</h1>
<p class="intro"><b>200 edited images · 40 examples · Step1X–RegionE · 28 steps · seed 42.</b>
A–D cover all 40 examples; E/F cover the fixed 20-example component subset. Click any image to inspect its full resolution.
This page works offline after cloning or downloading the branch; no GPU, API key or server is needed.</p>
<nav>""" + f'<a href="{link(ROOT / "RQ3_findings.md")}">Report findings</a> · <a href="{link(ROOT / "outputs/rq3/postprocessing/index.html")}">Experiment charts</a> · ' + human_nav + f'<a href="{link(scores_path)}">Original judge scores</a>' + """</nav>
<p class="intro">Q is the mean of 25 × (score − 1) across applicable rubric dimensions (0–100).
It is a study-specific aggregate, not an official benchmark score or ground truth. ChatGPT scores are from the recorded
ChatGPT interface sessions; model sampling and API snapshot were not controlled.</p><p class="intro">""" + escape(human_note) + """</p>
<div class="legend">""" + "".join(f"<span><b>{key}</b> {escape(value)}</span>" for key, value in CONDITIONS.items()) + """</div>
<div class="filters"><label>Find example <input id="search" type="search" placeholder="ID, instruction or knowledge type"></label>
<label>Condition <select id="condition"><option value="all">All conditions</option>""" + "".join(
        f'<option value="{key}">{key}</option>' for key in CONDITIONS) + """</select></label>
<label><input id="human" type="checkbox"> Human-reviewed examples only</label>
<output id="count" aria-live="polite">40 examples · 200 edited images</output></div>
<noscript><p>All images are shown below. Filters require JavaScript; the images and links work without it.</p></noscript>
</header><main>""" + "".join(examples) + """</main><footer>
<p>Rebuild with <code>python scripts/build_rq3_gallery.py</code>. The builder verifies the frozen condition coverage,
manifest identity, image checksums, judge scores and any included human-analysis table before writing this page.</p>
<details><summary>Experiment identity</summary><p>Generation protocol: <code>""" + escape(manifest["generation_protocol_hash"]) + """</code><br>
Evaluation protocol: <code>""" + escape(manifest["evaluation_protocol_hash"]) + """</code><br>
Bundle manifest: <code>""" + escape(manifest["manifest_hash"]) + """</code></p></details></footer>
<script>
const search=document.querySelector('#search'), condition=document.querySelector('#condition'), human=document.querySelector('#human');
function filter(){let examples=0,images=0;const query=search.value.trim().toLowerCase();
document.querySelectorAll('.example').forEach(section=>{let visible=0;
section.querySelectorAll('.output').forEach(card=>{card.hidden=condition.value!=='all'&&card.dataset.condition!==condition.value;if(!card.hidden)visible++;});
section.hidden=!section.dataset.search.includes(query)||(human.checked&&section.dataset.human!=='true')||visible===0;
if(!section.hidden){examples++;images+=visible;}});
document.querySelector('#count').textContent=`${examples} examples · ${images} edited images`;}
search.addEventListener('input',filter);condition.addEventListener('change',filter);human.addEventListener('change',filter);
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/rq3/evidence/chatgpt_main")
    parser.add_argument("--human-analysis", type=Path, default=ROOT / "outputs/rq3/human_validation/analysis")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rq3/index.html")
    args = parser.parse_args()
    manifest, records, scores_path = load_evidence(args.bundle.resolve())
    humans = load_humans(args.human_analysis.resolve(), manifest, records, scores_path)
    destination = args.output.resolve()
    page = render(manifest, records, humans, destination, scores_path, args.human_analysis.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    print(json.dumps({"status": "complete", "examples": 40, "quality_outputs": len(records),
                      "condition_counts": dict(sorted(Counter(item["row"]["condition"] for item in records).items())),
                      "human_scored_outputs": len(humans), "html_bytes": destination.stat().st_size}))


if __name__ == "__main__":
    main()

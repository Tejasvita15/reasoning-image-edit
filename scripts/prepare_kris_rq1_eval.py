#!/usr/bin/env python3
"""Stage a selected KRIS-Bench split for the official GPT-4o evaluation scripts.

Source of truth: Hugging Face Liang0223/KRIS_Bench, NOT the different GitHub
annotation.json release. Keeps original experiment files unchanged.

Run from the reasoning-image-edit project root.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image

VARIANTS = ("base", "thinking", "correction")
COMMON = {
    "count_change", "color_change", "anomaly_correction", "position_movement",
    "size_adjustment", "part_completion", "multi-instruction_execution",
}
KNOWLEDGE = {
    "abstract_reasoning", "mathematics", "practical_knowledge", "medicine",
    "rule-based_reasoning", "biology", "geography", "chemistry",
    "humanities", "physics",
}
VIEW = {"viewpoint_change"}
SUPPORTED = COMMON | KNOWLEDGE | VIEW
SCRIPTS = ("metrics_common.py", "metrics_knowledge.py", "metrics_view_change.py")


def read_rows(path):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row["example_uid"]
            if uid in result:
                raise ValueError(f"Duplicate example_uid in normalized file {path}: {uid}")
            if row.get("status") != "success":
                raise ValueError(f"Non-success row in {path}: {uid}")
            result[uid] = row
    return result


def save_jpeg(img, path):
    if img is None:
        raise ValueError(f"Missing image: {path}")
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected Pillow image for {path}, got {type(img)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, format="JPEG", quality=95, subsampling=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--official-repo", type=Path, default=Path("third_party/Kris_Bench"))
    p.add_argument("--workspace", type=Path, default=Path("analysis/rq1/kris_judge/pilot"))
    p.add_argument("--subset", choices=("pilot", "full", "ktest"), default="pilot")
    p.add_argument("--dataset-id", default="Liang0223/KRIS_Bench")
    args = p.parse_args()
    root = args.project_root.resolve()
    repo = (root / args.official_repo).resolve()
    work = (root / args.workspace).resolve()
    if work == root or work == repo:
        raise ValueError("Workspace must be separate from the project/repository root")
    for script in SCRIPTS:
        if not (repo / script).is_file():
            raise FileNotFoundError(repo / script)
    if not (repo / "utils" / "prompts.py").is_file():
        raise FileNotFoundError(repo / "utils" / "prompts.py")

    rows = {}
    expected = 300 if args.subset == "ktest" else 90

    for variant in VARIANTS:
        if args.subset == "ktest":
            src = root / f"logs/rq1_{variant}_k_test_seed42.jsonl"
        else:
            src = root / f"analysis/rq1/k_dev/inputs/rq1_{variant}_k_dev_seed42.jsonl"

        rows[variant] = read_rows(src)

    common_uids = set(rows["base"])
    if len(common_uids) != expected or any(
        set(rows[v]) != common_uids for v in VARIANTS
    ):
        raise ValueError(
            f"Expected {expected} matching examples across all three variants"
        )
    for uid in common_uids:
        base = rows["base"][uid]
        if base["category"] not in SUPPORTED:
            raise ValueError(f"Unsupported category {base['category']} for {uid}")
        for v in VARIANTS[1:]:
            other = rows[v][uid]
            if base["instruction"] != other["instruction"]:
                raise ValueError(f"Variant instruction mismatch for {uid}")

    uids = sorted(common_uids)
    if args.subset == "pilot":
        selected = []
        for family in (COMMON, KNOWLEDGE, VIEW):
            matches = [uid for uid in uids if rows["base"][uid]["category"] in family]
            if not matches:
                raise ValueError(f"No example in required family {family}")
            selected.append(matches[0])
        uids = selected

    work.mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        source = (repo / script).read_text(encoding="utf-8")
        # Patch only the *separate evaluation copy* to honor RQ1 judge temperature=0.
        # Upstream Step1X, KRIS-Bench clone, and generation outputs remain untouched.
        rx = re.compile(r"\bmax_tokens\s*=\s*1000\b")
        source, count = rx.subn("max_tokens=1000, temperature=0", source)
        if count != 1:
            raise RuntimeError(f"Expected exactly one API call in {script}; found {count}. Inspect manually.")
        (work / script).write_text(source, encoding="utf-8")

    # Official evaluation scripts import `utils.prompts` from their own directory.
    target = work / "utils"
    if target.is_symlink():
        if target.resolve() != (repo / "utils").resolve():
            raise ValueError(f"Unexpected utils symlink: {target}")
    elif target.exists():
        raise ValueError(f"Expected a utils symlink, got existing directory: {target}")
    else:
        target.symlink_to(repo / "utils", target_is_directory=True)

    print("Loading Hugging Face reference dataset...", flush=True)
    ds = load_dataset(args.dataset_id, split="train")
    references = {}
    for i, r in enumerate(ds.select_columns(["category", "id"])):
        uid = f'{r["category"]}__{r["id"]}'
        if uid in common_uids:
            if uid in references:
                raise ValueError(f"Duplicate Hugging Face reference: {uid}")
            references[uid] = i
    missing = sorted(set(uids) - references.keys())
    if missing:
        raise ValueError(f"Missing reference rows: {missing}")

    manifest = []
    annotation_by_cat = {}
    count = Counter()
    for uid in uids:
        base = rows["base"][uid]
        cat, exid = base["category"], str(base["example_id"])
        ref = ds[references[uid]]
        if str(ref["id"]) != exid or ref["category"] != cat:
            raise AssertionError(f"Reference mismatch for {uid}")
        if ref["instruction"].strip() != base["instruction"].strip():
            raise ValueError(f"Reference instruction mismatch for {uid}")
        original_file = f"original_{exid}.jpg"
        ground_file = f"ground_truth_{exid}.jpg"
        save_jpeg(ref["image"], work / "KRIS_Bench" / cat / original_file)
        gt_available = ref.get("gt_image") is not None
        if cat in VIEW:
            if not gt_available:
                raise ValueError(f"No viewpoint reference image for {uid}")
            save_jpeg(ref["gt_image"], work / "KRIS_Bench" / cat / ground_file)
        ann = {
            "ori_img": original_file,
            "gt_img": ground_file if cat in VIEW else "",
            "ins_en": ref["instruction"],
            "explain_en": ref.get("explanation") or "",
        }
        annotation_by_cat.setdefault(cat, {})[exid] = ann
        for variant in VARIANTS:
            source = root / rows[variant][uid]["output_path"]
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as img:
                save_jpeg(img, work / "results" / variant / cat / f"{exid}.jpg")
        count[cat] += 1
        manifest.append({"example_uid": uid, "category": cat, "id": exid, "variants": list(VARIANTS)})
    for cat, annotations in annotation_by_cat.items():
        dest = work / "KRIS_Bench" / cat / "annotation.json"
        dest.write_text(json.dumps(annotations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (work / "staging_manifest.json").write_text(
        json.dumps({"subset": args.subset, "dataset": args.dataset_id,
                    "n_examples": len(uids), "n_images": len(uids) * 3,
                    "categories": dict(sorted(count.items())), "entries": manifest},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Evaluation workspace:", work)
    print("Examples:", len(uids), "Generated JPEGs:", len(uids) * 3)
    print("Categories:", dict(sorted(count.items())))
    print("Official scripts staged with temperature=0; dataset and experiment sources untouched.")


if __name__ == "__main__":
    main()

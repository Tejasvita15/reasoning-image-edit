# src/export_rq1_annotations.py

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ANNOTATION_COLUMNS = [
    "wrong_knowledge",
    "omitted_constraints",
    "correct_plan_wrong_image",
    "unnecessary_changes",
    "mistaken_critique",
]


def load_jsonl(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for path_str in paths:
        path = Path(path_str)

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_no}"
                    ) from exc

                if row.get("status", "success") != "success":
                    continue

                rows.append(row)

    return rows


def safe_json(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def infer_condition(row: Dict[str, Any]) -> str:
    condition = row.get("condition")

    if condition:
        return str(condition)

    thinking = bool(
        row.get("thinking_enabled")
    )
    reflection = bool(
        row.get("reflection_enabled")
    )

    if thinking and reflection:
        return "correction"

    if thinking:
        return "thinking"

    return "base"


def select_one_row_per_example(
    rows: List[Dict[str, Any]],
    *,
    preferred_seed: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    """
    Keep one representative row per example.

    For RQ1.3, annotations should be performed on a fixed output.
    Prefer the requested seed when present; otherwise take the
    first row encountered.
    """

    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        uid = row.get("example_uid")

        if not uid:
            continue

        grouped.setdefault(
            str(uid),
            [],
        ).append(row)

    selected: Dict[str, Dict[str, Any]] = {}

    for uid, items in grouped.items():
        choice = None

        if preferred_seed is not None:
            for item in items:
                if item.get("base_seed") == preferred_seed:
                    choice = item
                    break

        if choice is None:
            choice = items[0]

        selected[uid] = choice

    return selected


def build_annotation_row(
    *,
    thinking_row: Dict[str, Any],
    correction_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build one reviewer-facing record.

    The thinking-only output is the main reasoning artifact.
    Correction/reflection fields are included when available.
    """

    row: Dict[str, Any] = {
        "example_uid":
            thinking_row.get("example_uid"),

        "example_id":
            thinking_row.get("example_id"),

        "category":
            thinking_row.get("category"),

        "reasoning_dimension":
            thinking_row.get("reasoning_dimension"),

        "knowledge_type":
            thinking_row.get("knowledge_type"),

        "instruction":
            thinking_row.get("instruction"),

        "base_seed":
            thinking_row.get("base_seed"),

        "thinking_output_path":
            thinking_row.get("output_path"),

        "thinking_candidate_paths":
            safe_json(
                thinking_row.get("candidate_paths")
            ),

        "reformat_prompt":
            thinking_row.get("reformat_prompt") or "",

        "think_info":
            safe_json(
                thinking_row.get("think_info")
            ),

        "correction_output_path":
            (
                correction_row.get("output_path")
                if correction_row else ""
            ),

        "correction_candidate_paths":
            safe_json(
                correction_row.get("candidate_paths")
                if correction_row else None
            ),

        "reflection_info":
            safe_json(
                correction_row.get("reflection_info")
                if correction_row else None
            ),

        "selected_candidate_index":
            (
                correction_row.get(
                    "selected_candidate_index"
                )
                if correction_row else ""
            ),

        "attempt_count":
            (
                correction_row.get("attempt_count")
                if correction_row else ""
            ),

        "retry_triggered":
            (
                correction_row.get("retry_triggered")
                if correction_row else ""
            ),

        # Reviewer labels: use 0/1.
        "wrong_knowledge": "",
        "omitted_constraints": "",
        "correct_plan_wrong_image": "",
        "unnecessary_changes": "",
        "mistaken_critique": "",

        # Free-text notes.
        "reviewer_notes": "",
    }

    return row


def write_csv(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "No annotation rows to write."
        )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_codebook(
    path: Path,
) -> None:
    """
    Save a short RQ1.3 annotation codebook.
    """

    text = """# RQ1.3 Annotation Codebook

Use multiple labels when more than one failure applies.

## wrong_knowledge
Mark 1 when the reasoning/rewrite uses incorrect factual, conceptual,
or procedural knowledge needed to perform the requested edit.

## omitted_constraints
Mark 1 when the reasoning/rewrite misses an explicit or necessary
constraint from the editing instruction.

## correct_plan_wrong_image
Mark 1 when the reasoning/rewrite is substantially correct, but the
generated image fails to execute that plan correctly.

## unnecessary_changes
Mark 1 when the generated image changes content that should have been
preserved and those changes are not required by the instruction.

## mistaken_critique
Mark 1 when correction/reflection incorrectly diagnoses a successful
or acceptable output as wrong, or critiques the wrong aspect and this
leads to an unnecessary or harmful correction.

## Reviewer procedure
- Judge the plan/rewrite against the KRIS knowledge hints when available.
- Judge the image against the original instruction and the intended plan.
- Multiple labels are allowed.
- Use 0 for absent and 1 for present.
- Use reviewer_notes for ambiguous cases.
- Pilot this codebook on 10 examples before full annotation.
"""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export RQ1.3 reviewer sheets from "
            "Thinking and Correction JSONL logs."
        )
    )

    parser.add_argument(
        "--thinking-log",
        nargs="+",
        required=True,
        help=(
            "Thinking-only K-small JSONL log(s)."
        ),
    )

    parser.add_argument(
        "--correction-log",
        nargs="*",
        default=[],
        help=(
            "Optional Thinking+Correction K-small "
            "JSONL log(s)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Representative seed to export for "
            "human annotation. Default: 42."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of examples. "
            "Use --limit 10 for the codebook pilot."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="annotations/rq1",
    )

    args = parser.parse_args()

    thinking_rows = load_jsonl(
        args.thinking_log
    )

    correction_rows = (
        load_jsonl(args.correction_log)
        if args.correction_log
        else []
    )

    thinking_by_example = (
        select_one_row_per_example(
            thinking_rows,
            preferred_seed=args.seed,
        )
    )

    correction_by_example = (
        select_one_row_per_example(
            correction_rows,
            preferred_seed=args.seed,
        )
        if correction_rows
        else {}
    )

    example_uids = sorted(
        thinking_by_example.keys()
    )

    if args.limit is not None:
        example_uids = (
            example_uids[: args.limit]
        )

    annotation_rows = []

    for uid in example_uids:
        thinking_row = (
            thinking_by_example[uid]
        )

        correction_row = (
            correction_by_example.get(uid)
        )

        annotation_rows.append(
            build_annotation_row(
                thinking_row=thinking_row,
                correction_row=correction_row,
            )
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reviewer1_path = (
        output_dir
        / "rq1_reviewer1.csv"
    )

    reviewer2_path = (
        output_dir
        / "rq1_reviewer2.csv"
    )

    write_csv(
        annotation_rows,
        reviewer1_path,
    )

    write_csv(
        annotation_rows,
        reviewer2_path,
    )

    codebook_path = (
        output_dir
        / "rq1_annotation_codebook.md"
    )

    write_codebook(
        codebook_path
    )

    manifest = {
        "preferred_seed":
            args.seed,

        "num_examples":
            len(annotation_rows),

        "thinking_rows_loaded":
            len(thinking_rows),

        "correction_rows_loaded":
            len(correction_rows),

        "reviewer1_csv":
            str(reviewer1_path),

        "reviewer2_csv":
            str(reviewer2_path),

        "codebook":
            str(codebook_path),

        "annotation_labels":
            ANNOTATION_COLUMNS,
    }

    manifest_path = (
        output_dir
        / "rq1_annotation_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("RQ1.3 ANNOTATION EXPORT")
    print("=" * 70)
    print(
        "Examples:",
        len(annotation_rows),
    )
    print(
        "Representative seed:",
        args.seed,
    )
    print(
        "Reviewer 1:",
        reviewer1_path,
    )
    print(
        "Reviewer 2:",
        reviewer2_path,
    )
    print(
        "Codebook:",
        codebook_path,
    )
    print(
        "Manifest:",
        manifest_path,
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

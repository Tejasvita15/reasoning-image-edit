# src/analyze_rq1_annotations.py

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None


LABELS = [
    "wrong_knowledge",
    "omitted_constraints",
    "correct_plan_wrong_image",
    "unnecessary_changes",
    "mistaken_critique",
]


def read_csv(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: Dict[str, Dict[str, str]] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if "example_uid" not in (reader.fieldnames or []):
            raise ValueError(
                f"{path} is missing required column: example_uid"
            )

        for line_no, row in enumerate(reader, start=2):
            uid = (row.get("example_uid") or "").strip()

            if not uid:
                raise ValueError(
                    f"{path}:{line_no} has empty example_uid"
                )

            if uid in rows:
                raise ValueError(
                    f"{path}:{line_no} duplicate example_uid: {uid}"
                )

            rows[uid] = row

    return rows


def parse_binary(
    value: str,
    *,
    path: Path,
    example_uid: str,
    label: str,
) -> int:
    text = (value or "").strip().lower()

    if text in {"1", "true", "yes", "y"}:
        return 1

    if text in {"0", "false", "no", "n"}:
        return 0

    raise ValueError(
        f"{path}: example={example_uid}, label={label} "
        f"must be binary 0/1, got {value!r}"
    )


def manual_kappa(
    a: List[int],
    b: List[int],
) -> float:
    if len(a) != len(b):
        raise ValueError("Kappa inputs must have equal length.")

    n = len(a)

    if n == 0:
        raise ValueError("Kappa requires at least one item.")

    observed = sum(x == y for x, y in zip(a, b)) / n

    p_a_1 = sum(a) / n
    p_a_0 = 1.0 - p_a_1

    p_b_1 = sum(b) / n
    p_b_0 = 1.0 - p_b_1

    expected = (
        p_a_1 * p_b_1
        + p_a_0 * p_b_0
    )

    if expected == 1.0:
        # Degenerate case: both reviewers assigned the same
        # single class to every item.
        return 1.0 if observed == 1.0 else 0.0

    return (observed - expected) / (1.0 - expected)


def compute_kappa(
    a: List[int],
    b: List[int],
) -> float:
    if cohen_kappa_score is not None:
        return float(
            cohen_kappa_score(a, b)
        )

    return float(
        manual_kappa(a, b)
    )


def analyze_label(
    *,
    label: str,
    uids: List[str],
    reviewer1: Dict[str, Dict[str, str]],
    reviewer2: Dict[str, Dict[str, str]],
    reviewer1_path: Path,
    reviewer2_path: Path,
) -> Tuple[Dict, List[Dict]]:
    values1: List[int] = []
    values2: List[int] = []
    disagreements: List[Dict] = []

    for uid in uids:
        value1 = parse_binary(
            reviewer1[uid].get(label, ""),
            path=reviewer1_path,
            example_uid=uid,
            label=label,
        )

        value2 = parse_binary(
            reviewer2[uid].get(label, ""),
            path=reviewer2_path,
            example_uid=uid,
            label=label,
        )

        values1.append(value1)
        values2.append(value2)

        if value1 != value2:
            disagreements.append(
                {
                    "example_uid": uid,
                    "label": label,
                    "reviewer1": value1,
                    "reviewer2": value2,
                    "instruction": (
                        reviewer1[uid].get("instruction")
                        or reviewer2[uid].get("instruction")
                        or ""
                    ),
                    "knowledge_type": (
                        reviewer1[uid].get("knowledge_type")
                        or reviewer2[uid].get("knowledge_type")
                        or ""
                    ),
                    "category": (
                        reviewer1[uid].get("category")
                        or reviewer2[uid].get("category")
                        or ""
                    ),
                    "reviewer1_notes":
                        reviewer1[uid].get("reviewer_notes", ""),
                    "reviewer2_notes":
                        reviewer2[uid].get("reviewer_notes", ""),
                }
            )

    n = len(values1)

    raw_agreement = (
        sum(x == y for x, y in zip(values1, values2))
        / n
    )

    both_positive = sum(
        x == 1 and y == 1
        for x, y in zip(values1, values2)
    )

    both_negative = sum(
        x == 0 and y == 0
        for x, y in zip(values1, values2)
    )

    positive_r1 = sum(values1)
    positive_r2 = sum(values2)

    result = {
        "label": label,
        "n": n,
        "cohen_kappa": compute_kappa(
            values1,
            values2,
        ),
        "raw_agreement": float(
            raw_agreement
        ),
        "reviewer1_positive_count":
            int(positive_r1),
        "reviewer2_positive_count":
            int(positive_r2),
        "reviewer1_positive_fraction":
            float(positive_r1 / n),
        "reviewer2_positive_fraction":
            float(positive_r2 / n),
        "both_positive_count":
            int(both_positive),
        "both_negative_count":
            int(both_negative),
        "disagreement_count":
            len(disagreements),
        "disagreement_fraction":
            float(
                len(disagreements) / n
            ),
    }

    return result, disagreements


def write_disagreements(
    rows: List[Dict],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "example_uid",
        "label",
        "reviewer1",
        "reviewer2",
        "knowledge_type",
        "category",
        "instruction",
        "reviewer1_notes",
        "reviewer2_notes",
        "adjudicated_value",
        "adjudication_notes",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            output = {
                key: row.get(key, "")
                for key in fieldnames
            }
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute RQ1.3 inter-reviewer agreement "
            "for the five failure labels."
        )
    )

    parser.add_argument(
        "--reviewer1",
        default="annotations/rq1/rq1_reviewer1.csv",
    )

    parser.add_argument(
        "--reviewer2",
        default="annotations/rq1/rq1_reviewer2.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="analysis/rq1_annotations",
    )

    args = parser.parse_args()

    reviewer1_path = Path(args.reviewer1)
    reviewer2_path = Path(args.reviewer2)

    reviewer1 = read_csv(
        reviewer1_path
    )
    reviewer2 = read_csv(
        reviewer2_path
    )

    ids1 = set(
        reviewer1.keys()
    )
    ids2 = set(
        reviewer2.keys()
    )

    missing_from_reviewer1 = sorted(
        ids2 - ids1
    )
    missing_from_reviewer2 = sorted(
        ids1 - ids2
    )

    if (
        missing_from_reviewer1
        or missing_from_reviewer2
    ):
        raise ValueError(
            "Reviewer files do not contain the same examples.\n"
            f"Missing from reviewer1: {missing_from_reviewer1}\n"
            f"Missing from reviewer2: {missing_from_reviewer2}"
        )

    uids = sorted(
        ids1
    )

    if not uids:
        raise ValueError(
            "No annotation rows found."
        )

    label_results = {}
    all_disagreements: List[Dict] = []

    for label in LABELS:
        result, disagreements = (
            analyze_label(
                label=label,
                uids=uids,
                reviewer1=reviewer1,
                reviewer2=reviewer2,
                reviewer1_path=
                    reviewer1_path,
                reviewer2_path=
                    reviewer2_path,
            )
        )

        label_results[label] = result
        all_disagreements.extend(
            disagreements
        )

    mean_kappa = sum(
        result["cohen_kappa"]
        for result in label_results.values()
    ) / len(label_results)

    mean_raw_agreement = sum(
        result["raw_agreement"]
        for result in label_results.values()
    ) / len(label_results)

    # Frequency after simple agreement-only consensus:
    # count positive only where both reviewers marked 1.
    agreed_positive_counts = {
        label: label_results[label][
            "both_positive_count"
        ]
        for label in LABELS
    }

    summary = {
        "num_examples": len(uids),
        "labels": LABELS,
        "mean_cohen_kappa": float(
            mean_kappa
        ),
        "mean_raw_agreement": float(
            mean_raw_agreement
        ),
        "per_label": label_results,
        "total_disagreements":
            len(all_disagreements),
        "agreed_positive_counts":
            agreed_positive_counts,
        "note": (
            "Per-label Cohen's kappa is the primary "
            "agreement statistic. Disagreement rows "
            "should be adjudicated separately; do not "
            "replace reviewer values before reporting "
            "inter-reviewer agreement."
        ),
    }

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "rq1_annotation_agreement.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    disagreement_path = (
        output_dir
        / "rq1_annotation_disagreements.csv"
    )

    write_disagreements(
        all_disagreements,
        disagreement_path,
    )

    print("=" * 72)
    print("RQ1.3 ANNOTATION AGREEMENT")
    print("=" * 72)
    print("Examples:", len(uids))
    print()

    for label in LABELS:
        result = label_results[
            label
        ]

        print(label)
        print(
            "  kappa:",
            round(
                result["cohen_kappa"],
                4,
            ),
        )
        print(
            "  raw agreement:",
            round(
                result["raw_agreement"],
                4,
            ),
        )
        print(
            "  reviewer1 positives:",
            result[
                "reviewer1_positive_count"
            ],
        )
        print(
            "  reviewer2 positives:",
            result[
                "reviewer2_positive_count"
            ],
        )
        print(
            "  disagreements:",
            result[
                "disagreement_count"
            ],
        )
        print()

    print(
        "Mean kappa:",
        round(
            mean_kappa,
            4,
        ),
    )

    print(
        "Mean raw agreement:",
        round(
            mean_raw_agreement,
            4,
        ),
    )

    print(
        "Total label disagreements:",
        len(all_disagreements),
    )

    print()
    print(
        "Summary:",
        summary_path,
    )

    print(
        "Disagreements:",
        disagreement_path,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()

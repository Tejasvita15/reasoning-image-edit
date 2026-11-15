# src/analyze_rq1.py

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


DEFAULT_SCORE_KEYS = [
    "judge_result.Q",
    "judge_result.primary_score",
    "judge_result.aggregate",
    "judge_result.score",
    "Q",
    "primary_score",
    "aggregate",
    "score",
]


def load_jsonl(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows = []

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

                if row.get("status", "success") == "success":
                    rows.append(row)

    return rows


def get_nested_value(
    obj: Dict[str, Any],
    dotted_key: str,
) -> Any:
    value: Any = obj

    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]

    return value


def extract_score(
    row: Dict[str, Any],
    score_key: Optional[str],
) -> Optional[float]:
    keys = [score_key] if score_key else DEFAULT_SCORE_KEYS

    for key in keys:
        if not key:
            continue

        value = get_nested_value(row, key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def aggregate_by_example(
    rows: List[Dict[str, Any]],
    score_key: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Average repeated seeds within each example first.

    This is the required aggregation order for K-small:
        seed-level scores -> example mean -> paired analysis.
    """

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        uid = row.get("example_uid")
        if uid:
            grouped[uid].append(row)

    result = {}

    for uid, items in grouped.items():
        scores = [
            score
            for score in (
                extract_score(item, score_key)
                for item in items
            )
            if score is not None
        ]

        latencies = [
            float(item["latency_total_sec"])
            for item in items
            if item.get("latency_total_sec") is not None
        ]

        result[uid] = {
            "example_uid": uid,
            "knowledge_type": items[0].get("knowledge_type"),
            "category": items[0].get("category"),
            "num_rows": len(items),
            "seeds": sorted(
                {
                    item.get("base_seed")
                    for item in items
                    if item.get("base_seed") is not None
                }
            ),
            "score": mean_or_none(scores),
            "latency_total_sec": mean_or_none(latencies),
        }

    return result


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Optional[Tuple[float, float]]:
    if differences.size == 0:
        return None

    rng = np.random.default_rng(seed)

    sample_indices = rng.integers(
        0,
        differences.size,
        size=(n_bootstrap, differences.size),
    )

    boot_means = differences[sample_indices].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0

    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))

    return lower, upper


def wilcoxon_result(
    before: np.ndarray,
    after: np.ndarray,
) -> Optional[Dict[str, float]]:
    if wilcoxon is None or before.size == 0:
        return None

    diff = after - before

    if np.allclose(diff, 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
        }

    result = wilcoxon(
        after,
        before,
        alternative="two-sided",
        zero_method="wilcox",
    )

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def paired_comparison(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
    *,
    label: str,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    common = sorted(
        set(before.keys()) & set(after.keys())
    )

    paired_rows = []

    for uid in common:
        score_before = before[uid]["score"]
        score_after = after[uid]["score"]

        if score_before is None or score_after is None:
            continue

        paired_rows.append(
            {
                "example_uid": uid,
                "knowledge_type":
                    before[uid].get("knowledge_type"),
                "category":
                    before[uid].get("category"),
                "before": score_before,
                "after": score_after,
                "difference": score_after - score_before,
                "before_latency_sec":
                    before[uid].get("latency_total_sec"),
                "after_latency_sec":
                    after[uid].get("latency_total_sec"),
            }
        )

    if not paired_rows:
        return {
            "label": label,
            "n_pairs": 0,
            "warning": (
                "No paired judge scores were found. "
                "Populate judge_result or provide --score-key."
            ),
            "pairs": [],
        }

    before_scores = np.asarray(
        [row["before"] for row in paired_rows],
        dtype=np.float64,
    )

    after_scores = np.asarray(
        [row["after"] for row in paired_rows],
        dtype=np.float64,
    )

    differences = after_scores - before_scores

    ci = paired_bootstrap_ci(
        differences,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )

    result = {
        "label": label,
        "n_pairs": len(paired_rows),
        "mean_before": float(before_scores.mean()),
        "mean_after": float(after_scores.mean()),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "improved_fraction": float(np.mean(differences > 0)),
        "unchanged_fraction": float(np.mean(differences == 0)),
        "degraded_fraction": float(np.mean(differences < 0)),
        "paired_bootstrap_95_ci": (
            list(ci) if ci is not None else None
        ),
        "wilcoxon": wilcoxon_result(
            before_scores,
            after_scores,
        ),
        "pairs": paired_rows,
    }

    # Descriptive per-knowledge-type breakdown.
    by_type: Dict[str, List[float]] = defaultdict(list)

    for row in paired_rows:
        by_type[
            row.get("knowledge_type") or "unknown"
        ].append(row["difference"])

    result["by_knowledge_type"] = {
        knowledge_type: {
            "n": len(values),
            "mean_difference": float(np.mean(values)),
            "median_difference": float(np.median(values)),
        }
        for knowledge_type, values in sorted(by_type.items())
    }

    latency_pairs = [
        (
            row["before_latency_sec"],
            row["after_latency_sec"],
        )
        for row in paired_rows
        if row["before_latency_sec"] is not None
        and row["after_latency_sec"] is not None
    ]

    if latency_pairs:
        before_latency = np.asarray(
            [pair[0] for pair in latency_pairs],
            dtype=np.float64,
        )
        after_latency = np.asarray(
            [pair[1] for pair in latency_pairs],
            dtype=np.float64,
        )

        result["latency"] = {
            "n_pairs": len(latency_pairs),
            "mean_before_sec":
                float(before_latency.mean()),
            "mean_after_sec":
                float(after_latency.mean()),
            "mean_overhead_sec":
                float(
                    (after_latency - before_latency).mean()
                ),
            "mean_ratio_after_over_before":
                float(
                    np.mean(
                        after_latency / before_latency
                    )
                ),
        }

    return result


def correction_metadata_summary(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not rows:
        return {"n_rows": 0}

    retry_values = [
        bool(row.get("retry_triggered"))
        for row in rows
        if row.get("retry_triggered") is not None
    ]

    attempts = [
        int(row["attempt_count"])
        for row in rows
        if row.get("attempt_count") is not None
    ]

    selected_not_last = []

    for row in rows:
        selected = row.get("selected_candidate_index")
        candidates = row.get("candidate_paths") or []

        if selected is None or not candidates:
            continue

        selected_not_last.append(
            int(selected) != len(candidates) - 1
        )

    latencies = [
        float(row["latency_total_sec"])
        for row in rows
        if row.get("latency_total_sec") is not None
    ]

    return {
        "n_rows": len(rows),
        "retry_trigger_rate": (
            float(np.mean(retry_values))
            if retry_values else None
        ),
        "mean_attempt_count": (
            float(np.mean(attempts))
            if attempts else None
        ),
        "max_attempt_count": (
            int(max(attempts))
            if attempts else None
        ),
        "selected_not_last_rate": (
            float(np.mean(selected_not_last))
            if selected_not_last else None
        ),
        "mean_latency_total_sec": (
            float(np.mean(latencies))
            if latencies else None
        ),
        "median_latency_total_sec": (
            float(np.median(latencies))
            if latencies else None
        ),
    }


def save_pair_csv(
    comparison: Dict[str, Any],
    path: Path,
) -> None:
    pairs = comparison.get("pairs", [])

    if not pairs:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "example_uid",
        "knowledge_type",
        "category",
        "before",
        "after",
        "difference",
        "before_latency_sec",
        "after_latency_sec",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in pairs:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )


def print_comparison(
    comparison: Dict[str, Any],
) -> None:
    print()
    print("=" * 70)
    print(comparison["label"])
    print("=" * 70)

    if comparison.get("n_pairs", 0) == 0:
        print(comparison.get("warning"))
        return

    print("Paired examples:", comparison["n_pairs"])
    print("Mean before:", comparison["mean_before"])
    print("Mean after:", comparison["mean_after"])
    print("Mean difference:", comparison["mean_difference"])
    print(
        "95% paired-bootstrap CI:",
        comparison["paired_bootstrap_95_ci"],
    )
    print("Improved fraction:", comparison["improved_fraction"])
    print("Degraded fraction:", comparison["degraded_fraction"])
    print("Wilcoxon:", comparison["wilcoxon"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Step1X RQ1 Base, Thinking, "
            "and Correction experiment logs."
        )
    )

    parser.add_argument(
        "--base-log",
        nargs="+",
        required=True,
        help=(
            "One or more Base JSONL logs. "
            "Multiple seed logs may be supplied."
        ),
    )

    parser.add_argument(
        "--thinking-log",
        nargs="+",
        required=True,
        help=(
            "One or more Thinking-only JSONL logs."
        ),
    )

    parser.add_argument(
        "--correction-log",
        nargs="*",
        default=[],
        help=(
            "Optional Thinking+Correction JSONL logs."
        ),
    )

    parser.add_argument(
        "--score-key",
        default=None,
        help=(
            "Optional dotted path to the primary judge score, "
            "for example judge_result.Q. "
            "If omitted, common score fields are tried."
        ),
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        default="analysis/rq1",
    )

    args = parser.parse_args()

    base_rows = load_jsonl(args.base_log)
    thinking_rows = load_jsonl(args.thinking_log)
    correction_rows = (
        load_jsonl(args.correction_log)
        if args.correction_log
        else []
    )

    base = aggregate_by_example(
        base_rows,
        args.score_key,
    )

    thinking = aggregate_by_example(
        thinking_rows,
        args.score_key,
    )

    correction = aggregate_by_example(
        correction_rows,
        args.score_key,
    )

    rq11 = paired_comparison(
        base,
        thinking,
        label="RQ1.1: Base -> Thinking",
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    rq12 = None

    if correction_rows:
        rq12 = paired_comparison(
            thinking,
            correction,
            label="RQ1.2: Thinking -> Correction",
            n_bootstrap=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )

    correction_metadata = correction_metadata_summary(
        correction_rows
    )

    output = {
        "score_key_requested": args.score_key,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "input_counts": {
            "base_rows": len(base_rows),
            "thinking_rows": len(thinking_rows),
            "correction_rows": len(correction_rows),
            "base_examples": len(base),
            "thinking_examples": len(thinking),
            "correction_examples": len(correction),
        },
        "rq1_1_base_vs_thinking": rq11,
        "rq1_2_thinking_vs_correction": rq12,
        "correction_metadata": correction_metadata,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir / "rq1_analysis.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    save_pair_csv(
        rq11,
        output_dir
        / "rq1_1_base_vs_thinking_pairs.csv",
    )

    if rq12 is not None:
        save_pair_csv(
            rq12,
            output_dir
            / "rq1_2_thinking_vs_correction_pairs.csv",
        )

    print_comparison(rq11)

    if rq12 is not None:
        print_comparison(rq12)

    print()
    print("=" * 70)
    print("CORRECTION METADATA")
    print("=" * 70)
    print(
        json.dumps(
            correction_metadata,
            indent=2,
        )
    )

    print()
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()

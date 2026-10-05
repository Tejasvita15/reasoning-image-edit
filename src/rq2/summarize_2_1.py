# src/rq2/summarize_2_1.py

"""
RQ2.1 summary table from the imported judge scores.

    python -m src.rq2.summarize_2_1

Reads outputs/rq2/judge/chatgpt_main/scores_main.jsonl (one row per output)
and writes, per thinking mode x guidance:
    outputs/rq2/analysis/rq2_1_summary.csv
    outputs/rq2/analysis/rq2_1_summary.md

Q is on 0-100. Components are the judge's 1-5 scores. Knowledge
Plausibility is averaged over knowledge-family outputs only.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


COLUMNS = (
    "thinking",
    "true_cfg_scale",
    "n",
    "mean_Q",
    "visual_consistency",
    "visual_quality",
    "instruction_following",
    "knowledge_plausibility",
    "n_knowledge",
)


def parse_condition(condition: str):
    """
    rq2_step1x_think-<mode>_cfg<g>_steps<n> -> (thinking, guidance).
    """

    thinking = "on" if "_think-cached_" in condition else "off"
    guidance = float(condition.split("_cfg")[1].split("_")[0])

    return thinking, guidance


def summarize(rows):

    groups = {}

    for row in rows:
        groups.setdefault(parse_condition(row["condition"]), []).append(row)

    table = []

    for (thinking, guidance), group in sorted(groups.items()):
        knowledge = [r["knowledge_score"] for r in group if r["knowledge_required"]]

        table.append({
            "thinking": thinking,
            "true_cfg_scale": f"{guidance:g}",
            "n": len(group),
            "mean_Q": round(statistics.mean(r["Q"] for r in group), 2),
            "visual_consistency": round(statistics.mean(r["consistency_score"] for r in group), 2),
            "visual_quality": round(statistics.mean(r["quality_score"] for r in group), 2),
            "instruction_following": round(statistics.mean(r["instruction_score"] for r in group), 2),
            "knowledge_plausibility": round(statistics.mean(knowledge), 2) if knowledge else None,
            "n_knowledge": len(knowledge),
        })

    return table


def write_markdown(path: Path, table, rows, calibration_report) -> None:

    judge = sorted({r["judge_model"] for r in rows})
    steps = sorted({r["generation_snapshot"]["split"] for r in rows})

    lines = [
        "# RQ2.1 summary: guidance x thinking (Step1X-Edit-v1p2)",
        "",
        f"Judge: {', '.join(judge)} (ChatGPT UI, prompt version "
        f"{rows[0]['evaluation_protocol']['prompt_version']}). "
        f"Split: {', '.join(steps)}, seed 42, 50 steps, reflection off.",
    ]

    if calibration_report:
        lines.append(
            f"Judge repeatability (calibration, {calibration_report['outputs']} outputs): "
            f"MMD = {calibration_report['candidate_minimum_meaningful_difference']:.2f} Q points, "
            f"mean |dQ| = {calibration_report['mean_absolute_difference']:.2f}."
        )

    lines += [
        "",
        "Q on 0-100; components are mean judge scores on 1-5; knowledge plausibility "
        "over knowledge-family outputs only.",
        "",
        "| Thinking | Guidance | n | Q | Visual consistency | Visual quality "
        "| Instruction following | Knowledge plausibility (n) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for row in table:
        lines.append(
            f"| {row['thinking']} | {row['true_cfg_scale']} | {row['n']} | {row['mean_Q']:.1f} "
            f"| {row['visual_consistency']:.2f} | {row['visual_quality']:.2f} "
            f"| {row['instruction_following']:.2f} "
            f"| {row['knowledge_plausibility']:.2f} ({row['n_knowledge']}) |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():

    parser = argparse.ArgumentParser(description="RQ2.1 summary table.")
    parser.add_argument("--scores", default="outputs/rq2/judge/chatgpt_main/scores_main.jsonl")
    parser.add_argument("--calibration_report", default="outputs/rq2/judge/chatgpt_calibration/repeatability.json")
    parser.add_argument("--out_dir", default="outputs/rq2/analysis")
    args = parser.parse_args()

    with open(args.scores, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    report_path = Path(args.calibration_report)
    calibration_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else None
    )

    table = summarize(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "rq2_1_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)

    write_markdown(out_dir / "rq2_1_summary.md", table, rows, calibration_report)

    print(f"Wrote {out_dir / 'rq2_1_summary.csv'} and {out_dir / 'rq2_1_summary.md'}")


if __name__ == "__main__":
    main()

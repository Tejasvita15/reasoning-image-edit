# src/rq2/summarize_2_2.py

"""
RQ2.2 summary on K-dev and the frozen configuration for the K-test run.

    python -m src.rq2.summarize_2_2

Combines, at the RQ2.1-selected guidance (6):
    - judge scores: 20 and 28 steps from RQ2.2, 50 steps from RQ2.1
    - latency: the single-node timing table (src.rq2.summarize_2_2_latency)

and writes:
    outputs/rq2_2/analysis/rq2_2_summary.csv / .md
    outputs/rq2_2/analysis/rq2_2_frozen_config.json

Frozen-configuration rule (agreed before any K-test run): the configuration
with the highest mean Q on K-dev. Its edge over the runners-up is recorded
as is; it is not claimed to be significant.
"""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


COLUMNS = (
    "thinking",
    "steps",
    "n",
    "latency_total_s",
    "mean_Q",
    "visual_consistency",
    "visual_quality",
    "instruction_following",
    "knowledge_plausibility",
    "generator_calls",
)

FROZEN_RULE = "highest mean Q on K-dev (guidance 6, thinking off/on x 20/28/50 steps)"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def config_of(row):
    thinking = "on" if "_think-cached_" in row["condition"] else "off"
    steps = int(row["condition"].split("_steps")[1])
    return thinking, steps


def summarize(rows, latency):

    groups = {}

    for row in rows:
        groups.setdefault(config_of(row), []).append(row)

    table = []

    for (thinking, steps), group in sorted(groups.items()):
        knowledge = [r["knowledge_score"] for r in group if r["knowledge_required"]]

        table.append({
            "thinking": thinking,
            "steps": steps,
            "n": len(group),
            "latency_total_s": latency[(thinking, steps)],
            "mean_Q": round(statistics.mean(r["Q"] for r in group), 2),
            "visual_consistency": round(statistics.mean(r["consistency_score"] for r in group), 2),
            "visual_quality": round(statistics.mean(r["quality_score"] for r in group), 2),
            "instruction_following": round(statistics.mean(r["instruction_score"] for r in group), 2),
            "knowledge_plausibility": round(statistics.mean(knowledge), 2),
            "generator_calls": steps * 2,
        })

    return table


def write_markdown(path: Path, table, frozen, mmd) -> None:

    lines = [
        "# RQ2.2 summary: steps x thinking on K-dev (Step1X-Edit-v1p2, guidance 6)",
        "",
        "Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2). "
        "K-dev, 90 examples, seed 42, reflection off. 20/28 steps from RQ2.2, "
        "50 steps from RQ2.1. Latency: end-to-end per image (preprocess + think + "
        "generate + postprocess), single-node timing run on an A100 "
        "(see rq2_2_latency.md).",
        f"Judge MMD (RQ2.1 calibration): {mmd:.2f} Q points.",
        "",
        "Q on 0-100; components are mean judge scores on 1-5; knowledge plausibility "
        "over knowledge-family outputs only.",
        "",
        "| Thinking | Steps | n | Latency (s) | Q | Visual consistency | Visual quality "
        "| Instruction following | Knowledge plausibility | Generator calls |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for row in table:
        lines.append(
            f"| {row['thinking']} | {row['steps']} | {row['n']} | {row['latency_total_s']:.1f} "
            f"| {row['mean_Q']:.1f} | {row['visual_consistency']:.2f} | {row['visual_quality']:.2f} "
            f"| {row['instruction_following']:.2f} | {row['knowledge_plausibility']:.2f} "
            f"| {row['generator_calls']} |"
        )

    lines += [
        "",
        f"Frozen configuration for the single K-test run ({FROZEN_RULE}): "
        f"thinking {frozen['thinking']}, {frozen['steps']} steps, guidance 6 "
        f"(mean Q {frozen['mean_Q']:.1f}).",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():

    parser = argparse.ArgumentParser(description="RQ2.2 summary and frozen configuration.")
    parser.add_argument("--scores_2_1", default="outputs/rq2/judge/chatgpt_main/scores_main.jsonl")
    parser.add_argument("--scores_2_2", default="outputs/rq2_2/judge/chatgpt_main/scores_main.jsonl")
    parser.add_argument("--latency", default="outputs/rq2_2/analysis/rq2_2_latency.csv")
    parser.add_argument("--calibration_report", default="outputs/rq2/judge/chatgpt_calibration/repeatability.json")
    parser.add_argument("--out_dir", default="outputs/rq2_2/analysis")
    args = parser.parse_args()

    rows = [
        r for r in load_jsonl(args.scores_2_1)
        if "_cfg6_steps50" in r["condition"]
    ] + load_jsonl(args.scores_2_2)

    with open(args.latency, "r", encoding="utf-8") as f:
        latency = {
            (r["thinking"], int(r["steps"])): float(r["total_s"])
            for r in csv.DictReader(f)
        }

    mmd = json.loads(Path(args.calibration_report).read_text(encoding="utf-8"))[
        "candidate_minimum_meaningful_difference"
    ]

    table = summarize(rows, latency)
    ranked = sorted(table, key=lambda row: row["mean_Q"], reverse=True)
    frozen = ranked[0]

    if len(ranked) > 1 and ranked[1]["mean_Q"] == frozen["mean_Q"]:
        raise SystemExit("Tie for the highest mean Q; agree a tie-break before freezing.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "rq2_2_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)

    write_markdown(out_dir / "rq2_2_summary.md", table, frozen, mmd)

    decision = {
        "rule": FROZEN_RULE,
        "rule_agreed_before_k_test": True,
        "frozen_configuration": {
            "thinking": frozen["thinking"],
            "thinking_condition": "cached" if frozen["thinking"] == "on" else "off",
            "num_inference_steps": frozen["steps"],
            "true_cfg_scale": 6,
            "reflection": "off",
        },
        "k_dev_mean_Q": {f"thinking-{r['thinking']}_steps{r['steps']}": r["mean_Q"] for r in ranked},
        "margin_over_runner_up": round(frozen["mean_Q"] - ranked[1]["mean_Q"], 2),
        "mmd": mmd,
        "note": (
            "The margin over the runner-up is below the MMD and its paired bootstrap 95% CI "
            "includes 0, so the selected configuration is not claimed to be significantly "
            "better; K-test reports this K-dev-selected configuration on held-out data."
        ),
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_dir / "rq2_2_frozen_config.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_dir}/rq2_2_summary.csv, rq2_2_summary.md, rq2_2_frozen_config.json")
    print(f"Frozen: thinking {frozen['thinking']}, {frozen['steps']} steps (mean Q {frozen['mean_Q']:.2f}); "
          f"margin over runner-up {decision['margin_over_runner_up']:.2f}")


if __name__ == "__main__":
    main()

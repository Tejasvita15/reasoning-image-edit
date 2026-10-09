# src/rq2/summarize_2_2_ktest.py

"""
RQ2.2 frozen-configuration result on K-test.

    python -m src.rq2.summarize_2_2_ktest

Reads the imported K-test judge scores and the frozen-configuration record,
and writes:
    outputs/rq2_2/analysis/rq2_2_ktest_summary.csv / .md

Reports mean Q with a 95% paired-example bootstrap interval (10,000 resamples,
plan S8), the component means, and per-knowledge-type means (descriptive only),
next to the same configuration's K-dev mean. This is reported separately from
RQ1.1 (different settings).
"""

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from src.taxonomy import get_knowledge_type


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def bootstrap_ci(values):
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return means[int(0.025 * BOOTSTRAP_RESAMPLES) - 1], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


def summarize(rows, label):
    knowledge = [r["knowledge_score"] for r in rows if r["knowledge_required"]]
    low, high = bootstrap_ci([r["Q"] for r in rows])

    return {
        "subset": label,
        "n": len(rows),
        "mean_Q": round(statistics.mean(r["Q"] for r in rows), 2),
        "Q_ci95_low": round(low, 2),
        "Q_ci95_high": round(high, 2),
        "visual_consistency": round(statistics.mean(r["consistency_score"] for r in rows), 2),
        "visual_quality": round(statistics.mean(r["quality_score"] for r in rows), 2),
        "instruction_following": round(statistics.mean(r["instruction_score"] for r in rows), 2),
        "knowledge_plausibility": round(statistics.mean(knowledge), 2) if knowledge else None,
        "n_knowledge": len(knowledge),
    }


def main():

    parser = argparse.ArgumentParser(description="RQ2.2 K-test summary.")
    parser.add_argument("--scores", default="outputs/rq2_2_ktest/judge/chatgpt_main/scores_main.jsonl")
    parser.add_argument("--frozen", default="outputs/rq2_2/analysis/rq2_2_frozen_config.json")
    parser.add_argument("--out_dir", default="outputs/rq2_2/analysis")
    args = parser.parse_args()

    with open(args.scores, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    config = frozen["frozen_configuration"]
    expected = f"_think-{config['thinking_condition']}_cfg{config['true_cfg_scale']:g}_steps{config['num_inference_steps']}"

    if {r["condition"] for r in rows} != {f"rq2_step1x{expected}"}:
        raise SystemExit("K-test scores are not exactly the frozen configuration.")

    dev_key = f"thinking-{config['thinking']}_steps{config['num_inference_steps']}"

    table = [summarize(rows, "all")]

    for knowledge_type in ("factual", "conceptual", "procedural"):
        subset = [r for r in rows if get_knowledge_type(r["example_uid"].rsplit("__", 1)[0]) == knowledge_type]
        table.append(summarize(subset, knowledge_type))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "rq2_2_ktest_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    lines = [
        "# RQ2.2 frozen configuration on K-test (Step1X-Edit-v1p2)",
        "",
        f"Configuration frozen on K-dev before any K-test run ({frozen['rule']}): "
        f"thinking {config['thinking']}, {config['num_inference_steps']} steps, guidance "
        f"{config['true_cfg_scale']}, reflection off. K-test, seed 42. Judge: GPT-6 Astra Pro "
        "(ChatGPT UI, prompt version kris-chatgpt-ui-v2). Reported separately from RQ1.1 "
        "(different settings).",
        f"Same configuration on K-dev: mean Q {frozen['k_dev_mean_Q'][dev_key]:.2f} (n=90). "
        f"Judge MMD: {frozen['mmd']:.2f} Q points.",
        "",
        f"Q on 0-100 with a 95% bootstrap interval ({BOOTSTRAP_RESAMPLES:,} resamples over examples); "
        "components are mean judge scores on 1-5. Per-knowledge-type rows are descriptive only.",
        "",
        "| Subset | n | Q | 95% CI | Visual consistency | Visual quality "
        "| Instruction following | Knowledge plausibility (n) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for row in table:
        kp = "-" if row["knowledge_plausibility"] is None else f"{row['knowledge_plausibility']:.2f} ({row['n_knowledge']})"
        lines.append(
            f"| {row['subset']} | {row['n']} | {row['mean_Q']:.1f} "
            f"| {row['Q_ci95_low']:.1f}-{row['Q_ci95_high']:.1f} | {row['visual_consistency']:.2f} "
            f"| {row['visual_quality']:.2f} | {row['instruction_following']:.2f} | {kp} |"
        )

    (out_dir / "rq2_2_ktest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_dir / 'rq2_2_ktest_summary.csv'} and {out_dir / 'rq2_2_ktest_summary.md'}")


if __name__ == "__main__":
    main()

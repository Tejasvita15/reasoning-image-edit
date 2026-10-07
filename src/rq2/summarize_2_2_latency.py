# src/rq2/summarize_2_2_latency.py

"""
RQ2.2 latency table from the single-job timing run.

    python -m src.rq2.summarize_2_2_latency

Reads logs/rq2_2_timing (every configuration timed in one job on one node,
after warm-up; generation is the median of the per-row timing repeats) and
writes, per thinking mode x steps:
    outputs/rq2_2/analysis/rq2_2_latency.csv
    outputs/rq2_2/analysis/rq2_2_latency.md

Each stage is the median over the timed examples, in seconds. Thinking is the
one thinking pass charged to replayed thinking-on rows, timed on the same node
(median of its own repeats). The total is reconstructed from the stages
(preprocess + think + generate + postprocess).
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


COLUMNS = (
    "thinking",
    "steps",
    "true_cfg_scale",
    "n_examples",
    "timing_repeats",
    "preprocess_s",
    "think_s",
    "generate_s",
    "postprocess_s",
    "total_s",
    "generator_calls",
    "thinking_tokens",
    "peak_vram_gb",
)


def median_of(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else None


def summarize(rows):

    groups = {}

    for row in rows:
        thinking = "on" if row["thinking_condition"] == "cached" else "off"
        groups.setdefault((thinking, row["num_inference_steps"]), []).append(row)

    table = []

    for (thinking, steps), group in sorted(groups.items()):
        table.append({
            "thinking": thinking,
            "steps": steps,
            "true_cfg_scale": f"{group[0]['true_cfg_scale']:g}",
            "n_examples": len(group),
            "timing_repeats": min(len(r["latency_pipeline_repeats_sec"]) for r in group),
            "preprocess_s": round(median_of(group, "latency_preprocess_sec"), 3),
            "think_s": (
                round(median_of(group, "latency_thinking_sec"), 2)
                if thinking == "on"
                else 0.0
            ),
            "generate_s": round(median_of(group, "latency_pipeline_median_sec"), 2),
            "postprocess_s": round(median_of(group, "latency_postprocess_sec"), 3),
            "total_s": round(median_of(group, "latency_reconstructed_total_sec"), 2),
            "generator_calls": int(median_of(group, "generator_calls")),
            "thinking_tokens": round(median_of(group, "thinking_token_count")),
            "peak_vram_gb": round(median_of(group, "peak_vram_gb"), 1),
        })

    return table


def write_markdown(path: Path, table, rows) -> None:

    env = rows[0]["env"]
    nodes = sorted({r["env"]["hostname"] for r in rows})

    lines = [
        "# RQ2.2 latency: steps x thinking (Step1X-Edit-v1p2)",
        "",
        f"Timed in one job on one node ({', '.join(nodes)}; {env['gpu']}), "
        f"offload: {rows[0]['offload_mode']}, after a warm-up generation. "
        f"Each stage is the median over {table[0]['n_examples']} K-dev examples; "
        f"generation per example is the median of {table[0]['timing_repeats']} repeats. "
        "Thinking is one thinking pass (timed on the same node), charged to every "
        "thinking-on generation. Total is the median of each example's "
        "preprocess + think + generate + postprocess, so it can differ slightly "
        "from the sum of the stage medians.",
        "",
        "| Thinking | Steps | Preprocess (s) | Think (s) | Generate (s) | Postprocess (s) "
        "| Total (s) | Generator calls | Thinking tokens | Peak VRAM (GB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for row in table:
        lines.append(
            f"| {row['thinking']} | {row['steps']} | {row['preprocess_s']:.3f} "
            f"| {row['think_s']:.2f} | {row['generate_s']:.2f} | {row['postprocess_s']:.3f} "
            f"| {row['total_s']:.2f} | {row['generator_calls']} | {row['thinking_tokens']} "
            f"| {row['peak_vram_gb']:.1f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():

    parser = argparse.ArgumentParser(description="RQ2.2 latency table.")
    parser.add_argument("--log_dir", default="logs/rq2_2_timing")
    parser.add_argument("--out_dir", default="outputs/rq2_2/analysis")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for path in sorted(Path(args.log_dir).glob("*.jsonl"))
        for line in open(path, "r", encoding="utf-8")
        if line.strip()
    ]

    rows = [row for row in rows if row["status"] == "success"]

    if len({row["env"]["hostname"] for row in rows}) != 1:
        raise SystemExit("Timing rows come from more than one node; re-run the timing job.")

    table = summarize(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "rq2_2_latency.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)

    write_markdown(out_dir / "rq2_2_latency.md", table, rows)

    print(f"Wrote {out_dir / 'rq2_2_latency.csv'} and {out_dir / 'rq2_2_latency.md'}")


if __name__ == "__main__":
    main()

# src/rq2/stats_rq2.py

"""
Formal statistics for RQ2.1 and RQ2.2 (plan S8).

    python -m src.rq2.stats_rq2

Every comparison is paired by example (same K-dev examples, same noise):
    - mean of the per-example differences (A - B)
    - 95% paired bootstrap CI (10,000 resamples over examples)
    - Wilcoxon signed-rank test as a check (zero differences dropped)
    - examples favouring A / tied / favouring B
    - for Q only: whether |mean difference| reaches the judge's MMD (plan:
      do not claim differences smaller than the MMD)

Metrics: Q (primary, 0-100) and the judge components on 1-5: visual
consistency (VC), visual quality (VQ), instruction following (IF), and
knowledge plausibility (KP, knowledge-family examples only).

Writes:
    outputs/rq2/analysis/rq2_1_stats.csv / .md
    outputs/rq2_2/analysis/rq2_2_stats.csv / .md
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

# metric -> (score field, knowledge-family examples only)
METRICS = {
    "Q": ("Q", False),
    "VC": ("consistency_score", False),
    "VQ": ("quality_score", False),
    "IF": ("instruction_score", False),
    "KP": ("knowledge_score", True),
}

COLUMNS = (
    "rq", "family", "comparison", "metric", "n", "mean_a", "mean_b",
    "mean_diff", "ci_low", "ci_high", "wilcoxon_p", "n_nonzero",
    "a_better", "ties", "b_better", "reaches_mmd", "latency_diff_s",
)


# =============================================================
# Data
# =============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def config_of(condition):
    """
    rq2_step1x_think-<off|cached>_cfg<g>_steps<n> -> (thinking, cfg, steps).
    """

    thinking = "on" if "_think-cached_" in condition else "off"
    cfg = float(condition.split("_cfg")[1].split("_")[0])
    steps = int(condition.split("_steps")[1])

    return thinking, cfg, steps


def by_config(rows):
    configs = {}

    for row in rows:
        configs.setdefault(config_of(row["condition"]), {})[row["example_uid"]] = row

    return configs


def values(config_rows, metric):
    field, knowledge_only = METRICS[metric]

    return {
        uid: row[field]
        for uid, row in config_rows.items()
        if not knowledge_only or row["knowledge_required"]
    }


def describe(config):
    thinking, cfg, steps = config
    return f"thinking {thinking}, cfg {cfg:g}, {steps} steps"


# =============================================================
# Statistics
# =============================================================

def bootstrap_ci(differences):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    data = np.asarray(differences, dtype=float)
    idx = rng.integers(0, len(data), size=(BOOTSTRAP_RESAMPLES, len(data)))
    low, high = np.percentile(data[idx].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def wilcoxon_p(differences):
    nonzero = [d for d in differences if d != 0]

    if not nonzero:
        return 1.0, 0

    return float(wilcoxon(differences, zero_method="wilcox").pvalue), len(nonzero)


def summarize_differences(differences, a_values, b_values, metric, mmd):
    low, high = bootstrap_ci(differences)
    p, n_nonzero = wilcoxon_p(differences)
    mean_diff = float(np.mean(differences))

    # Full precision here; rounded only when written.
    return {
        "n": len(differences),
        "mean_a": float(np.mean(a_values)),
        "mean_b": float(np.mean(b_values)),
        "mean_diff": mean_diff,
        "ci_low": low,
        "ci_high": high,
        "wilcoxon_p": p,
        "n_nonzero": n_nonzero,
        "a_better": sum(d > 0 for d in differences),
        "ties": sum(d == 0 for d in differences),
        "b_better": sum(d < 0 for d in differences),
        "reaches_mmd": (abs(mean_diff) >= mmd) if metric == "Q" else "",
    }


def compare(configs, a, b, metric, mmd):
    """
    A - B over the examples both configurations share.
    """

    va, vb = values(configs[a], metric), values(configs[b], metric)
    uids = sorted(set(va) & set(vb))

    return summarize_differences(
        [va[u] - vb[u] for u in uids],
        [va[u] for u in uids],
        [vb[u] for u in uids],
        metric,
        mmd,
    )


def compare_interaction(configs, high_on, low_on, high_off, low_off, metric, mmd):
    """
    Per example: (on high - on low) - (off high - off low). Negative means
    thinking flattens the response to the dial.
    """

    v = [values(configs[c], metric) for c in (high_on, low_on, high_off, low_off)]
    uids = sorted(set.intersection(*(set(x) for x in v)))
    on = [v[0][u] - v[1][u] for u in uids]
    off = [v[2][u] - v[3][u] for u in uids]

    return summarize_differences([x - y for x, y in zip(on, off)], on, off, metric, mmd)


# =============================================================
# Comparisons
# =============================================================

def rq2_1_comparisons(configs, mmd):
    rows = []

    def add(family, label, result):
        rows.append({"rq": "2.1", "family": family, "comparison": label, **result, "latency_diff_s": ""})

    for metric in METRICS:
        for cfg in (2.0, 6.0, 10.0):
            a, b = ("on", cfg, 50), ("off", cfg, 50)
            add("thinking effect", f"{describe(a)} vs {describe(b)}", {"metric": metric, **compare(configs, a, b, metric, mmd)})

        for thinking in ("off", "on"):
            for high, low in ((2.0, 6.0), (10.0, 6.0), (10.0, 2.0)):
                a, b = (thinking, high, 50), (thinking, low, 50)
                add("guidance effect", f"{describe(a)} vs {describe(b)}", {"metric": metric, **compare(configs, a, b, metric, mmd)})

        add(
            "guidance sensitivity (interaction)",
            "(on cfg10 - on cfg2) - (off cfg10 - off cfg2), 50 steps",
            {"metric": metric, **compare_interaction(
                configs, ("on", 10.0, 50), ("on", 2.0, 50), ("off", 10.0, 50), ("off", 2.0, 50), metric, mmd
            )},
        )

    return rows


def rq2_2_comparisons(configs, latency, mmd):
    rows = []

    def add(family, a, b, metric):
        result = compare(configs, a, b, metric, mmd)
        rows.append({
            "rq": "2.2", "family": family, "comparison": f"{describe(a)} vs {describe(b)}",
            "metric": metric, **result,
            "latency_diff_s": round(latency[(a[0], a[2])] - latency[(b[0], b[2])], 2),
        })

    for metric in METRICS:
        for steps in (20, 28, 50):
            add("thinking at matched latency", ("on", 6.0, steps), ("off", 6.0, steps), metric)

        for thinking in ("off", "on"):
            for more, fewer in ((28, 20), (50, 28), (50, 20)):
                add("steps effect", (thinking, 6.0, more), (thinking, 6.0, fewer), metric)

        # The plan's illustrative pair (thinking costs ~1.4 s here, so the two
        # are not latency-matched; reported because the plan names it).
        add("plan example pair", ("on", 6.0, 20), ("off", 6.0, 50), metric)

    return rows


def guidance_selection(configs, mmd):
    """
    RQ2.1 rule: highest mean Q; keep the default (cfg 6) if the winner beats
    it by less than the MMD.
    """

    lines = []

    for thinking in ("off", "on"):
        means = {
            cfg: float(np.mean([r["Q"] for r in configs[(thinking, cfg, 50)].values()]))
            for cfg in (2.0, 6.0, 10.0)
        }
        best = max(means, key=means.get)
        margin = means[best] - means[6.0]
        chosen = best if margin >= mmd else 6.0

        lines.append(
            f"| {thinking} | {means[2.0]:.2f} | {means[6.0]:.2f} | {means[10.0]:.2f} "
            f"| cfg {best:g} | {margin:+.2f} | **cfg {chosen:g}** |"
        )

    return lines


# =============================================================
# Output
# =============================================================

def fmt_diff(row):
    return f"{row['mean_diff']:+.2f} [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]"


def write_csv(path: Path, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for row in rows
        )


def write_markdown(path: Path, title, intro, rows, mmd, extra=None, show_latency=False) -> None:

    lines = [f"# {title}", "", *intro, "",
             f"Paired by example; mean of per-example differences (A - B) with a 95% bootstrap "
             f"CI ({BOOTSTRAP_RESAMPLES:,} resamples over examples) and a Wilcoxon signed-rank p "
             f"(zero differences dropped) as a check; no multiple-comparison correction. "
             f"Judge MMD = {mmd:.2f} Q points: differences in Q smaller than the MMD are not "
             "claimed as meaningful (plan S8). Components are on the judge's 1-5 scale; "
             "KP covers knowledge-family examples only.", ""]

    if extra:
        lines += extra + [""]

    families = list(dict.fromkeys(r["family"] for r in rows))

    for family in families:
        q_rows = [r for r in rows if r["family"] == family and r["metric"] == "Q"]

        lines += [f"## {family[0].upper() + family[1:]}", "",
                  "### Q (primary)", "",
                  "| Comparison (A vs B) | n | Q A | Q B | A - B [95% CI] | Wilcoxon p "
                  "| A better / tie / B better | Reaches MMD |"
                  + (" Latency A - B (s) |" if show_latency else ""),
                  "|---|---|---|---|---|---|---|---|" + ("---|" if show_latency else "")]

        for r in q_rows:
            lines.append(
                f"| {r['comparison']} | {r['n']} | {r['mean_a']:.2f} | {r['mean_b']:.2f} "
                f"| {fmt_diff(r)} | {r['wilcoxon_p']:.4f} | {r['a_better']} / {r['ties']} / {r['b_better']} "
                f"| {'yes' if r['reaches_mmd'] else 'no'} |"
                + (f" {r['latency_diff_s']:+.2f} |" if show_latency else "")
            )

        lines += ["", "### Components (A - B [95% CI], Wilcoxon p)", "",
                  "| Comparison (A vs B) | VC | VQ | IF | KP |", "|---|---|---|---|---|"]

        for label in dict.fromkeys(r["comparison"] for r in q_rows):
            cells = []

            for metric in ("VC", "VQ", "IF", "KP"):
                r = next(x for x in rows if x["family"] == family and x["comparison"] == label and x["metric"] == metric)
                cells.append(f"{fmt_diff(r)}, p={r['wilcoxon_p']:.3f}")

            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():

    parser = argparse.ArgumentParser(description="RQ2.1 / RQ2.2 statistics (plan S8).")
    parser.add_argument("--scores_2_1", default="outputs/rq2/judge/chatgpt_main/scores_main.jsonl")
    parser.add_argument("--scores_2_2", default="outputs/rq2_2/judge/chatgpt_main/scores_main.jsonl")
    parser.add_argument("--latency", default="outputs/rq2_2/analysis/rq2_2_latency.csv")
    parser.add_argument("--calibration_report", default="outputs/rq2/judge/chatgpt_calibration/repeatability.json")
    parser.add_argument("--out_2_1", default="outputs/rq2/analysis")
    parser.add_argument("--out_2_2", default="outputs/rq2_2/analysis")
    args = parser.parse_args()

    configs = by_config(load_jsonl(args.scores_2_1) + load_jsonl(args.scores_2_2))

    with open(args.latency, "r", encoding="utf-8") as f:
        latency = {(r["thinking"], int(r["steps"])): float(r["total_s"]) for r in csv.DictReader(f)}

    mmd = json.loads(Path(args.calibration_report).read_text(encoding="utf-8"))[
        "candidate_minimum_meaningful_difference"
    ]

    judge = ("Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2). "
             "K-dev, 90 examples, seed 42, reflection off.")

    rows_2_1 = rq2_1_comparisons(configs, mmd)
    out_2_1 = Path(args.out_2_1)
    out_2_1.mkdir(parents=True, exist_ok=True)
    write_csv(out_2_1 / "rq2_1_stats.csv", rows_2_1)
    write_markdown(
        out_2_1 / "rq2_1_stats.md",
        "RQ2.1 statistics: guidance x thinking (Step1X-Edit-v1p2, 50 steps)",
        [judge],
        rows_2_1,
        mmd,
        extra=[
            "## Guidance selected for RQ2.2",
            "",
            "Rule: highest mean Q; keep the default (cfg 6) if the winner beats it by less than the MMD.",
            "",
            "| Thinking | Q cfg 2 | Q cfg 6 | Q cfg 10 | Highest | Winner - cfg 6 | Selected |",
            "|---|---|---|---|---|---|---|",
            *guidance_selection(configs, mmd),
        ],
    )

    rows_2_2 = rq2_2_comparisons(configs, latency, mmd)
    out_2_2 = Path(args.out_2_2)
    out_2_2.mkdir(parents=True, exist_ok=True)
    write_csv(out_2_2 / "rq2_2_stats.csv", rows_2_2)
    write_markdown(
        out_2_2 / "rq2_2_stats.md",
        "RQ2.2 statistics: steps x thinking (Step1X-Edit-v1p2, guidance 6)",
        [judge + " 20/28 steps from RQ2.2, 50 steps from RQ2.1. Latency: end-to-end "
         "per image from the single-node timing run (rq2_2_latency.md). Thinking costs "
         "~1.4 s, under one ~2 s denoising step, so thinking on vs off at the same step "
         "count is the matched-latency comparison."],
        rows_2_2,
        mmd,
        show_latency=True,
    )

    print(f"Wrote {out_2_1 / 'rq2_1_stats.csv'}, {out_2_1 / 'rq2_1_stats.md'}")
    print(f"Wrote {out_2_2 / 'rq2_2_stats.csv'}, {out_2_2 / 'rq2_2_stats.md'}")


if __name__ == "__main__":
    main()

# src/rq2/summarize_2_3.py

"""
RQ2.3 results: feedback vs resampling (K-small, seeds 42 / 1234 / 7).

    python -m src.rq2.summarize_2_3

Every RQ2.3 image was judged once, blind to arm and selector. Each strategy's
result is the judge score of the candidate it keeps:

    First candidate only    candidate 1 (no second attempt; reference)
    Resampling (selector)   EditReward's pick of candidate 1 / resampling candidate 2
    Feedback (selector)     EditReward's pick of candidate 1 / feedback candidate 2
    Feedback (native)       Step1X's own pick from its reflection scores

Both arms make a second candidate exactly when Step1X's reflection rejects
candidate 1, so their candidate budgets (and generator calls) match per
example. Latency per arm = thinking (charged from the cache) + every timed
pipeline call of that arm (generation, and for feedback the reflection rounds
and native selection) + EditReward selection when the arm has two candidates.
Pre/postprocessing is excluded (identical across arms, < 3 s).

Statistics (plan S8): per-example values are averaged over the three seeds
first; comparisons are paired over the 40 examples (95% bootstrap CI with
10,000 resamples, Wilcoxon as a check, MMD flag for Q). Per-knowledge-type
and per-seed results are descriptive only.

Writes outputs/rq2_3/analysis/:
    rq2_3_summary.md / .csv       arm means, latency, budget, by type and seed
    rq2_3_stats.md / .csv         paired comparisons
    rq2_3_per_example.csv         every example-seed: candidate Q and picks
    rq2_3_second_candidate.csv    did candidate 2 beat candidate 1
    figures/rq2_3_paired_differences.png
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from src.rq2.stats_rq2 import (
    METRICS,
    load_jsonl,
    summarize_differences,
    write_csv,
    write_markdown,
)
from src.taxonomy import get_knowledge_type


SEEDS = (42, 1234, 7)
CONDITION = "rq2_3_step1x_think-cached_cfg6_steps28"

ARMS = {
    "first": "First candidate only",
    "resample": "Resampling (selector)",
    "feedback": "Feedback (selector)",
    "native": "Feedback (native)",
}

COMPARISONS = (
    ("feedback", "resample"),
    ("native", "resample"),
    ("resample", "first"),
    ("feedback", "first"),
    ("native", "first"),
)

JUDGE_CANDIDATE = {
    "cand1": "cand1",
    "feedback-cand2": "feedback_cand2",
    "resample-cand2": "resample_cand2",
}


# =============================================================
# Data
# =============================================================

def latest_by_uid(path):
    return {row["example_uid"]: row for row in load_jsonl(path)}


def load_seed(seed, args):
    """
    Per example: judge rows of each candidate, the arm picks and latencies.
    """

    generation = latest_by_uid(Path(args.log_dir) / f"{CONDITION}_k_small_seed{seed}.jsonl")
    selection = latest_by_uid(Path(args.log_dir) / "selector" / f"rq2_3_selector_k_small_seed{seed}.jsonl")

    scores = {}

    for bundle in ("cand1", "cand2"):
        for row in load_jsonl(Path(args.judge_dir) / f"chatgpt_{bundle}_seed{seed}" / "scores_main.jsonl"):
            candidate = JUDGE_CANDIDATE[row["condition"].rsplit("_", 1)[1]]
            scores[(row["example_uid"], candidate)] = row

    examples = {}

    for uid, gen in generation.items():

        sel = selection[uid]
        fb = gen["feedback"]
        rs = gen["resample"]

        judged = {}

        for candidate, sha in gen["candidate_sha256"].items():
            row = scores[(uid, candidate)]

            if row["output_sha256"] != sha or sel["candidate_sha256"][candidate] != sha:
                raise SystemExit(f"seed {seed} {uid}: {candidate} judged/selected image differs from generation")

            judged[candidate] = row

        retried = fb["attempt_count"] == 2
        thinking = gen["latency_thinking_sec"]

        picks = {
            "first": "cand1",
            "resample": sel["selection"]["resample"]["selected"],
            "feedback": sel["selection"]["feedback"]["selected"],
            "native": "feedback_cand2" if fb["native_selected_index"] == 1 else "cand1",
        }

        latency = {
            "first": {"thinking": thinking, "generation": rs["cand1_latency_pipeline_sec"], "selector": 0.0},
            "resample": {
                "thinking": thinking,
                "generation": rs["cand1_latency_pipeline_sec"] + (rs["redraw_latency_pipeline_sec"] or 0.0),
                "selector": sel["selection"]["resample"]["latency_selector_sec"],
            },
            # The feedback run is one timed call: generation, reflection on
            # every candidate, and native selection.
            "feedback": {"thinking": thinking, "generation": fb["latency_pipeline_sec"],
                         "selector": sel["selection"]["feedback"]["latency_selector_sec"]},
            "native": {"thinking": thinking, "generation": fb["latency_pipeline_sec"], "selector": 0.0},
        }

        candidates = {"first": 1, "resample": 2 if rs["redraw"] else 1,
                      "feedback": fb["attempt_count"], "native": fb["attempt_count"]}

        if rs["redraw"] != retried or rs["redraw_forced"]:
            raise SystemExit(f"seed {seed} {uid}: budgets do not match")

        examples[uid] = {
            "seed": seed,
            "uid": uid,
            "knowledge_type": get_knowledge_type(uid.rsplit("__", 1)[0]),
            "retried": retried,
            "retry_type": fb["retry_type"],
            "judged": judged,
            "picks": picks,
            "latency": latency,
            "candidates": candidates,
            "generator_calls": {"first": generator_calls(gen, 1), "resample": rs["generator_calls"],
                                "feedback": fb["generator_calls"], "native": fb["generator_calls"]},
            "selector_scores": {name: value["score"] for name, value in sel["scores"].items()},
        }

    return examples


def generator_calls(gen, attempts):
    return gen["num_inference_steps"] * 2 * attempts


def metric_value(example, arm, metric):
    field, knowledge_only = METRICS[metric]
    row = example["judged"][example["picks"][arm]]

    if knowledge_only and not row["knowledge_required"]:
        return None

    return row[field]


def per_example(data, arm, metric):
    """
    {uid: mean over seeds} for one arm and metric (plan S8: seeds first).
    """

    out = {}

    for uid in sorted({uid for seed in data for uid in data[seed]}):
        vals = [metric_value(data[seed][uid], arm, metric) for seed in data]
        vals = [v for v in vals if v is not None]

        if vals:
            out[uid] = float(np.mean(vals))

    return out


def per_example_latency(data, arm):
    return {
        uid: float(np.mean([sum(data[seed][uid]["latency"][arm].values()) for seed in data]))
        for uid in sorted({uid for seed in data for uid in data[seed]})
    }


# =============================================================
# Tables
# =============================================================

def stats_rows(data, mmd):

    rows = []

    for a, b in COMPARISONS:

        lat_a, lat_b = per_example_latency(data, a), per_example_latency(data, b)
        latency_diff = float(np.mean([lat_a[u] - lat_b[u] for u in lat_a]))

        for metric in METRICS:
            va, vb = per_example(data, a, metric), per_example(data, b, metric)
            uids = sorted(set(va) & set(vb))

            row = summarize_differences(
                [va[u] - vb[u] for u in uids],
                [va[u] for u in uids],
                [vb[u] for u in uids],
                metric,
                mmd,
            )

            rows.append({"rq": "2.3", "family": "feedback vs resampling",
                         "comparison": f"{ARMS[a]} vs {ARMS[b]}", "metric": metric,
                         **row, "latency_diff_s": latency_diff})

    return rows


def bootstrap_mean_ci(values):
    from src.rq2.stats_rq2 import bootstrap_ci

    low, high = bootstrap_ci(values)
    return float(np.mean(values)), low, high


def arm_summary(data):

    rows = []
    flat = [ex for seed in data for ex in data[seed].values()]

    for arm, label in ARMS.items():
        q = per_example(data, arm, "Q")
        mean, low, high = bootstrap_mean_ci(list(q.values()))

        lat = [ex["latency"][arm] for ex in flat]

        row = {
            "arm": label,
            "n_examples": len(q),
            "mean_Q": mean,
            "Q_ci95_low": low,
            "Q_ci95_high": high,
        }

        for metric, name in (("VC", "visual_consistency"), ("VQ", "visual_quality"),
                             ("IF", "instruction_following"), ("KP", "knowledge_plausibility")):
            row[name] = float(np.mean(list(per_example(data, arm, metric).values())))

        for knowledge_type in ("factual", "conceptual", "procedural"):
            row[f"Q_{knowledge_type}"] = float(np.mean([
                v for u, v in q.items() if get_knowledge_type(u.rsplit("__", 1)[0]) == knowledge_type
            ]))

        for seed in data:
            row[f"Q_seed{seed}"] = float(np.mean([metric_value(ex, arm, "Q") for ex in data[seed].values()]))

        row.update({
            "mean_candidates": float(np.mean([ex["candidates"][arm] for ex in flat])),
            "mean_generator_calls": float(np.mean([ex["generator_calls"][arm] for ex in flat])),
            "latency_thinking_s": float(np.mean([x["thinking"] for x in lat])),
            "latency_generation_s": float(np.mean([x["generation"] for x in lat])),
            "latency_selector_s": float(np.mean([x["selector"] for x in lat])),
            "latency_total_s": float(np.mean([sum(x.values()) for x in lat])),
            "kept_candidate_2": sum(ex["picks"][arm] != "cand1" for ex in flat),
        })

        rows.append(row)

    return rows


def second_candidate_rows(data):
    """
    Over example-seeds with a second candidate: did it beat candidate 1
    (by judge Q), and did each selection rule keep the judge-better one?
    """

    flat = [ex for seed in data for ex in data[seed].values() if ex["retried"]]
    rows = []

    for arm, second in (("feedback", "feedback_cand2"), ("resample", "resample_cand2")):
        for subset, members in (("all retries", flat),
                                ("feedback retry = refine", [ex for ex in flat if ex["retry_type"] == "refine"]),
                                ("feedback retry = restart", [ex for ex in flat if ex["retry_type"] == "restart"])):

            diffs = [ex["judged"][second]["Q"] - ex["judged"]["cand1"]["Q"] for ex in members]
            decided = [ex for ex, d in zip(members, diffs) if d != 0]

            def agrees(ex, pick):
                better = second if ex["judged"][second]["Q"] > ex["judged"]["cand1"]["Q"] else "cand1"
                return pick == better

            row = {
                "candidate_2": "feedback" if arm == "feedback" else "resampling",
                "subset": subset,
                "n": len(members),
                "beat_candidate_1": sum(d > 0 for d in diffs),
                "tied": sum(d == 0 for d in diffs),
                "lost_to_candidate_1": sum(d < 0 for d in diffs),
                "mean_Q_gain": float(np.mean(diffs)) if diffs else None,
                "selector_kept_better": sum(agrees(ex, ex["picks"][arm]) for ex in decided),
                "native_kept_better": (sum(agrees(ex, ex["picks"]["native"]) for ex in decided)
                                       if arm == "feedback" else None),
                "n_decided": len(decided),
            }

            rows.append(row)

    return rows


def write_rows_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in rows)


def write_summary_md(path, arms, second, data, mmd, judge):

    flat = [ex for seed in data for ex in data[seed].values()]
    retried = [ex for ex in flat if ex["retried"]]
    n_retry = sum(ex["retried"] for ex in flat)
    retry_types = {t: sum(ex["retry_type"] == t for ex in retried) for t in ("refine", "restart")}

    lines = [
        "# RQ2.3 Feedback vs resampling (Step1X-Edit-v1p2)",
        "",
        "K-small (40 examples) x seeds 42, 1234, 7; thinking on (cached plan), guidance 6, 28 steps. "
        "Both strategies share candidate 1. Feedback: Step1X reflection with max_try_cnt=2. Resampling: "
        "a fresh draw from the same plan with new noise. A second candidate is made only when Step1X's "
        "reflection rejects candidate 1, in both arms, so candidate budgets match per example. "
        "Selector: EditReward-MiMo-VL-7B on the original KRIS instruction, never the judge. "
        + judge,
        "",
        f"Second attempts: {n_retry}/{len(flat)} example-seeds ({100 * n_retry / len(flat):.0f}%); "
        f"feedback retry type: refine (edit candidate 1) {retry_types['refine']}, restart (redraw from "
        f"source) {retry_types['restart']}.",
        "",
        "Q on 0-100: per-example mean over the three seeds, then the mean over the 40 examples, with a "
        f"95% bootstrap interval. Judge MMD = {mmd:.2f} Q points. Latency is the mean per example-seed "
        "on one A100 node with model CPU offload: thinking (charged from the cache) + generation "
        "(+ reflection and native selection for feedback) + EditReward selection when there are two "
        "candidates. Feedback and resampling match in candidates and generator calls, not in latency.",
        "",
        "| Strategy | Q [95% CI] | VC | VQ | IF | KP | Candidates | Generator calls "
        "| Think + generate (+ reflect) + select = total (s) | Kept candidate 2 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in arms:
        lines.append(
            f"| {r['arm']} | {r['mean_Q']:.2f} [{r['Q_ci95_low']:.2f}, {r['Q_ci95_high']:.2f}] "
            f"| {r['visual_consistency']:.2f} | {r['visual_quality']:.2f} | {r['instruction_following']:.2f} "
            f"| {r['knowledge_plausibility']:.2f} | {r['mean_candidates']:.2f} | {r['mean_generator_calls']:.0f} "
            f"| {r['latency_thinking_s']:.1f} + {r['latency_generation_s']:.1f} + {r['latency_selector_s']:.1f} "
            f"= {r['latency_total_s']:.1f} | {r['kept_candidate_2']}/{len(flat)} |"
        )

    lines += [
        "",
        "## Did the second candidate beat the first? (judge Q, example-seeds with a second candidate)",
        "",
        "\"Kept the better\" counts, over pairs the judge did not tie, how often the selection rule kept the "
        "candidate the judge scored higher.",
        "",
        "| Candidate 2 | Subset | n | Beat / tie / lost to candidate 1 | Mean Q gain | "
        "Selector kept the better | Native kept the better |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in second:
        native = "-" if r["native_kept_better"] is None else f"{r['native_kept_better']}/{r['n_decided']}"
        gain = "-" if r["mean_Q_gain"] is None else f"{r['mean_Q_gain']:+.2f}"
        lines.append(
            f"| {r['candidate_2']} | {r['subset']} | {r['n']} | {r['beat_candidate_1']} / {r['tied']} / "
            f"{r['lost_to_candidate_1']} | {gain} | {r['selector_kept_better']}/{r['n_decided']} | {native} |"
        )

    lines += [
        "",
        "The refine/restart split is by the feedback arm's retry type; for resampling it groups the same "
        "example-seeds (resampling itself always redraws).",
        "",
        "## By knowledge type and seed (descriptive only)",
        "",
        "| Strategy | Factual | Conceptual | Procedural | "
        + " | ".join(f"Seed {s}" for s in data) + " |",
        "|---|---|---|---|" + "---|" * len(data),
    ]

    for r in arms:
        lines.append(
            f"| {r['arm']} | {r['Q_factual']:.2f} | {r['Q_conceptual']:.2f} | {r['Q_procedural']:.2f} | "
            + " | ".join(f"{r[f'Q_seed{s}']:.2f}" for s in data) + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_example(path, data):
    rows = []

    for seed in data:
        for uid, ex in sorted(data[seed].items()):
            judged = ex["judged"]
            rows.append({
                "seed": seed,
                "example_uid": uid,
                "knowledge_type": ex["knowledge_type"],
                "retried": ex["retried"],
                "retry_type": ex["retry_type"] or "",
                "Q_cand1": judged["cand1"]["Q"],
                "Q_feedback_cand2": judged["feedback_cand2"]["Q"] if "feedback_cand2" in judged else "",
                "Q_resample_cand2": judged["resample_cand2"]["Q"] if "resample_cand2" in judged else "",
                "editreward_cand1": ex["selector_scores"]["cand1"],
                "editreward_feedback_cand2": ex["selector_scores"].get("feedback_cand2", ""),
                "editreward_resample_cand2": ex["selector_scores"].get("resample_cand2", ""),
                **{f"pick_{arm}": ex["picks"][arm] for arm in ARMS},
                **{f"Q_{arm}": metric_value(ex, arm, "Q") for arm in ARMS},
                **{f"latency_{arm}_s": sum(ex["latency"][arm].values()) for arm in ARMS},
            })

    write_rows_csv(path, rows)


# =============================================================
# Figure
# =============================================================

def figure(stats, data, mmd, out_dir):
    """
    Paired Q differences (95% bootstrap CI) per comparison, with the extra
    latency each costs. Paired over examples, so example-to-example spread
    (which dominates the arm means' CIs) cancels.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from src.rq2.plots_rq2 import AXIS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, save, style_axes, titled

    # Strategy A of each comparison carries the mark: resampling and feedback
    # use the two validated hues; Step1X's own selection is the feedback hue
    # with a hollow marker.
    marks = {
        "resample": ("#2a78d6", "o", True),
        "feedback": ("#eb6834", "s", True),
        "native": ("#eb6834", "s", False),
    }

    groups = (
        ("Second attempt vs first candidate only", [("resample", "first"), ("feedback", "first"), ("native", "first")]),
        ("Feedback vs resampling (same candidate budget)", [("feedback", "resample"), ("native", "resample")]),
    )

    q = {(r["comparison"]): r for r in stats if r["metric"] == "Q"}

    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6, linestyle="-")

    y, ticks, labels = 0, [], []

    for title, pairs in groups:
        ax.text(-0.02, y + 0.15, title, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8.5, color=TEXT_PRIMARY, fontweight="bold")
        y -= 0.85

        for a, b in pairs:
            r = q[f"{ARMS[a]} vs {ARMS[b]}"]
            color, marker, filled = marks[a]

            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=color, linewidth=1.6, zorder=2)
            ax.plot([r["mean_diff"]], [y], linestyle="none", marker=marker, markersize=8, color=color,
                    markerfacecolor=color if filled else SURFACE,
                    markeredgecolor=SURFACE if filled else color, markeredgewidth=1.6, zorder=3)

            ax.text(1.02, y, f"{r['mean_diff']:+.2f} Q   {r['latency_diff_s']:+.0f} s",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, color=TEXT_PRIMARY)

            ticks.append(y)
            labels.append(f"{ARMS[a]}\nvs {ARMS[b].lower()}")
            y -= 1

        y -= 0.35

    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.6, 0.6)

    span = max(abs(r["ci_low"]) for r in q.values()) , max(abs(r["ci_high"]) for r in q.values())
    limit = max(span) + 0.8
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Paired difference in Q, A - B (0-100 scale), mean over 40 examples")
    ax.text(1.02, 0.6, "Q diff   extra latency", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=7.5, color=TEXT_SECONDARY)

    handles = [
        Line2D([], [], linestyle="none", marker=m, markersize=7, color=c,
               markerfacecolor=c if f else SURFACE, markeredgewidth=1.6, label=ARMS[arm])
        for arm, (c, m, f) in marks.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)

    flat = [ex for seed in data for ex in data[seed].values()]
    fig.text(0.0, -0.1,
             f"Per-example values averaged over seeds 42, 1234, 7 first; 95% paired bootstrap CI (10,000 resamples). "
             f"A second candidate is made only when Step1X's reflection rejects the first "
             f"({sum(ex['retried'] for ex in flat)}/{len(flat)} example-seeds), in both arms. "
             f"Judge MMD = {mmd:.1f} Q, far outside this axis: no difference here is meaningful.",
             fontsize=7.5, color=TEXT_SECONDARY, ha="left", va="top", wrap=True)

    titled(fig, ax, "RQ2.3 Feedback vs resampling: paired change in quality",
           "Step1X-Edit-v1p2 · K-small (n=40) x 3 seeds · guidance 6 · 28 steps · thinking on")

    save(fig, out_dir, "rq2_3_paired_differences")


# =============================================================
# Main
# =============================================================

def main():

    parser = argparse.ArgumentParser(description="RQ2.3 summary, statistics and figure.")
    parser.add_argument("--log_dir", default="logs/rq2_3")
    parser.add_argument("--judge_dir", default="outputs/rq2_3/judge")
    parser.add_argument("--calibration_report", default="outputs/rq2/judge/chatgpt_calibration/repeatability.json")
    parser.add_argument("--out_dir", default="outputs/rq2_3/analysis")
    args = parser.parse_args()

    mmd = json.loads(Path(args.calibration_report).read_text(encoding="utf-8"))[
        "candidate_minimum_meaningful_difference"
    ]

    data = {seed: load_seed(seed, args) for seed in SEEDS}

    if any(len(data[seed]) != 40 for seed in SEEDS):
        raise SystemExit("Every seed must have all 40 K-small examples.")

    judge = ("Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2), blind to arm and "
             "selector; every distinct image judged once.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = arm_summary(data)
    second = second_candidate_rows(data)
    stats = stats_rows(data, mmd)

    write_rows_csv(out_dir / "rq2_3_summary.csv", arms)
    write_rows_csv(out_dir / "rq2_3_second_candidate.csv", second)
    write_summary_md(out_dir / "rq2_3_summary.md", arms, second, data, mmd, judge)

    write_csv(out_dir / "rq2_3_stats.csv", stats)
    write_markdown(
        out_dir / "rq2_3_stats.md",
        "RQ2.3 statistics: feedback vs resampling (Step1X-Edit-v1p2)",
        ["K-small (40 examples) x seeds 42, 1234, 7; thinking on (cached plan), guidance 6, 28 steps. "
         "Each example's value is the mean over the three seeds (plan S8), so n = 40. " + judge,
         "Feedback and resampling match in candidate budget and generator calls per example; the last "
         "column is their mean latency difference (A - B)."],
        stats,
        mmd,
        show_latency=True,
    )

    write_per_example(out_dir / "rq2_3_per_example.csv", data)
    figure(stats, data, mmd, out_dir / "figures")

    for r in arms:
        print(f"{r['arm']:<24} Q {r['mean_Q']:.2f} [{r['Q_ci95_low']:.2f}, {r['Q_ci95_high']:.2f}] "
              f"| latency {r['latency_total_s']:.1f}s | candidates {r['mean_candidates']:.2f}")

    for r in stats:
        if r["metric"] == "Q":
            print(f"{r['comparison']:<50} {r['mean_diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] "
                  f"p={r['wilcoxon_p']:.3f} | latency {r['latency_diff_s']:+.1f}s")

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

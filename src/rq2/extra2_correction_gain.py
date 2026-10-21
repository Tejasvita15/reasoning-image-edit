# src/rq2/extra2_correction_gain.py

"""
Extra 2 (extras.md): when does correction help?

    python -m src.rq2.extra2_correction_gain
    python -m src.rq2.extra2_correction_gain --model Step1X=<csv> --model UniReason=<csv>

Uses RQ2.3's existing judge scores (no new runs or judging). Each --model is
an RQ2.3 per-example CSV (src/rq2/summarize_2_3.py format: seed, example_uid,
retried, Q_cand1, Q_feedback_cand2, Q_resample_cand2), one panel per model.

Per example and seed with a second candidate (Step1X: only runs where the
reflection triggered a retry; the others are counted and reported):
    feedback gain   = Q(feedback 2nd)   - Q(first)
    resampling gain = Q(resampling 2nd) - Q(first)
    difference      = feedback gain - resampling gain
                    = Q(feedback 2nd) - Q(resampling 2nd)
Raw second-candidate scores are used, not the selector's pick.

Each is summarised against Q(first) with Spearman rho and the mean per tercile
of Q(first), both with 95% bootstrap CIs that resample examples (all of an
example's seeds together; 10,000 resamples). Terciles are cut on Q(first)
values, so tied scores stay in one tercile. If any tercile has fewer than 10
runs, the tercile results are descriptive only and the overall rho is the
summary.

Low first scores tend to rise on any retry (regression to the mean), so a
downward trend in either gain alone proves nothing. Both arms share the first
candidate, so that effect cancels in the difference.

Writes outputs/rq2_3/analysis/:
    extra2_correction_gain.md / .csv
    figures/extra2_correction_gain.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
MIN_TERCILE_RUNS = 10

QUANTITIES = {
    "feedback": "Feedback gain",
    "resample": "Resampling gain",
    "difference": "Feedback - resampling",
}


# =============================================================
# Data
# =============================================================

def load_runs(path):
    """
    (runs with a second candidate, number of runs without one).
    """

    runs, skipped = [], 0

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):

            if row["retried"] != "True":
                skipped += 1
                continue

            first = float(row["Q_cand1"])
            feedback = float(row["Q_feedback_cand2"]) - first
            resample = float(row["Q_resample_cand2"]) - first

            runs.append({
                "example_uid": row["example_uid"],
                "seed": int(row["seed"]),
                "first": first,
                "feedback": feedback,
                "resample": resample,
                "difference": feedback - resample,
            })

    return runs, skipped


def tercile_edges(firsts):
    low, high = np.quantile(firsts, [1 / 3, 2 / 3])
    return float(low), float(high)


def tercile_of(value, edges):
    low, high = edges
    return 0 if value <= low else (1 if value <= high else 2)


# =============================================================
# Statistics
# =============================================================

def bootstrap_examples(runs):
    """
    Index arrays of resampled runs, resampling whole examples.
    """

    uids = sorted({r["example_uid"] for r in runs})
    by_uid = {uid: [i for i, r in enumerate(runs) if r["example_uid"] == uid] for uid in uids}

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for _ in range(BOOTSTRAP_RESAMPLES):
        picks = rng.integers(0, len(uids), size=len(uids))
        yield np.concatenate([by_uid[uids[p]] for p in picks])


def spearman(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")

    return float(spearmanr(x, y).correlation)


def summarize(runs, edges):

    first = np.array([r["first"] for r in runs])
    groups = np.array([tercile_of(r["first"], edges) for r in runs])
    values = {q: np.array([r[q] for r in runs]) for q in QUANTITIES}

    boot = {q: {"mean": [], "rho": [], "terciles": [[], [], []]} for q in QUANTITIES}

    for idx in bootstrap_examples(runs):
        for q in QUANTITIES:
            v = values[q][idx]
            boot[q]["mean"].append(v.mean())
            boot[q]["rho"].append(spearman(first[idx], v))

            for t in range(3):
                members = v[groups[idx] == t]
                boot[q]["terciles"][t].append(members.mean() if len(members) else np.nan)

    def ci(samples):
        return [float(x) for x in np.nanpercentile(samples, [2.5, 97.5])]

    rows = []

    for q, label in QUANTITIES.items():
        v = values[q]
        row = {
            "quantity": label,
            "n_runs": len(runs),
            "mean": float(v.mean()),
            "mean_ci": ci(boot[q]["mean"]),
            "rho": spearman(first, v),
            "rho_ci": ci(boot[q]["rho"]),
            "terciles": [],
        }

        for t in range(3):
            members = v[groups == t]
            row["terciles"].append({
                "n": int(len(members)),
                "mean": float(members.mean()) if len(members) else float("nan"),
                "ci": ci(boot[q]["terciles"][t]),
            })

        rows.append(row)

    return rows


# =============================================================
# Outputs
# =============================================================

def tercile_labels(runs, edges):
    labels = []

    for t in range(3):
        firsts = [r["first"] for r in runs if tercile_of(r["first"], edges) == t]
        labels.append(f"Q(first) {min(firsts):.0f}-{max(firsts):.0f}" if firsts else "empty")

    return labels


def write_outputs(results, out_dir):

    csv_rows = []

    for model, res in results.items():
        for row in res["rows"]:
            base = {"model": model, "quantity": row["quantity"]}

            csv_rows.append({**base, "subset": "all", "n_runs": row["n_runs"], "mean_gain": row["mean"],
                             "ci_low": row["mean_ci"][0], "ci_high": row["mean_ci"][1],
                             "spearman_rho": row["rho"], "rho_ci_low": row["rho_ci"][0],
                             "rho_ci_high": row["rho_ci"][1]})

            for label, t in zip(res["tercile_labels"], row["terciles"]):
                csv_rows.append({**base, "subset": label, "n_runs": t["n"], "mean_gain": t["mean"],
                                 "ci_low": t["ci"][0], "ci_high": t["ci"][1],
                                 "spearman_rho": "", "rho_ci_low": "", "rho_ci_high": ""})

    with open(out_dir / "extra2_correction_gain.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in csv_rows)

    lines = [
        "# Extra 2: when does correction help? (RQ2.3 data)",
        "",
        "Gains are judge Q (0-100) of a second candidate minus Q of the shared first candidate, per example "
        "and seed, using raw second-candidate scores (not the selector's pick). Existing RQ2.3 judge scores; "
        "no new runs or judging. 95% CIs bootstrap whole examples (all seeds together, "
        f"{BOOTSTRAP_RESAMPLES:,} resamples). Terciles are cut on Q(first) values.",
        "",
        "Caveat: low first scores tend to rise on any retry (regression to the mean), so a downward trend in "
        "either gain alone proves nothing. Feedback - resampling cancels that effect, because both arms share "
        "the first candidate.",
        "",
    ]

    for model, res in results.items():
        descriptive = any(t["n"] < MIN_TERCILE_RUNS for t in res["rows"][0]["terciles"])

        lines += [
            f"## {model}",
            "",
            f"Runs with a second candidate: {res['n_runs']} over {res['n_examples']} examples; "
            f"runs without one (reflection accepted the first candidate, not plotted): {res['skipped']}.",
            "",
            "| | Mean gain [95% CI] | Spearman ρ vs Q(first) [95% CI] | "
            + " | ".join(f"{label} (n={t['n']})" for label, t in zip(res["tercile_labels"], res["rows"][0]["terciles"]))
            + " |",
            "|---|---|---|---|---|---|",
        ]

        for row in res["rows"]:
            lines.append(
                f"| {row['quantity']} | {row['mean']:+.2f} [{row['mean_ci'][0]:+.2f}, {row['mean_ci'][1]:+.2f}] "
                f"| {row['rho']:+.2f} [{row['rho_ci'][0]:+.2f}, {row['rho_ci'][1]:+.2f}] | "
                + " | ".join(f"{t['mean']:+.2f} [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]" for t in row["terciles"])
                + " |"
            )

        lines += [
            "",
            ("Every tercile has fewer than "
             if all(t["n"] < MIN_TERCILE_RUNS for t in res["rows"][0]["terciles"])
             else "At least one tercile has fewer than ") + f"{MIN_TERCILE_RUNS} runs, so tercile results are descriptive "
            "only; the overall Spearman ρ is the summary." if descriptive else
            f"Every tercile has at least {MIN_TERCILE_RUNS} runs.",
            "",
        ]

    (out_dir / "extra2_correction_gain.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure(results, out_dir):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from src.rq2.plots_rq2 import AXIS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, save, style_axes

    colors = {"feedback": "#eb6834", "resample": "#2a78d6", "difference": "#52514e"}
    markers = {"feedback": "s", "resample": "o", "difference": "D"}
    dodge = {"feedback": -0.12, "resample": 0.12, "difference": 0.0}

    models = list(results)
    # Shared y-axis so the gains and their difference read on one scale.
    fig, axes = plt.subplots(len(models), 2, figsize=(10.4, 4.3 * len(models)), facecolor=SURFACE,
                             squeeze=False, sharey=True)
    fig.subplots_adjust(wspace=0.32, hspace=0.75, top=0.84)

    for row_axes, model in zip(axes, models):
        res = results[model]
        stats = {q: next(r for r in res["rows"] if r["quantity"] == QUANTITIES[q]) for q in QUANTITIES}

        for ax, quantities, title in (
            (row_axes[0], ("feedback", "resample"), "Gain of each second candidate over the first"),
            (row_axes[1], ("difference",), "Feedback gain - resampling gain"),
        ):
            style_axes(ax)
            ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)

            for q in quantities:
                s = stats[q]
                xs = np.arange(3) + dodge[q]
                means = [t["mean"] for t in s["terciles"]]
                errors = [[t["mean"] - t["ci"][0] for t in s["terciles"]],
                          [t["ci"][1] - t["mean"] for t in s["terciles"]]]

                ax.errorbar(xs, means, yerr=errors, color=colors[q], linewidth=0, elinewidth=1.2,
                            capsize=0, zorder=2)
                ax.plot(xs, means, color=colors[q], linewidth=2.0, marker=markers[q], markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
                ax.annotate(f"{QUANTITIES[q]}\nρ = {s['rho']:+.2f}", (xs[-1], means[-1]), xytext=(10, 0),
                            textcoords="offset points", va="center", fontsize=8, color=TEXT_PRIMARY)

            ax.set_xticks(range(3), [f"{label}\n(n={t['n']})" for label, t in
                                     zip(res["tercile_labels"], stats[quantities[0]]["terciles"])], fontsize=8)
            ax.set_xlim(-0.4, 3.0)
            ax.tick_params(axis="y", labelleft=True)
            ax.set_ylabel("Mean change in Q (0-100)")
            ax.set_xlabel("Tercile of first-candidate Q")
            ax.set_title(f"{model}: {title}", loc="left", fontsize=10, color=TEXT_PRIMARY, fontweight="bold")

        handles = [Line2D([], [], color=colors[q], marker=markers[q], markersize=7, linewidth=2.0,
                          markeredgecolor=SURFACE, label=QUANTITIES[q]) for q in QUANTITIES]
        row_axes[0].legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY,
                           loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=3)

    descriptive = any(t["n"] < MIN_TERCILE_RUNS for res in results.values()
                      for t in res["rows"][0]["terciles"])

    fig.suptitle("Extra 2: when does correction help? (RQ2.3, K-small x 3 seeds, raw second candidates)",
                 x=0.06, ha="left", fontsize=12, color=TEXT_PRIMARY, fontweight="bold")
    fig.text(0.06, -0.12 if len(models) == 1 else -0.05,
             "Tercile means with 95% bootstrap CIs (whole examples resampled). Step1X: only runs where the "
             "reflection triggered a retry. Downward trends in the left panel are expected from regression to "
             "the mean; the right panel cancels it."
             + (" Some terciles have fewer than 10 runs: descriptive only." if descriptive else ""),
             fontsize=7.5, color=TEXT_SECONDARY, ha="left", va="top", wrap=True)

    save(fig, out_dir, "extra2_correction_gain")


def main():

    parser = argparse.ArgumentParser(description="Extra 2: correction gain vs first-candidate quality.")
    parser.add_argument("--model", action="append", default=None,
                        help="NAME=per-example CSV; repeat for each model.")
    parser.add_argument("--out_dir", default="outputs/rq2_3/analysis")
    args = parser.parse_args()

    models = args.model or ["Step1X=outputs/rq2_3/analysis/rq2_3_per_example.csv"]
    results = {}

    for spec in models:
        name, path = spec.split("=", 1)
        runs, skipped = load_runs(path)
        edges = tercile_edges([r["first"] for r in runs])

        results[name] = {
            "rows": summarize(runs, edges),
            "tercile_labels": tercile_labels(runs, edges),
            "n_runs": len(runs),
            "n_examples": len({r["example_uid"] for r in runs}),
            "skipped": skipped,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_outputs(results, out_dir)
    figure(results, out_dir / "figures")

    for name, res in results.items():
        print(f"{name}: {res['n_runs']} runs over {res['n_examples']} examples ({res['skipped']} without a retry)")

        for row in res["rows"]:
            print(f"  {row['quantity']:<22} mean {row['mean']:+.2f} [{row['mean_ci'][0]:+.2f}, {row['mean_ci'][1]:+.2f}] "
                  f"| rho {row['rho']:+.2f} [{row['rho_ci'][0]:+.2f}, {row['rho_ci'][1]:+.2f}] | terciles "
                  + ", ".join(f"{t['mean']:+.1f} (n={t['n']})" for t in row["terciles"]))

    print(f"Wrote {out_dir / 'extra2_correction_gain.md'}")


if __name__ == "__main__":
    main()

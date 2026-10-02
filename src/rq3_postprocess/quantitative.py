"""Supplementary, example-level figures from the unchanged NSCC analysis.

The primary contrast intervals are copied from the archived report. Supplementary
intervals resample examples after taking each example's three-repeat timing
median; timing repetitions and score dimensions are never independent samples.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from .common import bootstrap_mean, write_csv


COLORS = dict(zip("ABCDEF", ("#777777", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")))
LABELS = {
    "A": "A: thinking off, RegionE off", "B": "B: thinking on, RegionE off",
    "C": "C: thinking off, RegionE on", "D": "D: thinking on, RegionE on",
    "E": "E: D without RIKV", "F": "F: D without AVD",
}
COMPONENTS = ("consistency_score", "instruction_score", "quality_score", "knowledge_score")
MAIN_CONTRASTS = (
    ("G0_B_minus_A", "Thinking gain, RegionE off: B − A"),
    ("G1_D_minus_C", "Thinking gain, RegionE on: D − C"),
    ("interaction_D_minus_C_minus_B_plus_A", "Interaction: (D − C) − (B − A)"),
    ("thinking_on_acceleration_D_minus_B", "Acceleration, thinking on: D − B"),
    ("thinking_off_acceleration_C_minus_A", "Acceleration, thinking off: C − A"),
)
ABLATION_CONTRASTS = (
    ("full_D_minus_vanilla_B", "Full RegionE: D − B"),
    ("without_RIKV_E_minus_full_D", "Remove RIKV: E − D"),
    ("without_AVD_F_minus_full_D", "Remove AVD: F − D"),
    ("without_RIKV_E_minus_vanilla_B", "Without RIKV vs baseline: E − B"),
    ("without_AVD_F_minus_vanilla_B", "Without AVD vs baseline: F − B"),
)


def _ci(data, values):
    stats = data.statistics
    return bootstrap_mean(values, resamples=stats["bootstrap_resamples"],
                          confidence=stats["confidence"], seed=stats["bootstrap_seed"])


def _errors(summary):
    # Numerical noise can put a percentile endpoint infinitesimally past the mean.
    return np.maximum(0, [[summary["mean"] - summary["ci_low"]],
                          [summary["ci_high"] - summary["mean"]]])


def _finish(fig, folder, name, note="", *, bottom=None):
    if note:
        fig.text(.01, .012, note, ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, bottom if bottom is not None else (.075 if note else 0), 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _condition_summary(data, ids, conditions, cohort):
    rows = []
    for condition in conditions:
        metrics = {"Q": [data.scores[(uid, condition)]["Q"] for uid in ids]}
        for field in ("diffusion_seconds", "charged_total_seconds", "measured_replay_seconds", "peak_vram_bytes"):
            metrics[field] = [data.timing[(uid, condition)][field] for uid in ids]
        for metric, values in metrics.items():
            rows.append({"cohort": cohort, "condition": condition, "metric": metric, **_ci(data, values)})
    return rows


def _forest(data, folder, tables):
    report = data.report
    records = [{"contrast": name, "label": label, "source": "canonical_NSCC_report", **report["contrasts"][name]}
               for name, label in MAIN_CONTRASTS]
    write_csv(tables / "main_contrasts.csv", records)
    fig, ax = plt.subplots(figsize=(10, 5.1))
    delta = report["retention"]["equivalence_margin"]
    # Only the interaction has a two-sided equivalence margin; D-B has a loss threshold.
    ax.fill_betweenx([1.66, 2.34], -delta, delta, color="#009E73", alpha=.12)
    ax.plot([-delta, delta], [2, 2], "|", color="#009E73", markersize=26)
    ax.plot([-delta, -delta], [2.66, 3.34], color="#D55E00", linestyle="--", linewidth=1.5)
    for i, record in enumerate(records):
        ax.errorbar(record["mean"], i, xerr=_errors(record), fmt="o", capsize=4,
                    color=COLORS["D"] if i in (2, 3) else COLORS["B"])
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(yticks=range(len(records)), yticklabels=[r["label"] for r in records],
           xlabel="Paired mean difference in Q (points)", title="RQ3 main effects: 40 matched examples")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    _finish(fig, folder, "main_effects", "Points and 95% intervals: frozen NSCC report. Green band: interaction ±5; dashed mark: D − B loss threshold −5.\nThe threshold is not a benefit criterion for B − A or D − C. One generation seed; intervals resample examples.")


def _interaction(data, folder, tables, conditions):
    lookup = {(r["condition"], r["metric"]): r for r in conditions if r["cohort"] == "main40"}
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for pair, color, label, offset in (("AB", COLORS["B"], "RegionE off: A → B", -.025),
                                       ("CD", COLORS["D"], "RegionE on: C → D", .025)):
        ys = [lookup[(c, "Q")]["mean"] for c in pair]
        ax.errorbar(np.arange(2) + offset, ys,
                    yerr=np.column_stack([_errors(lookup[(c, "Q")])[:, 0] for c in pair]),
                    marker="o", capsize=4, color=color, label=label)
        for x, y, c in zip(np.arange(2) + offset, ys, pair):
            offset_y = -15 if c in "AD" else 6
            ax.annotate(f"{c}: {y:.2f}", (x, y), xytext=(8, offset_y), textcoords="offset points", color=color)
    ax.set(xlim=(-.3, 1.4), xticks=[0, 1], xticklabels=["Thinking off", "Thinking on"],
           ylabel="Mean Q (0–100; expanded vertical scale)", title="Thinking × RegionE interaction")
    ax.grid(alpha=.2)
    ax.legend(loc="best")
    _finish(fig, folder, "interaction", "95% intervals for condition means resample 40 examples. The paired interaction estimate is in main_effects.\nCondition-mean interval overlap is not a test of the interaction.")


def _quality_latency(data, folder, conditions):
    lookup = {(r["condition"], r["metric"]): r for r in conditions if r["cohort"] == "main40"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    for ax, metric, title in zip(axes, ("diffusion_seconds", "charged_total_seconds"),
                                ("Generator", "Charged total (reconstructed)")):
        for c in "ABCD":
            x, y = lookup[(c, metric)], lookup[(c, "Q")]
            ax.errorbar(x["mean"], y["mean"], xerr=_errors(x), yerr=_errors(y),
                        fmt="o", color=COLORS[c], capsize=3, label=LABELS[c])
            ax.annotate(c, (x["mean"], y["mean"]), xytext=(7, 2), textcoords="offset points", color=COLORS[c])
        ax.set(xlabel="Mean seconds per example", title=title)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean Q (0–100; expanded vertical scale)")
    _finish(fig, folder, "quality_latency_intervals", "Marginal 95% bootstrap intervals; 40 examples. Timing: median of three measured repeats per example.\nCharged total includes cached thinking/offload costs and excludes model loading, warmup and saving artifacts.")


def _speedup_rows(data, ids, conditions, baseline, cohort):
    rows = []
    for c in conditions:
        for field in ("diffusion_seconds", "charged_total_seconds"):
            base = np.array([data.timing[(uid, baseline)][field] for uid in ids])
            treatment = np.array([data.timing[(uid, c)][field] for uid in ids])
            rows.append({"cohort": cohort, "condition": c, "baseline": baseline, "metric": field,
                         "estimator": "mean_within_example_baseline_over_condition", **_ci(data, base / treatment),
                         "ratio_of_means": float(base.mean() / treatment.mean())})
    return rows


def _ablations(data, folder, tables, conditions, speedups):
    records = [{"contrast": name, "label": label, "source": "canonical_NSCC_report", **data.report["ablations"]["contrasts"][name]}
               for name, label in ABLATION_CONTRASTS]
    write_csv(tables / "ablation_contrasts.csv", records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
    for i, row in enumerate(records):
        axes[0].errorbar(row["mean"], i, xerr=_errors(row), fmt="o", color=COLORS["E"] if "RIKV" in row["contrast"] else COLORS["D"], capsize=4)
    axes[0].axvline(0, color="black", linewidth=.8)
    axes[0].set(yticks=range(len(records)), yticklabels=[r["label"] for r in records],
                xlabel="Paired mean difference in Q (points)", title="Component-removal quality effects")
    axes[0].invert_yaxis()
    for j, (field, label, marker) in enumerate((("diffusion_seconds", "Generator", "o"),
                                             ("charged_total_seconds", "Charged total", "s"))):
        for i, c in enumerate("DEF"):
            summary = next(r for r in speedups if r["cohort"] == "ablation20" and r["metric"] == field and r["condition"] == c)
            axes[1].errorbar(summary["mean"], i + (j - .5) * .17, xerr=_errors(summary),
                            fmt=marker, color=COLORS[c], capsize=4, label=label if i == 0 else None,
                            markerfacecolor=COLORS[c] if j == 0 else "white")
    axes[1].axvline(1, color="black", linewidth=.8)
    axes[1].set(yticks=range(3), yticklabels=["D: full RegionE", "E: without RIKV", "F: without AVD"],
                xlabel="Mean paired speedup vs B (×)", title="Efficiency on the same 20 examples")
    axes[1].invert_yaxis()
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
    _finish(fig, folder, "ablation_effects_efficiency", "20 predetermined examples in every comparison. Quality intervals: canonical report; speedup intervals: example bootstrap.\nEqual ratings give a degenerate empirical interval; they do not establish identical images or zero population variability.")
    # Paired trajectories retain discrete ratings and expose the high tie rate.
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for uid in data.subset:
        ax.plot(range(4), [data.scores[(uid, c)]["Q"] for c in "BDEF"], color="#999999", alpha=.28, linewidth=.8)
    for i, c in enumerate("BDEF"):
        row = next(r for r in conditions if r["cohort"] == "ablation20" and r["condition"] == c and r["metric"] == "Q")
        ax.errorbar(i, row["mean"], yerr=_errors(row), fmt="o", color=COLORS[c], capsize=5, markersize=7)
    ax.set(xticks=range(4), xticklabels=["B: baseline", "D: full RegionE", "E: without RIKV", "F: without AVD"],
           ylabel="Q (0–100)", ylim=(-3, 103), title="Paired component ablations: all 20 examples")
    ax.grid(axis="y", alpha=.2)
    _finish(fig, folder, "ablation_paired_quality", "Gray lines connect the same example; colored points are condition means with 95% example-bootstrap intervals.")


def _per_example(data, tables):
    rows = []
    for uid in data.uids:
        q = {c: data.scores[(uid, c)]["Q"] if (uid, c) in data.scores else None for c in "ABCDEF"}
        meta = data.quality[(uid, "A")]
        rows.append({"example_uid": uid, "category": meta["category"], "knowledge_type": meta["knowledge_type"],
                     "in_ablation_subset": uid in data.subset, **{f"Q_{c}": value for c, value in q.items()},
                     "B_minus_A": q["B"]-q["A"], "D_minus_C": q["D"]-q["C"],
                     "interaction": (q["D"]-q["C"])-(q["B"]-q["A"]),
                     "D_minus_B": q["D"]-q["B"], "C_minus_A": q["C"]-q["A"],
                     "E_minus_D": q["E"]-q["D"] if q["E"] is not None else None,
                     "F_minus_D": q["F"]-q["D"] if q["F"] is not None else None,
                     "diffusion_speedup_B_over_D": data.timing[(uid, "B")]["diffusion_seconds"] / data.timing[(uid, "D")]["diffusion_seconds"],
                     "charged_speedup_B_over_D": data.timing[(uid, "B")]["charged_total_seconds"] / data.timing[(uid, "D")]["charged_total_seconds"]})
    write_csv(tables / "per_example_effects.csv", rows)
    return rows


def _scatter(data, folder, examples):
    groups = sorted({r["knowledge_type"] for r in examples})
    palette = dict(zip(groups, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, metric, title in zip(axes, ("diffusion_speedup_B_over_D", "charged_speedup_B_over_D"),
                                ("Generator", "Charged total (reconstructed)")):
        for group in groups:
            rows = [r for r in examples if r["knowledge_type"] == group]
            ax.scatter([r[metric] for r in rows], [r["D_minus_B"] for r in rows],
                       s=45, alpha=.75, edgecolors="white", linewidth=.5, color=palette[group], label=group.capitalize())
        ax.axhline(0, color="black", linewidth=.8)
        ax.axhline(-data.statistics["equivalence_margin"], color="#D55E00", linestyle="--", linewidth=.8)
        ax.axvline(1, color="#999999", linewidth=.8)
        ax.set(xlabel="Per-example B / D speedup (×)", title=title)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Per-example Q(D) − Q(B), points")
    axes[1].legend(title="Knowledge type", loc="best")
    _finish(fig, folder, "quality_change_vs_speedup", "40 examples; exact ratings, no jitter. Overlapping points are possible. Dashed line: −5 points for descriptive context.\nThe retention rule applies to the mean effect interval, not an individual example's threshold crossing.")


def _components(data, folder, tables):
    all_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True)
    names = ["Consistency", "Instruction following", "Visual quality", "Knowledge (applicable only)"]
    for ax, ids, conditions, cohort, title in ((axes[0], data.uids, "ABCD", "main40", "Main experiment"),
                                              (axes[1], data.subset, "BDEF", "ablation20", "Component ablations")):
        for j, c in enumerate(conditions):
            for i, metric in enumerate(COMPONENTS):
                raw = [data.scores[(uid, c)][metric] for uid in ids if data.scores[(uid, c)].get(metric) is not None]
                if not raw:
                    continue
                row = {"cohort": cohort, "condition": c, "metric": metric,
                       "normalization": "25*(raw_score-1)", **_ci(data, 25 * (np.array(raw)-1))}
                all_rows.append(row)
                ax.errorbar(row["mean"], i + (j - 1.5) * .13, xerr=_errors(row), fmt="o", color=COLORS[c],
                            capsize=2, label=c if i == 0 else None)
        counts = [sum(data.scores[(uid, conditions[0])].get(m) is not None for uid in ids) for m in COMPONENTS]
        ax.set(yticks=range(4), yticklabels=[f"{n}\n(n={count})" for n, count in zip(names, counts)],
               xlim=(-2, 102), xlabel="Normalized dimension score (0–100)", title=title)
        ax.invert_yaxis()
        ax.legend(ncol=4)
        ax.grid(axis="x", alpha=.2)
    write_csv(tables / "component_means.csv", all_rows)
    _finish(fig, folder, "score_dimensions", "95% example-bootstrap intervals; n is the number of applicable examples per condition.\nQ is an example-level mean across applicable dimensions; it is not the unweighted mean of these plotted dimension means.")
    return all_rows


def _ties(data, tables):
    comparisons = [(data.uids, "main40", a, b) for a, b in (("B", "A"), ("D", "C"), ("D", "B"), ("C", "A"))]
    comparisons += [(data.subset, "ablation20", a, b) for a, b in (("D", "B"), ("E", "D"), ("F", "D"), ("E", "B"), ("F", "B"))]
    rows = []
    for ids, cohort, treatment, baseline in comparisons:
        for metric in ("Q",) + COMPONENTS + ("all_applicable_components",):
            if metric == "all_applicable_components":
                n_ties = sum(all(data.scores[(uid, treatment)].get(m) == data.scores[(uid, baseline)].get(m) for m in COMPONENTS) for uid in ids)
                rows.append({"cohort": cohort, "contrast": f"{treatment}_minus_{baseline}", "metric": metric,
                             "n_examples": len(ids), "ties": n_ties, "tie_fraction": n_ties/len(ids), "higher": None, "lower": None})
                continue
            differences = [data.scores[(uid, treatment)][metric]-data.scores[(uid, baseline)][metric] for uid in ids
                           if data.scores[(uid, treatment)].get(metric) is not None and data.scores[(uid, baseline)].get(metric) is not None]
            values = np.asarray(differences)
            ties = int(np.count_nonzero(np.isclose(values, 0, atol=1e-9, rtol=0)))
            rows.append({"cohort": cohort, "contrast": f"{treatment}_minus_{baseline}", "metric": metric,
                         "n_examples": len(values), "ties": ties, "tie_fraction": ties/len(values) if len(values) else None,
                         "higher": int(np.count_nonzero(values > 1e-9)), "lower": int(np.count_nonzero(values < -1e-9))})
    write_csv(tables / "rating_ties.csv", rows)
    return rows


def _step_activity(data, folder, tables):
    cohorts = [("main40", "D", data.uids)] + [("ablation20", c, data.subset) for c in "DEF"]
    per_step, summaries, matrices = [], [], {}
    fields = ("selected_fraction", "rho_t", "avd_cache_reused", "full_image_generator_calls",
              "reduced_token_generator_calls", "rikv_context_reuses", "rikv_cache_writes", "refresh_event")
    expected_steps = data.report["generation"]["num_inference_steps"]
    for cohort, c, ids in cohorts:
        grouped = [[] for _ in range(expected_steps)]
        matrix = []
        for uid in ids:
            steps = data.quality[(uid, c)]["regione"]["steps"]
            if [s["step"] for s in steps] != list(range(expected_steps)):
                raise ValueError(f"Incomplete region steps for {uid}/{c}")
            events = []
            for step in steps:
                record = {"cohort": cohort, "condition": c, "example_uid": uid, "step_zero_based": step["step"],
                          **{field: float(step[field]) for field in fields},
                          "generator_calls": len(step["generator_calls"])}
                per_step.append(record)
                grouped[step["step"]].append(record)
                # Full/reduced means an actual generator invocation; cache steps execute none.
                if step["full_image_generator_calls"]:
                    events.append(2)
                elif step["reduced_token_generator_calls"]:
                    events.append(1)
                elif step["avd_cache_reused"]:
                    events.append(0)
                else:
                    raise ValueError(f"Unclassified step {uid}/{c}/{step['step']}")
            matrix.append(events)
        matrices[(cohort, c)] = np.array(matrix)
        for step, records in enumerate(grouped):
            for metric in fields + ("generator_calls",):
                summaries.append({"cohort": cohort, "condition": c, "step_zero_based": step,
                                  "metric": metric, "mean": float(np.mean([r[metric] for r in records])), "n_examples": len(ids)})
    write_csv(tables / "region_step_records.csv", per_step)
    write_csv(tables / "region_step_means.csv", summaries)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    plots = (("selected_fraction", "Selected image-token fraction"), ("rho_t", "Active-token fraction (ρ)"),
             ("generator_calls", "Actual generator calls"), ("avd_cache_reused", "Fraction of examples reusing AVD"),
             ("rikv_context_reuses", "RIKV attention-context reuse calls"), ("refresh_event", "Fraction of examples refreshing cache"))
    for ax, (metric, title) in zip(axes.flat, plots):
        for cohort, c, ids in cohorts:
            selected = [r for r in summaries if r["cohort"] == cohort and r["condition"] == c and r["metric"] == metric]
            ax.plot([r["step_zero_based"] for r in selected], [r["mean"] for r in selected],
                    color="#555555" if cohort == "main40" else COLORS[c],
                    linestyle="--" if cohort == "main40" else "-", linewidth=1.5,
                    label=f"{c}, all 40" if cohort == "main40" else f"{c}, shared 20")
        ax.set(title=title, xlabel="Denoising step (zero based)", xlim=(0, expected_steps-1))
        if metric in ("selected_fraction", "rho_t", "avd_cache_reused", "refresh_event"):
            ax.set_ylim(-.03, 1.04)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    _finish(fig, folder, "region_step_activity", "Repeat-zero telemetry. D/E/F use the same 20 examples; dashed D uses all 40. Coincident series overlap.\nρ records active tokens even on AVD cache-only steps; it is not the fraction of computation executed.")
    cmap = ListedColormap(["#f0f0f0", "#56B4E9", "#0072B2"])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(matrices[("main40", "D")], aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set(yticks=range(len(data.uids)), yticklabels=data.uids, xlabel="Denoising step (zero based)",
           title="D: generator/cache execution across all 40 examples")
    ax.tick_params(axis="y", labelsize=7)
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=.022).ax.set_yticklabels(["AVD cache", "Reduced tokens", "Full image"])
    _finish(fig, folder, "region_execution_main", "Repeat-zero telemetry. Rows retain the frozen example order; colors represent actual generator execution.")
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    for ax, c in zip(axes, "DEF"):
        ax.imshow(matrices[("ablation20", c)], aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set(title=LABELS[c], xlabel="Denoising step (zero based)")
    axes[0].set(yticks=range(len(data.subset)), yticklabels=data.subset)
    axes[0].tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=cmap(i), label=name) for i, name in enumerate(("AVD cache", "Reduced tokens", "Full image"))],
               loc="lower center", bbox_to_anchor=(.62, .065), ncol=3, frameon=False)
    _finish(fig, folder, "region_execution_ablations", "Same 20 examples and row order in D/E/F; repeat-zero telemetry. F disables AVD, so its intervening steps execute the generator.", bottom=.13)
    return {"region_step_records": len(per_step), "region_step_summary_rows": len(summaries)}


def _resources(data, folder, tables):
    rows = []
    fields = ("thinking_seconds", "thinking_offload_seconds", "diffusion_seconds", "offload_seconds", "measured_replay_seconds", "charged_total_seconds", "peak_vram_bytes")
    for cohort, ids, conditions in (("main40", data.uids, "ABCD"), ("ablation20", data.subset, "BDEF")):
        for c in conditions:
            for metric in fields:
                values = [data.timing[(uid, c)].get(metric) for uid in ids]
                if any(v is None for v in values):
                    continue
                rows.append({"cohort": cohort, "condition": c, "metric": metric, **_ci(data, values)})
    write_csv(tables / "resources_and_stages.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, cohort, conditions, title in ((axes[0], "main40", "ABCD", "Main: 40 examples"),
                                         (axes[1], "ablation20", "BDEF", "Ablations: same 20 examples")):
        for i, c in enumerate(conditions):
            row = next(r for r in rows if r["cohort"] == cohort and r["condition"] == c and r["metric"] == "peak_vram_bytes")
            ax.bar(i, row["mean"]/2**30, color=COLORS[c], width=.6)
            ax.errorbar(i, row["mean"]/2**30, yerr=_errors(row)/2**30, fmt="none", color="black", capsize=3)
        ax.set(xticks=range(4), xticklabels=list(conditions), title=title)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Mean peak PyTorch allocated VRAM (GiB)")
    _finish(fig, folder, "peak_memory", "Per example: maximum peak across three timing repeats; bars: example means, intervals: example bootstrap.\nAllocated VRAM excludes other processes and is not total device memory usage.")
    display = (("thinking_seconds", "Thinking"), ("thinking_offload_seconds", "Thinking offload"),
               ("diffusion_seconds", "Generator"), ("offload_seconds", "Pipeline offload"),
               ("measured_replay_seconds", "Measured replay"), ("charged_total_seconds", "Charged total"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (metric, title) in zip(axes.flat, display):
        for i, c in enumerate("ABCD"):
            row = next(r for r in rows if r["cohort"] == "main40" and r["condition"] == c and r["metric"] == metric)
            ax.bar(i, row["mean"], color=COLORS[c], width=.6)
            ax.errorbar(i, row["mean"], yerr=_errors(row), fmt="none", color="black", capsize=2)
        ax.set(xticks=range(4), xticklabels=list("ABCD"), title=title, ylabel="Mean seconds")
        ax.grid(axis="y", alpha=.2)
    _finish(fig, folder, "timing_stages", "40 examples; timing medians then example means and 95% intervals. Components are displayed separately because they overlap.\nThinking and its offload are reused from a common cached thought; charged total is a reconstructed metric.")
    return rows


def _exploratory_groups(data, folder, tables, examples):
    rows = []
    for group_field in ("knowledge_type", "category"):
        for group in sorted({r[group_field] for r in examples}):
            group_rows = [r for r in examples if r[group_field] == group]
            for metric in ("B_minus_A", "D_minus_C", "D_minus_B", "interaction", "charged_speedup_B_over_D"):
                values = [r[metric] for r in group_rows]
                rows.append({"grouping": group_field, "group": group, "metric": metric, "n_examples": len(values),
                             "mean": float(np.mean(values)), "minimum": float(np.min(values)), "maximum": float(np.max(values))})
    write_csv(tables / "exploratory_groups.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    groups = sorted({r["knowledge_type"] for r in examples})
    for ax, metric, title in zip(axes, ("B_minus_A", "D_minus_B"), ("Thinking gain without RegionE: B − A", "Acceleration with thinking: D − B")):
        for i, group in enumerate(groups):
            values = [r[metric] for r in examples if r["knowledge_type"] == group]
            ax.scatter(np.full(len(values), i), values, alpha=.35, color=COLORS["B"], s=40)
            ax.scatter(i, np.mean(values), color="black", marker="D", s=45)
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.set(xticks=range(len(groups)), xticklabels=[f"{g.capitalize()}\n(n={sum(r['knowledge_type']==g for r in examples)})" for g in groups], title=title)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Per-example Q difference (points)")
    _finish(fig, folder, "knowledge_type_exploratory", "Descriptive only: individual examples and group means (black diamonds). No subgroup significance tests.\nExact values overlap; small, unequal groups do not support stable category-level conclusions.")
    return rows


def run(data, output: Path) -> dict:
    """Write report-ready figures and machine-readable supporting tables."""
    output = Path(output)
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}):
        conditions = _condition_summary(data, data.uids, "ABCD", "main40") + _condition_summary(data, data.subset, "BDEF", "ablation20")
        write_csv(tables / "condition_means_intervals.csv", conditions)
        speedups = (_speedup_rows(data, data.uids, "D", "B", "main40")
                    + _speedup_rows(data, data.uids, "C", "A", "main40")
                    + _speedup_rows(data, data.subset, "DEF", "B", "ablation20"))
        write_csv(tables / "paired_speedups.csv", speedups)
        _forest(data, figures, tables)
        _interaction(data, figures, tables, conditions)
        _quality_latency(data, figures, conditions)
        _ablations(data, figures, tables, conditions, speedups)
        examples = _per_example(data, tables)
        _scatter(data, figures, examples)
        components = _components(data, figures, tables)
        ties = _ties(data, tables)
        activity = _step_activity(data, figures, tables)
        resources = _resources(data, figures, tables)
        groups = _exploratory_groups(data, figures, tables, examples)
    summary = {
        "primary_effect_source": "Unchanged canonical NSCC report.json",
        "supplementary_interval_method": "Percentile bootstrap across examples after within-example timing medians",
        "bootstrap_resamples": data.statistics["bootstrap_resamples"],
        "confidence": data.statistics["confidence"], "bootstrap_seed": data.statistics["bootstrap_seed"],
        "condition_summaries": conditions, "speedups": speedups, "rating_ties": ties,
        "component_means": components, "resource_summaries": resources, "exploratory_groups": groups,
        "main_examples": len(data.uids), "ablation_examples": len(data.subset), **activity,
        "notes": ["Timing uses measured NSCC records; local plotting times are not inference performance.",
                  "Charged total is reconstructed and excludes initialization, model loading and saving.",
                  "Intervals describe example variability at one generation seed, not seed robustness or judge uncertainty.",
                  "Equal ratings and zero-width empirical intervals do not establish pixel identity or zero population variability.",
                  "Main quality and ablation quality use 40 and 20 paired examples respectively, not 800 independent observations.",
                  "Subgroups are descriptive and were explored after observing the main results."]}
    (output / "quantitative_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary

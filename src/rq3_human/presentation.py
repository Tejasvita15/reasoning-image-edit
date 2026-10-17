"""Render the supplementary human-validation report without changing ratings."""
from __future__ import annotations

from collections import Counter
from html import escape
import math
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RATERS = ("reviewer_1", "reviewer_2", "chatgpt", "human_mean")
LABELS = {
    "reviewer_1": "Reviewer 1", "reviewer_2": "Reviewer 2",
    "chatgpt": "ChatGPT", "human_mean": "Human mean (secondary)",
}
COLORS = dict(zip(RATERS, ("#0072B2", "#D55E00", "#222222", "#009E73")))
DIMENSIONS = (
    ("consistency_score", "Consistency"), ("instruction_score", "Instruction"),
    ("quality_score", "Visual quality"), ("knowledge_score", "Knowledge"),
)
CONTRASTS = (
    ("G0_B_minus_A", "Thinking, RegionE off: B − A"),
    ("G1_D_minus_C", "Thinking, RegionE on: D − C"),
    ("interaction_D_minus_C_minus_B_plus_A", "Interaction: (D − C) − (B − A)"),
    ("thinking_on_acceleration_D_minus_B", "Acceleration, thinking on: D − B"),
    ("thinking_off_acceleration_C_minus_A", "Acceleration, thinking off: C − A"),
)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _number(value, digits=2):
    if not _finite(value):
        return "undefined"
    if abs(value) < .5 * 10 ** -digits:
        value = 0
    return f"{value:.{digits}f}"


def _interval(row, digits=2):
    return (f"{_number(row.get('estimate'), digits)} "
            f"[{_number(row.get('ci_low'), digits)}, {_number(row.get('ci_high'), digits)}]")


def _pair(row):
    return f"{LABELS.get(row['first'], row['first'])} / {LABELS.get(row['second'], row['second'])}"


def _find_agreement(report, first, second, metric, dimension="Q"):
    return next((r for r in report["agreements"] if r["first"] == first
                 and r["second"] == second and r["metric"] == metric
                 and r["dimension"] == dimension), None)


def _point_interval(ax, row, y, **kwargs):
    estimate = row.get("estimate")
    if not _finite(estimate):
        return
    low, high = row.get("ci_low"), row.get("ci_high")
    color = kwargs.get("color", "black")
    if _finite(low) and _finite(high):
        ax.plot([low, high], [y, y], color=color, linewidth=1.4, zorder=2)
        ax.plot([low, high], [y, y], linestyle="none", marker="|", color=color,
                markersize=6, zorder=2)
    ax.plot(estimate, y, linestyle="none", markersize=5, zorder=3, **kwargs)


def _save(fig, folder, name, note):
    fig.text(.012, .012, note, ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _scatter(folder, report, joined_rows):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.1), sharex=True, sharey=True)
    pairs = (("reviewer_1", "chatgpt"), ("reviewer_2", "chatgpt"),
             ("reviewer_1", "reviewer_2"))
    for ax, (first, second) in zip(axes, pairs):
        points = Counter((round(float(r[f"{first}_Q"]), 10), round(float(r[f"{second}_Q"]), 10))
                         for r in joined_rows
                         if _finite(r.get(f"{first}_Q")) and _finite(r.get(f"{second}_Q")))
        for (x, y), count in points.items():
            ax.scatter(x, y, s=18 * count, color=COLORS[first], alpha=.6,
                       edgecolors="white", linewidths=.4)
        ax.plot([0, 100], [0, 100], "--", color="#666666", linewidth=1)
        rho = _find_agreement(report, first, second, "spearman_rho")
        icc = _find_agreement(report, first, second, "icc_a1")
        details = []
        if rho:
            details.append(f"ρ = {_number(rho['estimate'])}")
        if icc:
            details.append(f"ICC(A,1) = {_number(icc['estimate'])}")
        ax.text(.03, .96, "\n".join(details), transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        ax.set(xlim=(-4, 104), ylim=(-4, 104), xlabel=f"{LABELS[first]} Q",
               ylabel=f"{LABELS[second]} Q", title=f"{LABELS[first]} vs {LABELS[second]}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.18)
    _save(fig, folder, "agreement_q", "Each point represents one or more of the same 120 outputs; marker area is proportional to multiplicity (no jitter).\nDashed line: exact agreement. Scores are quantized; 120 outputs are clustered within 30 examples. Correlation is not absolute agreement.")


def _kappa(folder, report):
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for y, (dimension, _) in enumerate(DIMENSIONS):
        row = _find_agreement(report, "reviewer_1", "reviewer_2", "quadratic_kappa", dimension)
        if row:
            _point_interval(ax, row, y, marker="o", color=COLORS["reviewer_1"])
            ax.text(1.02, y, _interval(row, 3), va="center", fontsize=9,
                    transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#666666", linewidth=.8)
    ax.axvline(1, color="#aaaaaa", linestyle=":", linewidth=.8)
    ax.set(yticks=range(4), yticklabels=[label for _, label in DIMENSIONS],
           xlabel="Quadratically weighted Cohen's κ", title="Agreement between the two human reviewers")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    fig.subplots_adjust(right=.70)
    _save(fig, folder, "human_human_kappa", "Points and 95% percentile intervals use example-cluster resampling. Knowledge: 64 outputs from 16 examples; other dimensions: 120 outputs / 30 examples.\nIntervals describe these two fixed reviewers. No qualitative agreement-category cutoffs are imposed.")


def _effects(folder, report):
    fig, ax = plt.subplots(figsize=(11, 7.2))
    records = {(r["rater"], r["contrast"]): r for r in report["effects"]}
    offsets = (-.27, -.09, .09, .27)
    markers = ("o", "s", "D", "^")
    for offset, marker, rater in zip(offsets, markers, RATERS):
        for index, (contrast, _) in enumerate(CONTRASTS):
            if (rater, contrast) in records:
                _point_interval(ax, records[(rater, contrast)], index + offset,
                                color=COLORS[rater], marker=marker)
        ax.plot([], [], color=COLORS[rater], marker=marker, linestyle="none", label=LABELS[rater])
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(yticks=range(len(CONTRASTS)), yticklabels=[label for _, label in CONTRASTS],
           xlabel="Paired mean difference in Q (points)",
           title="Condition effects on the same 30-example validation subset")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.18)
    ax.legend(loc="best", fontsize=9)
    _save(fig, folder, "condition_effects", "95% percentile intervals resample the same 30 example clusters, retaining all four conditions and all raters.\nHuman mean is a secondary average of two ratings for each output, not an additional independent reviewer. Interval overlap is not a test of rater differences.")


class _Document:
    """Write parallel Markdown and escaped, dependency-free HTML."""

    def __init__(self, title):
        self.markdown = [f"# {title}", ""]
        self.html = [f"<h1>{escape(title)}</h1>"]

    def heading(self, title):
        self.markdown.extend([f"## {title}", ""])
        self.html.append(f"<h2>{escape(title)}</h2>")

    def paragraph(self, text):
        self.markdown.extend([text, ""])
        self.html.append(f"<p>{escape(text)}</p>")

    def table(self, headers, rows):
        rows = [[str(value) for value in row] for row in rows]
        self.markdown.extend(["| " + " | ".join(headers) + " |",
                              "| " + " | ".join("---" for _ in headers) + " |"])
        self.markdown.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
        self.markdown.append("")
        self.html.append("<div class='table'><table><thead><tr>" +
                         "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" +
                         "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>" for row in rows) +
                         "</tbody></table></div>")

    def figure(self, name, caption):
        path = f"figures/{name}"
        self.markdown.extend([f"![{caption}]({path}.png)", "", f"{caption} [PDF]({path}.pdf).", ""])
        self.html.append(f"<figure><a href='{path}.pdf'><img src='{path}.png' alt='{escape(caption, quote=True)}'></a>"
                         f"<figcaption>{escape(caption)} <a href='{path}.pdf'>PDF</a>.</figcaption></figure>")

    def link(self, path, label):
        self.markdown.extend([f"[{label}]({path})", ""])
        self.html.append(f"<p><a href='{escape(path, quote=True)}'>{escape(label)}</a></p>")

    def write(self, output):
        (output / "RESULTS.md").write_text("\n".join(self.markdown), encoding="utf-8")
        style = ("body{font:16px/1.55 system-ui,sans-serif;max-width:1220px;margin:35px auto;padding:0 24px;color:#222}"
                 "h1,h2{line-height:1.2}h2{margin-top:2em}table{border-collapse:collapse;width:100%;font-size:14px}"
                 "td,th{border:1px solid #ddd;padding:7px 10px;text-align:left}th{background:#eef2f5}"
                 ".table{overflow-x:auto}figure{margin:28px 0}img{max-width:100%;height:auto}"
                 "figcaption{font-size:14px;color:#444}a{color:#125aaa}")
        (output / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'>"
                                            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                                            "<title>RQ3 human–judge validation</title>"
                                            f"<style>{style}</style><body>" + "\n".join(self.html) +
                                            "</body></html>", encoding="utf-8")


def render(output: Path, report: dict, joined_rows: list):
    """Create a report and three figure families from already validated statistics."""
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _scatter(figures, report, joined_rows)
    _kappa(figures, report)
    _effects(figures, report)
    doc = _Document("RQ3 supplementary human–judge validation")
    doc.paragraph("This analysis compares two human reviewers with the recorded ChatGPT judge on 120 identical outputs "
                  "from 30 examples (10 factual, 10 conceptual and 10 procedural), with conditions A–D for every example. "
                  "The four practice ratings from each reviewer are excluded. The subset was selected after generation "
                  "and the original analysis; this is supplementary validation, not a preregistered independent replication.")
    doc.paragraph("Q is the mean of 25 × (score − 1) over the applicable ordinal scoring dimensions. "
                  "Knowledge is applicable to 64 outputs from 16 examples. All other dimensions apply to all 120 outputs. "
                  "Human mean denotes the per-output average of the two human Q values and is a secondary summary.")
    doc.heading("Agreement and score differences")
    offsets = []
    for reviewer in ("reviewer_1", "reviewer_2"):
        difference = _find_agreement(report, reviewer, "chatgpt", "mean_difference")
        absolute = _find_agreement(report, reviewer, "chatgpt", "mean_absolute_difference")
        if difference and absolute:
            offsets.append(f"relative to {LABELS[reviewer]}, ChatGPT's mean Q difference was "
                           f"{_interval(difference)} points and its mean absolute difference was "
                           f"{_interval(absolute)} points")
    if offsets:
        summary = "; ".join(offsets)
        doc.paragraph(summary[0].upper() + summary[1:] + ". These are paired comparisons of identical outputs; "
                      "bracketed values are 95% confidence intervals.")
    kappa = [r for r in report["agreements"] if r["metric"] == "quadratic_kappa"
             and r["first"] == "reviewer_1" and r["second"] == "reviewer_2"]
    dimensions = dict(DIMENSIONS)
    doc.table(["Human–human dimension", "Outputs / examples", "Quadratic κ [95% CI]"],
              [[dimensions.get(r["dimension"], r["dimension"]), f"{r['n_outputs']} / {r['n_examples']}",
                _interval(r, 3)] for r in kappa])
    pairs = (("reviewer_1", "reviewer_2"), ("reviewer_1", "chatgpt"), ("reviewer_2", "chatgpt"))
    records = []
    for first, second in pairs:
        metrics = [_find_agreement(report, first, second, name) for name in
                   ("spearman_rho", "icc_a1", "mean_difference", "mean_absolute_difference")]
        if any(metrics):
            records.append([f"{LABELS[first]} / {LABELS[second]}"] +
                           [_interval(r, 3 if i < 2 else 2) if r else "not available"
                            for i, r in enumerate(metrics)])
    doc.table(["First / second rater", "Spearman ρ [95% CI]", "ICC(A,1) [95% CI]",
               "Mean second − first Q [95% CI]", "Mean absolute Q difference [95% CI]"], records)
    doc.paragraph("Spearman's ρ measures association in ranking, whereas ICC(A,1) measures absolute agreement under "
                  "a two-way mixed-effects, single-measurement model for each named fixed rater pair. A positive signed "
                  "difference means that the second rater assigned higher Q. Correlation alone does not establish agreement, "
                  "and agreement between two reviewers does not establish objective correctness.")
    doc.figure("agreement_q", "Figure 1. Q agreement for each human–judge pair and the human–human pair. "
               "Exact overlaps are shown by proportional marker area; no jitter changes the recorded values.")
    doc.figure("human_human_kappa", "Figure 2. Quadratically weighted human–human agreement by scoring dimension, "
               "with confidence intervals obtained by resampling examples.")
    doc.heading("Condition effects on the validation subset")
    doc.paragraph("A uses neither thinking nor RegionE; B enables thinking; C enables RegionE; D enables both. "
                  "All comparisons below use the same 30 examples and the existing generations. These estimates should "
                  "be compared across raters within this subset rather than directly equated with the original 40-example results.")
    conditions = {(r["rater"], r["condition"]): r for r in report["conditions"]}
    doc.table(["Rater", "A: mean Q [95% CI]", "B: mean Q [95% CI]", "C: mean Q [95% CI]", "D: mean Q [95% CI]"],
              [[LABELS[rater]] + [_interval(conditions[(rater, c)]) if (rater, c) in conditions else "not available"
                                  for c in "ABCD"] for rater in RATERS])
    effects = {(r["rater"], r["contrast"]): r for r in report["effects"]}
    doc.table(["Contrast"] + [LABELS[rater] + " [95% CI]" for rater in RATERS],
              [[label] + [_interval(effects[(rater, contrast)]) if (rater, contrast) in effects else "not available"
                          for rater in RATERS] for contrast, label in CONTRASTS])
    doc.figure("condition_effects", "Figure 3. Paired condition effects under each rater and the secondary human mean. "
               "All estimates use the same examples, with paired cluster confidence intervals.")
    doc.paragraph("B − A and D − C estimate the thinking effects without and with RegionE. Their difference estimates "
                  "the thinking × acceleration interaction; D − B and C − A estimate acceleration effects at fixed thinking settings. "
                  "An interval containing zero does not prove equivalence. Preserving a positive thinking benefit requires "
                  "first establishing that benefit under the relevant scoring scheme; an unchanged or small interaction is insufficient.")
    if report.get("effect_differences"):
        doc.paragraph("The accompanying effect-difference table directly estimates each human-scored contrast minus "
                      "the ChatGPT-scored contrast on these same examples. These comparisons are exploratory; overlapping "
                      "individual confidence intervals are not a formal test of the difference between raters.")
    doc.heading("Disagreement inspection")
    doc.paragraph("The following outputs have the largest absolute difference between ChatGPT Q and the secondary "
                  "human mean. Selection is post hoc and intended for error inspection, not for estimating the prevalence "
                  "of a failure type. A discrepancy identifies a case for examination, not which rater is correct.")
    ranked = [r for r in joined_rows if all(_finite(r.get(f"{name}_Q")) for name in RATERS)]
    ranked.sort(key=lambda r: (-abs(r["chatgpt_Q"] - r["human_mean_Q"]), r["example_uid"], r["condition"]))
    doc.table(["Example", "Condition", "Reviewer 1 Q", "Reviewer 2 Q", "ChatGPT Q", "ChatGPT − human mean"],
              [[r["example_uid"], r["condition"], _number(r["reviewer_1_Q"]), _number(r["reviewer_2_Q"]),
                _number(r["chatgpt_Q"]), _number(r["chatgpt_Q"] - r["human_mean_Q"])] for r in ranked[:10]])
    for r in ranked[:3]:
        for reviewer in ("reviewer_1", "reviewer_2"):
            blind_id = r.get(f"{reviewer}_blind_id")
            if blind_id:
                doc.link(f"../{reviewer}/cases/{quote(str(blind_id), safe='')}.html",
                         f"{r['example_uid']} / {r['condition']}: {LABELS[reviewer]} case page")
    doc.heading("Uncertainty, provenance and limits")
    doc.paragraph("Confidence intervals are 95% percentile intervals from 10,000 bootstrap replicates (seed 42), "
                  "resampling the 30 examples as clusters while preserving their four conditions and rater pairings. "
                  "The 120 outputs are not 120 independent examples. Undefined bootstrap statistics are excluded from "
                  "their corresponding percentile calculation; valid and invalid replicate counts are recorded in agreement.csv. "
                  "Undefined point estimates or intervals are reported explicitly rather than replaced with zero.")
    doc.paragraph("The agreement measures and example-cluster resampling follow the existing validation guide. "
                  "Their executable implementation was completed after the human submissions were available; it was "
                  "not preregistered or implemented without access to outcomes. This timing must be distinguished from "
                  "the earlier specification of the high-level validation methods.")
    doc.paragraph("The two named humans are fixed reviewers. Their results do not establish agreement for a population "
                  "of human raters, all 40 original examples, additional generation seeds, or conditions E/F. Human validation "
                  "of A–D does not directly validate the RIKV/AVD component ablations. The human mean does not double the "
                  "number of independent examples. Prior exposure and independence disclosures remain part of the reviewer metadata.")
    doc.paragraph("Original submitted CSV and metadata files are preserved. Two encoding-damaged optional notes in "
                  "Reviewer 2's submission are represented safely in derived artifacts; the raw submission is unchanged. "
                  "This analysis does not alter human ratings, original ChatGPT judgments, the frozen quality margin, "
                  "or generation records, and it requires no new model inference. Read the machine-readable checks and "
                  "provenance alongside these summaries.")
    doc.heading("Machine-readable results")
    for filename, label in (
        ("report.json", "Report, checks and provenance"),
        ("tables/agreement.csv", "Agreement statistics and bootstrap validity counts"),
        ("tables/condition_means.csv", "Condition means"),
        ("tables/paired_effects.csv", "Paired condition effects"),
        ("tables/effect_differences_vs_judge.csv", "Exploratory effect differences from ChatGPT"),
        ("tables/joined_scores.csv", "Joined output-level scores"),
        ("tables/disagreements.csv", "Ranked output-level disagreements"),
    ):
        doc.link(filename, label)
    doc.write(output)

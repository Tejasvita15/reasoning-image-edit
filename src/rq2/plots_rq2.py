# src/rq2/plots_rq2.py

"""
Report figures for RQ2.1 and RQ2.2.

    python -m src.rq2.plots_rq2

Writes PNG figures:
    outputs/rq2/analysis/figures/rq2_1_editing_curve        IF vs guidance
    outputs/rq2/analysis/figures/rq2_1_consistency_curve    VC vs guidance
    outputs/rq2_2/analysis/figures/rq2_2_quality_latency    Q vs end-to-end latency

Points are configuration means over the K-dev examples; error bars are 95%
bootstrap intervals of the mean (10,000 resamples over examples). Thinking off
and on differ by colour AND marker shape, with direct labels and a legend, so
identity never relies on colour alone.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.rq2.stats_rq2 import by_config, load_jsonl, values  # noqa: E402


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

# Validated categorical slots 1-2 (light surface), plus text and chrome tokens.
SERIES = {
    "off": {"color": "#2a78d6", "marker": "o", "label": "Thinking off"},
    "on": {"color": "#eb6834", "marker": "s", "label": "Thinking on"},
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#b8b7b0"
SURFACE = "#ffffff"


def mean_ci(values_):
    data = np.asarray(values_, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(data), size=(BOOTSTRAP_RESAMPLES, len(data)))
    low, high = np.percentile(data[idx].mean(axis=1), [2.5, 97.5])
    return float(data.mean()), float(low), float(high)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3, width=0.8)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def titled(fig, ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=24, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=TEXT_SECONDARY, va="bottom")


def legend_below(ax):
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("png",):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)

    plt.close(fig)
    print(f"Wrote {out_dir / name}.png")


# =============================================================
# RQ2.1 curves
# =============================================================

def guidance_curve(configs, metric, ylabel, title, out_dir, name):
    cfgs = [2.0, 6.0, 10.0]
    dodge = {"off": -0.12, "on": 0.12}

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    style_axes(ax)

    for thinking, spec in SERIES.items():
        stats = [mean_ci(list(values(configs[(thinking, cfg, 50)], metric).values())) for cfg in cfgs]
        xs = [cfg + dodge[thinking] for cfg in cfgs]
        means = [s[0] for s in stats]
        errors = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]

        ax.errorbar(xs, means, yerr=errors, color=spec["color"], linewidth=0, elinewidth=1.0, capsize=0, zorder=2)
        ax.plot(xs, means, color=spec["color"], linewidth=2.0, marker=spec["marker"], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=spec["label"], zorder=3)

        # Direct label at the line end, in text ink (identity comes from the mark).
        ax.annotate(spec["label"], (xs[-1], means[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT_PRIMARY)

    ax.set_xticks(cfgs, [f"{c:g}" for c in cfgs])
    ax.set_xlim(1, 12.6)
    ax.set_xlabel("Guidance (true_cfg_scale)")
    ax.set_ylabel(ylabel)
    legend_below(ax)

    titled(fig, ax, title,
           "Step1X-Edit-v1p2 · K-dev (n=90) · 50 steps · mean with 95% bootstrap CI")

    save(fig, out_dir, name)


# =============================================================
# RQ2.2 quality vs latency
# =============================================================

def quality_latency(configs, latency, thinking_tokens, frozen, out_dir):
    steps_list = [20, 28, 50]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6, linestyle="-")

    points = {}

    for thinking, spec in SERIES.items():
        xs, means, lows, highs = [], [], [], []

        for steps in steps_list:
            mean, low, high = mean_ci(list(values(configs[(thinking, 6.0, steps)], "Q").values()))
            points[(thinking, steps)] = (latency[(thinking, steps)], mean)
            xs.append(latency[(thinking, steps)])
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)

        ax.errorbar(xs, means, yerr=[lows, highs], color=spec["color"], linewidth=0, elinewidth=1.0, capsize=0, zorder=2)
        ax.plot(xs, means, color=spec["color"], linewidth=2.0, marker=spec["marker"], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=spec["label"], zorder=3)

    # One header per step-count column (both thinking modes sit at almost the
    # same latency), stating the thinking gain at matched latency. Placed above
    # every mark so nothing crosses a line or an error bar.
    for steps in steps_list:
        (x_off, y_off), (x_on, y_on) = points[("off", steps)], points[("on", steps)]
        ax.text((x_off + x_on) / 2, 67.3, f"{steps} steps", ha="center", va="bottom",
                fontsize=8.5, color=TEXT_PRIMARY, fontweight="bold")
        ax.text((x_off + x_on) / 2, 67.15, f"thinking {y_on - y_off:+.1f} Q\nfor +{x_on - x_off:.1f} s",
                ha="center", va="top", fontsize=7.5, color=TEXT_SECONDARY, linespacing=1.25)

    # Frozen configuration for the K-test run.
    fx, fy = points[(frozen["thinking"], frozen["num_inference_steps"])]
    ax.scatter([fx], [fy], s=240, facecolors="none", edgecolors=TEXT_PRIMARY, linewidths=1.0, zorder=4)
    ax.annotate("frozen for K-test", (fx, fy), xytext=(18, 16), textcoords="offset points",
                ha="left", va="center", fontsize=8, color=TEXT_PRIMARY,
                arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "linewidth": 0.8, "shrinkA": 0, "shrinkB": 9})

    ax.set_xlim(40, 122)
    ax.set_ylim(48, 68.6)
    ax.set_yticks(range(48, 66, 2))
    ax.set_xlabel("End-to-end latency per image (s, A100, model CPU offload)")
    ax.set_ylabel("Mean Q (0-100)")
    legend_below(ax)

    fig.text(0.125, -0.1,
             f"Generator calls = 2 x steps (true CFG): 40 / 56 / 100. Thinking = one rewrite pass "
             f"(median {thinking_tokens} tokens), charged to every thinking-on image. Latency: median of "
             "3 repeats per example, one A100 node.",
             fontsize=7.5, color=TEXT_SECONDARY, ha="left", va="top", wrap=True)

    titled(fig, ax, "RQ2.2 Quality vs end-to-end latency",
           "Step1X-Edit-v1p2 · K-dev (n=90) · guidance 6 · mean Q with 95% bootstrap CI")

    save(fig, out_dir, "rq2_2_quality_latency")


def main():

    parser = argparse.ArgumentParser(description="RQ2.1 / RQ2.2 report figures.")
    parser.add_argument("--scores_2_1", default="outputs/rq2/judge/chatgpt_main/scores_main.jsonl")
    parser.add_argument("--scores_2_2", default="outputs/rq2_2/judge/chatgpt_main/scores_main.jsonl")
    parser.add_argument("--latency", default="outputs/rq2_2/analysis/rq2_2_latency.csv")
    parser.add_argument("--frozen", default="outputs/rq2_2/analysis/rq2_2_frozen_config.json")
    parser.add_argument("--out_2_1", default="outputs/rq2/analysis/figures")
    parser.add_argument("--out_2_2", default="outputs/rq2_2/analysis/figures")
    args = parser.parse_args()

    configs = by_config(load_jsonl(args.scores_2_1) + load_jsonl(args.scores_2_2))

    with open(args.latency, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    latency = {(r["thinking"], int(r["steps"])): float(r["total_s"]) for r in rows}
    thinking_tokens = next(int(r["thinking_tokens"]) for r in rows if r["thinking"] == "on")
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))["frozen_configuration"]

    out_2_1 = Path(args.out_2_1)
    guidance_curve(configs, "IF", "Instruction following (judge score, 1-5)",
                   "RQ2.1 Editing: instruction following vs guidance", out_2_1, "rq2_1_editing_curve")
    guidance_curve(configs, "VC", "Visual consistency (judge score, 1-5)",
                   "RQ2.1 Preservation: visual consistency vs guidance", out_2_1, "rq2_1_consistency_curve")

    quality_latency(configs, latency, thinking_tokens, frozen, Path(args.out_2_2))


if __name__ == "__main__":
    main()

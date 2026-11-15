import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/rq2_unireason_analysis")


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


groups = defaultdict(list)
for f in sorted(glob.glob("logs/rq2_unireason/*.jsonl") + glob.glob("logs/rq2_unireason_2/*.jsonl")):
    if "_rep" in Path(f).name:
        continue
    for r in jsonl(f):
        groups[(r["thinking_condition"], int(r["true_cfg_scale"]), int(r["num_inference_steps"]))].append(r)


def row_values(r):
    reps = r.get("latency_pipeline_repeats_sec") or [r["latency_pipeline_sec"]]
    gen = float(np.median(reps))
    think = float(r.get("latency_thinking_sec") or 0.0)
    prep = float(r.get("latency_preprocess_sec") or 0.0)
    post = float(r.get("latency_postprocess_sec") or 0.0)
    return {
        "e2e": prep + think + gen + post,
        "gen": gen,
        "think": think,
        "prep": prep,
        "post": post,
        "vram": float(r["peak_vram_gb"]),
        "recon": float(r["latency_reconstructed_total_sec"]),
    }


summary = {}
lines = [f"{'thinking':<9}{'cfg':>4}{'steps':>6}{'n':>4}{'e2e':>9}{'gen':>9}{'think':>8}"
         f"{'prep':>7}{'post':>7}{'VRAM':>7}{'recon':>9}"]
for key in sorted(groups, key=lambda k: (k[0], k[2])):
    vals = [row_values(r) for r in groups[key]]
    med = {k: float(np.median([v[k] for v in vals])) for k in vals[0]}
    summary[key] = med
    lines.append(f"{key[0]:<9}{key[1]:>4}{key[2]:>6}{len(vals):>4}{med['e2e']:>9.2f}{med['gen']:>9.2f}"
                 f"{med['think']:>8.2f}{med['prep']:>7.2f}{med['post']:>7.2f}{med['vram']:>7.1f}{med['recon']:>9.2f}")
lines.append("e2e = prep + think + gen (median of repeats) + post. recon = logged latency_reconstructed_total_sec.")
text = "\n".join(lines)
print(text)
(OUT / "latency_table_median.txt").write_text(text + "\n")

qrows = defaultdict(list)
for f in ("results/rq2_main_pass1.json", "results/rq2_22_pass1.json"):
    for r in jsonl(f):
        if r.get("Q") is not None:
            qrows[r["condition"]].append(float(r["Q"]))


def qmean(thinking, steps):
    return float(np.mean(qrows[f"rq2_unireason_think-{thinking}_cfg4_steps{steps}"]))


fig, ax = plt.subplots(figsize=(6, 4))
for label, tk in (("thinking off", "off"), ("thinking on", "cached")):
    xs = [summary[(tk, 4, s)]["e2e"] for s in (20, 28, 50)]
    ys = [qmean(tk, s) for s in (20, 28, 50)]
    ax.plot(xs, ys, marker="o", label=label)
    for x, y, s in zip(xs, ys, (20, 28, 50)):
        ax.annotate(f"{s} steps", (x, y), fontsize=8)
ax.set_xlabel("end-to-end latency per image (s, median)")
ax.set_ylabel("mean Q (K-dev)")
ax.set_title("UniReason RQ2.2 quality vs latency, cfg 4")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figures" / "rq2_unireason_2_2_quality_latency_median.png", dpi=150)
print("figure saved")

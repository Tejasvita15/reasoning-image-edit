import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from PIL import Image

MMD = 16.77
N_BOOT = 10000
OUT = Path("outputs/rq2_unireason_analysis")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cname(thinking, cfg, steps):
    return f"rq2_unireason_think-{thinking}_cfg{int(cfg)}_steps{int(steps)}"


def on(cfg, steps=50):
    return cname("cached", cfg, steps)


def off(cfg, steps=50):
    return cname("off", cfg, steps)


def avg(rs, field):
    vals = [float(r[field]) for r in rs if r.get(field) is not None]
    return float(np.mean(vals)) if vals else float("nan")


# ---- Judge scores (K-dev) ----
rows = defaultdict(dict)
for r in jsonl("results/rq2_main_pass1.json") + jsonl("results/rq2_22_pass1.json"):
    if r.get("Q") is None:
        continue
    rows[r["condition"]][r["example_uid"]] = r


def contrast(label, terms):
    common = set.intersection(*[set(rows[c]) for _, c in terms])
    d = np.array([
        sum(s * float(rows[c][u]["Q"]) for s, c in terms)
        for u in sorted(common)
    ])
    boots = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        p = wilcoxon(d).pvalue
    except ValueError:
        p = float("nan")
    flag = "" if abs(d.mean()) >= MMD else "  (below MMD)"
    print(f"{label:<50} n={len(d):3d} diff={d.mean():+7.2f} "
          f"[{lo:+7.2f}, {hi:+7.2f}] p={p:.3f}{flag}")


print("=== RQ2.1 (K-dev, 50 steps) ===")
for cfg in (2, 4, 6):
    contrast(f"thinking on - off, cfg {cfg}", [(1, on(cfg)), (-1, off(cfg))])
for mode, f in (("off", off), ("on", on)):
    contrast(f"guidance 2 - 4, thinking {mode}", [(1, f(2)), (-1, f(4))])
    contrast(f"guidance 6 - 4, thinking {mode}", [(1, f(6)), (-1, f(4))])
contrast("interaction (on6-on2) - (off6-off2)",
         [(1, on(6)), (-1, on(2)), (-1, off(6)), (1, off(2))])

print("\n=== RQ2.2 (K-dev, cfg 4) ===")
for s in (20, 28, 50):
    contrast(f"thinking on - off, {s} steps", [(1, on(4, s)), (-1, off(4, s))])
for mode, f in (("off", off), ("on", on)):
    contrast(f"28 - 20 steps, thinking {mode}", [(1, f(4, 28)), (-1, f(4, 20))])
    contrast(f"50 - 28 steps, thinking {mode}", [(1, f(4, 50)), (-1, f(4, 28))])
contrast("matched latency: on @20 - off @50", [(1, on(4, 20)), (-1, off(4, 50))])
contrast("frozen config: off @50 - on @50", [(1, off(4, 50)), (-1, on(4, 50))])

# ---- Latency (from logs, repeats excluded) ----
lat = defaultdict(list)
log_files = glob.glob("logs/rq2_unireason/*.jsonl") + glob.glob("logs/rq2_unireason_2/*.jsonl")
for f in sorted(log_files):
    if "_rep" in Path(f).name:
        continue
    for r in jsonl(f):
        key = (r["thinking_condition"], int(r["true_cfg_scale"]), int(r["num_inference_steps"]))
        lat[key].append(r)

print("\n=== Latency, seconds per image (offload none_all_on_gpu) ===")
print(f"{'think':<8}{'cfg':>4}{'steps':>7}{'n':>5}{'total':>9}{'generate':>10}"
      f"{'think':>8}{'prep':>8}{'post':>8}{'VRAM GB':>9}")
lat_total = {}
for key in sorted(lat, key=lambda k: (k[0], k[2])):
    rs = lat[key]
    total = avg(rs, "latency_reconstructed_total_sec")
    lat_total[key] = total
    vram = max(float(r["peak_vram_gb"]) for r in rs if r.get("peak_vram_gb") is not None)
    think_vals = [float(r["latency_thinking_sec"]) for r in rs if r.get("latency_thinking_sec") is not None]
    think = float(np.mean(think_vals)) if think_vals else 0.0
    print(f"{key[0]:<8}{key[1]:>4}{key[2]:>7}{len(rs):>5}{total:>9.2f}"
          f"{avg(rs, 'latency_generate_sec'):>10.2f}{think:>8.2f}"
          f"{avg(rs, 'latency_preprocess_sec'):>8.2f}{avg(rs, 'latency_postprocess_sec'):>8.2f}{vram:>9.1f}")

# ---- Determinism (S7): repeat runs vs originals ----
print("\n=== Determinism (S7) ===")
orig = {}
for f in glob.glob("logs/rq2_unireason*/*.jsonl"):
    if "_rep" in Path(f).name or "smoke" in f or "ktest" in f:
        continue
    for r in jsonl(f):
        orig[(r["condition"], r["example_uid"])] = r
same = different = not_comparable = 0
for f in glob.glob("logs/rq2_unireason*/*_rep*.jsonl"):
    for r in jsonl(f):
        o = orig.get((r["condition"], r["example_uid"]))
        p_new, p_old = r.get("output_path", ""), (o or {}).get("output_path", "")
        if not p_new or not p_old or not Path(p_new).exists() or not Path(p_old).exists():
            not_comparable += 1
            continue
        a = np.asarray(Image.open(p_new).convert("RGB"))
        b = np.asarray(Image.open(p_old).convert("RGB"))
        if a.shape == b.shape and np.array_equal(a, b):
            same += 1
        else:
            different += 1
print(f"identical {same}, different {different}, not comparable {not_comparable}")

# ---- Figures ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for label, f in (("thinking off", off), ("thinking on", on)):
    cfgs = [2, 4, 6]
    ax[0].plot(cfgs, [avg(list(rows[f(c)].values()), "instruction_score") for c in cfgs], marker="o", label=label)
    ax[1].plot(cfgs, [avg(list(rows[f(c)].values()), "consistency_score") for c in cfgs], marker="o", label=label)
ax[0].set_title("Instruction following vs text guidance")
ax[1].set_title("Visual consistency vs text guidance")
for a in ax:
    a.set_xlabel("text guidance (cfg)")
    a.legend()
fig.tight_layout()
fig.savefig(FIG / "rq2_unireason_2_1_curves.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 4))
for label, tk, f in (("thinking off", "off", off), ("thinking on", "cached", on)):
    xs = [lat_total[(tk, 4, s)] for s in (20, 28, 50)]
    ys = [float(np.mean([float(r["Q"]) for r in rows[f(4, s)].values()])) for s in (20, 28, 50)]
    ax.plot(xs, ys, marker="o", label=label)
    for x, y, s in zip(xs, ys, (20, 28, 50)):
        ax.annotate(f"{s} steps", (x, y), fontsize=8)
ax.set_xlabel("end-to-end latency per image (s)")
ax.set_ylabel("mean Q (K-dev)")
ax.set_title("RQ2.2 quality vs latency, cfg 4")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "rq2_unireason_2_2_quality_latency.png", dpi=150)
print(f"\nfigures saved in {FIG}")

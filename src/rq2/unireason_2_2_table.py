# Table of UniReason RQ2.2 results: mean Q, 95% bootstrap CI, median latency.
# Usage: python src/rq2/unireason_2_2_table.py [example_uid ...]
# Any example_uid passed in is dropped from every condition (paired).
import glob
import json
import sys

import numpy as np

EXCLUDE = set(sys.argv[1:])
STEP_CONDITIONS = ("cfg4_steps20", "cfg4_steps28", "cfg4_steps50")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Q per condition and example (50 steps from RQ2.1, 20 and 28 from RQ2.2)
scores = {}
for row in load("results/rq2_main_pass1.json") + load("results/rq2_22_pass1.json"):
    cond = row["condition"]
    if not cond.endswith(STEP_CONDITIONS) or row["example_uid"] in EXCLUDE:
        continue
    scores.setdefault(cond, {})[row["example_uid"]] = row["Q"]

# Per-image reconstructed latency from the run logs (no smoke logs)
latency = {}
for path in glob.glob("logs/rq2_unireason/*_k_dev_seed42.jsonl") + glob.glob("logs/rq2_unireason_2/*_k_dev_seed42.jsonl"):
    for row in load(path):
        if row.get("status") != "success" or row.get("latency_reconstructed_total_sec") is None:
            continue
        latency.setdefault(row["condition"], []).append(row["latency_reconstructed_total_sec"])

rng = np.random.default_rng(0)

print(f"Examples excluded: {sorted(EXCLUDE) or 'none'}")
for cond in sorted(scores):
    values = np.array(list(scores[cond].values()), dtype=float)
    idx = rng.integers(0, len(values), size=(10000, len(values)))
    low, high = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    thinking = "on " if "think-cached" in cond else "off"
    steps = cond.split("steps")[1]
    med = float(np.median(latency[cond])) if cond in latency else float("nan")
    print(f"thinking {thinking} steps {steps:>2}  n={len(values)}  meanQ={values.mean():.2f}  "
          f"CI=[{low:.2f}, {high:.2f}]  median_latency_s={med:.2f}")

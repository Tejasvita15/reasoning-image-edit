# RQ2.2 statistics: steps x thinking (Step1X-Edit-v1p2, guidance 6)

Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2). K-dev, 90 examples, seed 42, reflection off. 20/28 steps from RQ2.2, 50 steps from RQ2.1. Latency: end-to-end per image from the single-node timing run (rq2_2_latency.md). Thinking costs ~1.4 s, under one ~2 s denoising step, so thinking on vs off at the same step count is the matched-latency comparison.

Paired by example; mean of per-example differences (A - B) with a 95% bootstrap CI (10,000 resamples over examples) and a Wilcoxon signed-rank p (zero differences dropped) as a check; no multiple-comparison correction. Judge MMD = 19.06 Q points: differences in Q smaller than the MMD are not claimed as meaningful (plan S8). Components are on the judge's 1-5 scale; KP covers knowledge-family examples only.

## Thinking at matched latency

### Q (primary)

| Comparison (A vs B) | n | Q A | Q B | A - B [95% CI] | Wilcoxon p | A better / tie / B better | Reaches MMD | Latency A - B (s) |
|---|---|---|---|---|---|---|---|---|
| thinking on, cfg 6, 20 steps vs thinking off, cfg 6, 20 steps | 90 | 57.18 | 53.98 | +3.19 [-0.79, +7.29] | 0.0882 | 34 / 31 / 25 | no | +1.45 |
| thinking on, cfg 6, 28 steps vs thinking off, cfg 6, 28 steps | 90 | 59.49 | 55.90 | +3.59 [+0.14, +7.13] | 0.1098 | 40 / 18 / 32 | no | +1.60 |
| thinking on, cfg 6, 50 steps vs thinking off, cfg 6, 50 steps | 90 | 58.56 | 53.87 | +4.70 [+1.11, +8.47] | 0.0359 | 42 / 22 / 26 | no | +1.45 |

### Components (A - B [95% CI], Wilcoxon p)

| Comparison (A vs B) | VC | VQ | IF | KP |
|---|---|---|---|---|
| thinking on, cfg 6, 20 steps vs thinking off, cfg 6, 20 steps | +0.29 [+0.03, +0.57], p=0.049 | -0.23 [-0.47, -0.01], p=0.053 | +0.29 [+0.07, +0.52], p=0.019 | +0.18 [-0.06, +0.49], p=0.202 |
| thinking on, cfg 6, 28 steps vs thinking off, cfg 6, 28 steps | +0.24 [+0.00, +0.50], p=0.083 | +0.00 [-0.22, +0.22], p=0.951 | +0.21 [+0.03, +0.40], p=0.033 | +0.04 [-0.10, +0.22], p=0.726 |
| thinking on, cfg 6, 50 steps vs thinking off, cfg 6, 50 steps | +0.20 [-0.07, +0.48], p=0.220 | +0.04 [-0.21, +0.30], p=0.695 | +0.37 [+0.18, +0.57], p=0.001 | +0.12 [-0.06, +0.35], p=0.336 |

## Steps effect

### Q (primary)

| Comparison (A vs B) | n | Q A | Q B | A - B [95% CI] | Wilcoxon p | A better / tie / B better | Reaches MMD | Latency A - B (s) |
|---|---|---|---|---|---|---|---|---|
| thinking off, cfg 6, 28 steps vs thinking off, cfg 6, 20 steps | 90 | 55.90 | 53.98 | +1.92 [-0.23, +4.21] | 0.0647 | 29 / 39 / 22 | no | +15.65 |
| thinking off, cfg 6, 50 steps vs thinking off, cfg 6, 28 steps | 90 | 53.87 | 55.90 | -2.04 [-4.21, +0.14] | 0.0716 | 21 / 34 / 35 | no | +43.60 |
| thinking off, cfg 6, 50 steps vs thinking off, cfg 6, 20 steps | 90 | 53.87 | 53.98 | -0.12 [-2.59, +2.45] | 0.9836 | 25 / 34 / 31 | no | +59.25 |
| thinking on, cfg 6, 28 steps vs thinking on, cfg 6, 20 steps | 90 | 59.49 | 57.18 | +2.31 [-0.16, +5.14] | 0.1090 | 26 / 43 / 21 | no | +15.80 |
| thinking on, cfg 6, 50 steps vs thinking on, cfg 6, 28 steps | 90 | 58.56 | 59.49 | -0.93 [-3.31, +1.37] | 0.6683 | 24 / 39 / 27 | no | +43.45 |
| thinking on, cfg 6, 50 steps vs thinking on, cfg 6, 20 steps | 90 | 58.56 | 57.18 | +1.39 [-1.53, +4.49] | 0.3670 | 34 / 31 / 25 | no | +59.25 |

### Components (A - B [95% CI], Wilcoxon p)

| Comparison (A vs B) | VC | VQ | IF | KP |
|---|---|---|---|---|
| thinking off, cfg 6, 28 steps vs thinking off, cfg 6, 20 steps | +0.09 [-0.06, +0.24], p=0.267 | +0.01 [-0.13, +0.16], p=0.987 | +0.11 [-0.01, +0.26], p=0.129 | +0.10 [-0.08, +0.33], p=0.395 |
| thinking off, cfg 6, 50 steps vs thinking off, cfg 6, 28 steps | -0.20 [-0.38, -0.02], p=0.037 | -0.04 [-0.21, +0.12], p=0.652 | -0.04 [-0.14, +0.06], p=0.371 | -0.02 [-0.14, +0.08], p=0.705 |
| thinking off, cfg 6, 50 steps vs thinking off, cfg 6, 20 steps | -0.11 [-0.29, +0.07], p=0.236 | -0.03 [-0.22, +0.16], p=0.815 | +0.07 [-0.04, +0.20], p=0.361 | +0.08 [-0.12, +0.31], p=0.595 |
| thinking on, cfg 6, 28 steps vs thinking on, cfg 6, 20 steps | +0.04 [-0.12, +0.23], p=0.850 | +0.24 [+0.07, +0.43], p=0.013 | +0.03 [-0.09, +0.16], p=0.682 | -0.04 [-0.14, +0.04], p=0.414 |
| thinking on, cfg 6, 50 steps vs thinking on, cfg 6, 28 steps | -0.24 [-0.39, -0.10], p=0.002 | +0.00 [-0.19, +0.19], p=0.937 | +0.11 [+0.01, +0.21], p=0.043 | +0.06 [-0.02, +0.14], p=0.180 |
| thinking on, cfg 6, 50 steps vs thinking on, cfg 6, 20 steps | -0.20 [-0.40, +0.01], p=0.036 | +0.24 [+0.02, +0.48], p=0.026 | +0.14 [+0.03, +0.26], p=0.016 | +0.02 [-0.10, +0.12], p=0.739 |

## Plan example pair

### Q (primary)

| Comparison (A vs B) | n | Q A | Q B | A - B [95% CI] | Wilcoxon p | A better / tie / B better | Reaches MMD | Latency A - B (s) |
|---|---|---|---|---|---|---|---|---|
| thinking on, cfg 6, 20 steps vs thinking off, cfg 6, 50 steps | 90 | 57.18 | 53.87 | +3.31 [-0.72, +7.36] | 0.1144 | 43 / 18 / 29 | no | -57.80 |

### Components (A - B [95% CI], Wilcoxon p)

| Comparison (A vs B) | VC | VQ | IF | KP |
|---|---|---|---|---|
| thinking on, cfg 6, 20 steps vs thinking off, cfg 6, 50 steps | +0.40 [+0.14, +0.67], p=0.006 | -0.20 [-0.50, +0.08], p=0.179 | +0.22 [+0.01, +0.43], p=0.051 | +0.10 [-0.10, +0.35], p=0.491 |


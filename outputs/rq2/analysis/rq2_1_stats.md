# RQ2.1 statistics: guidance x thinking (Step1X-Edit-v1p2, 50 steps)

Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2). K-dev, 90 examples, seed 42, reflection off.

Paired by example; mean of per-example differences (A - B) with a 95% bootstrap CI (10,000 resamples over examples) and a Wilcoxon signed-rank p (zero differences dropped) as a check; no multiple-comparison correction. Judge MMD = 19.06 Q points: differences in Q smaller than the MMD are not claimed as meaningful (plan S8). Components are on the judge's 1-5 scale; KP covers knowledge-family examples only.

## Guidance selected for RQ2.2

Rule: highest mean Q; keep the default (cfg 6) if the winner beats it by less than the MMD.

| Thinking | Q cfg 2 | Q cfg 6 | Q cfg 10 | Highest | Winner - cfg 6 | Selected |
|---|---|---|---|---|---|---|
| off | 50.25 | 53.87 | 54.21 | cfg 10 | +0.35 | **cfg 6** |
| on | 53.84 | 58.56 | 56.53 | cfg 6 | +0.00 | **cfg 6** |

## Thinking effect

### Q (primary)

| Comparison (A vs B) | n | Q A | Q B | A - B [95% CI] | Wilcoxon p | A better / tie / B better | Reaches MMD |
|---|---|---|---|---|---|---|---|
| thinking on, cfg 2, 50 steps vs thinking off, cfg 2, 50 steps | 90 | 53.84 | 50.25 | +3.59 [+0.16, +7.13] | 0.0911 | 38 / 20 / 32 | no |
| thinking on, cfg 6, 50 steps vs thinking off, cfg 6, 50 steps | 90 | 58.56 | 53.87 | +4.70 [+1.11, +8.47] | 0.0359 | 42 / 22 / 26 | no |
| thinking on, cfg 10, 50 steps vs thinking off, cfg 10, 50 steps | 90 | 56.53 | 54.21 | +2.31 [-1.83, +6.55] | 0.4780 | 37 / 18 / 35 | no |

### Components (A - B [95% CI], Wilcoxon p)

| Comparison (A vs B) | VC | VQ | IF | KP |
|---|---|---|---|---|
| thinking on, cfg 2, 50 steps vs thinking off, cfg 2, 50 steps | +0.12 [-0.13, +0.39], p=0.479 | +0.04 [-0.18, +0.27], p=0.572 | +0.32 [+0.12, +0.53], p=0.003 | +0.02 [-0.22, +0.29], p=1.000 |
| thinking on, cfg 6, 50 steps vs thinking off, cfg 6, 50 steps | +0.20 [-0.07, +0.48], p=0.220 | +0.04 [-0.21, +0.30], p=0.695 | +0.37 [+0.18, +0.57], p=0.001 | +0.12 [-0.06, +0.35], p=0.336 |
| thinking on, cfg 10, 50 steps vs thinking off, cfg 10, 50 steps | +0.22 [-0.06, +0.51], p=0.168 | -0.19 [-0.42, +0.04], p=0.112 | +0.22 [+0.00, +0.46], p=0.095 | +0.06 [-0.16, +0.33], p=0.835 |

## Guidance effect

### Q (primary)

| Comparison (A vs B) | n | Q A | Q B | A - B [95% CI] | Wilcoxon p | A better / tie / B better | Reaches MMD |
|---|---|---|---|---|---|---|---|
| thinking off, cfg 2, 50 steps vs thinking off, cfg 6, 50 steps | 90 | 50.25 | 53.87 | -3.61 [-7.08, -0.37] | 0.0445 | 30 / 25 / 35 | no |
| thinking off, cfg 10, 50 steps vs thinking off, cfg 6, 50 steps | 90 | 54.21 | 53.87 | +0.35 [-1.76, +2.50] | 0.9516 | 28 / 31 / 31 | no |
| thinking off, cfg 10, 50 steps vs thinking off, cfg 2, 50 steps | 90 | 54.21 | 50.25 | +3.96 [+0.37, +7.87] | 0.0854 | 40 / 21 / 29 | no |
| thinking on, cfg 2, 50 steps vs thinking on, cfg 6, 50 steps | 90 | 53.84 | 58.56 | -4.72 [-8.59, -1.00] | 0.0192 | 26 / 27 / 37 | no |
| thinking on, cfg 10, 50 steps vs thinking on, cfg 6, 50 steps | 90 | 56.53 | 58.56 | -2.04 [-4.84, +0.58] | 0.1124 | 24 / 28 / 38 | no |
| thinking on, cfg 10, 50 steps vs thinking on, cfg 2, 50 steps | 90 | 56.53 | 53.84 | +2.69 [-1.13, +6.71] | 0.1816 | 34 / 21 / 35 | no |

### Components (A - B [95% CI], Wilcoxon p)

| Comparison (A vs B) | VC | VQ | IF | KP |
|---|---|---|---|---|
| thinking off, cfg 2, 50 steps vs thinking off, cfg 6, 50 steps | +0.01 [-0.19, +0.20], p=0.863 | -0.22 [-0.46, +0.00], p=0.057 | -0.19 [-0.40, +0.01], p=0.083 | -0.14 [-0.41, +0.06], p=0.269 |
| thinking off, cfg 10, 50 steps vs thinking off, cfg 6, 50 steps | -0.11 [-0.30, +0.09], p=0.197 | +0.06 [-0.11, +0.23], p=0.475 | +0.09 [-0.02, +0.21], p=0.134 | +0.06 [-0.04, +0.18], p=0.317 |
| thinking off, cfg 10, 50 steps vs thinking off, cfg 2, 50 steps | -0.12 [-0.37, +0.13], p=0.336 | +0.28 [+0.02, +0.54], p=0.029 | +0.28 [+0.06, +0.52], p=0.032 | +0.20 [-0.02, +0.49], p=0.155 |
| thinking on, cfg 2, 50 steps vs thinking on, cfg 6, 50 steps | -0.07 [-0.31, +0.18], p=0.601 | -0.22 [-0.50, +0.06], p=0.112 | -0.23 [-0.41, -0.06], p=0.025 | -0.24 [-0.51, -0.02], p=0.061 |
| thinking on, cfg 10, 50 steps vs thinking on, cfg 6, 50 steps | -0.09 [-0.29, +0.11], p=0.445 | -0.18 [-0.39, +0.03], p=0.075 | -0.06 [-0.16, +0.04], p=0.302 | +0.00 [-0.06, +0.06], p=1.000 |
| thinking on, cfg 10, 50 steps vs thinking on, cfg 2, 50 steps | -0.02 [-0.27, +0.22], p=0.813 | +0.04 [-0.24, +0.33], p=0.602 | +0.18 [-0.01, +0.38], p=0.088 | +0.24 [+0.02, +0.51], p=0.061 |

## Guidance sensitivity (interaction)

### Q (primary)

| Comparison (A vs B) | n | Q A | Q B | A - B [95% CI] | Wilcoxon p | A better / tie / B better | Reaches MMD |
|---|---|---|---|---|---|---|---|
| (on cfg10 - on cfg2) - (off cfg10 - off cfg2), 50 steps | 90 | 2.69 | 3.96 | -1.27 [-5.16, +2.57] | 0.7712 | 34 / 15 / 41 | no |

### Components (A - B [95% CI], Wilcoxon p)

| Comparison (A vs B) | VC | VQ | IF | KP |
|---|---|---|---|---|
| (on cfg10 - on cfg2) - (off cfg10 - off cfg2), 50 steps | +0.10 [-0.20, +0.41], p=0.689 | -0.23 [-0.53, +0.07], p=0.144 | -0.10 [-0.36, +0.14], p=0.476 | +0.04 [-0.20, +0.27], p=0.560 |


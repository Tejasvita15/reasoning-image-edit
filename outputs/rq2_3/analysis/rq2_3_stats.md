# RQ2.3 statistics: feedback vs resampling (Step1X-Edit-v1p2)

K-small (40 examples) x seeds 42, 1234, 7; thinking on (cached plan), guidance 6, 28 steps. Each example's value is the mean over the three seeds (plan S8), so n = 40. Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2), blind to arm and selector; every distinct image judged once.
Feedback and resampling match in candidate budget and generator calls per example; the last column is their mean latency difference (A - B).

Paired by example; mean of per-example differences (A - B) with a 95% bootstrap CI (10,000 resamples over examples) and a Wilcoxon signed-rank p (zero differences dropped) as a check; no multiple-comparison correction. Judge MMD = 19.06 Q points: differences in Q smaller than the MMD are not claimed as meaningful (plan S8). Components are on the judge's 1-5 scale; KP covers knowledge-family examples only.

## Feedback vs resampling

### Q (primary)

| Comparison (A vs B) | n | Q A | Q B | A - B [95% CI] | Wilcoxon p | A better / tie / B better | Reaches MMD | Latency A - B (s) |
|---|---|---|---|---|---|---|---|---|
| Feedback (selector) vs Resampling (selector) | 40 | 53.59 | 54.62 | -1.02 [-2.45, +0.26] | 0.2168 | 8 / 19 / 13 | no | +32.49 |
| Feedback (native) vs Resampling (selector) | 40 | 53.61 | 54.62 | -1.01 [-2.85, +0.83] | 0.3604 | 10 / 14 / 16 | no | +31.96 |
| Resampling (selector) vs First candidate only | 40 | 54.62 | 53.98 | +0.64 [-0.71, +2.10] | 0.6141 | 11 / 20 / 9 | no | +40.36 |
| Feedback (selector) vs First candidate only | 40 | 53.59 | 53.98 | -0.38 [-1.55, +0.75] | 0.6790 | 7 / 24 / 9 | no | +72.85 |
| Feedback (native) vs First candidate only | 40 | 53.61 | 53.98 | -0.36 [-1.70, +0.90] | 0.7171 | 9 / 21 / 10 | no | +72.32 |

### Components (A - B [95% CI], Wilcoxon p)

| Comparison (A vs B) | VC | VQ | IF | KP |
|---|---|---|---|---|
| Feedback (selector) vs Resampling (selector) | -0.05 [-0.14, +0.03], p=0.376 | -0.08 [-0.22, +0.03], p=0.166 | -0.02 [-0.08, +0.04], p=0.892 | +0.02 [+0.00, +0.05], p=0.317 |
| Feedback (native) vs Resampling (selector) | -0.06 [-0.17, +0.05], p=0.242 | -0.12 [-0.26, +0.02], p=0.116 | +0.03 [-0.05, +0.11], p=0.420 | +0.03 [+0.00, +0.08], p=0.157 |
| Resampling (selector) vs First candidate only | -0.01 [-0.12, +0.10], p=0.875 | +0.12 [+0.00, +0.25], p=0.059 | -0.00 [-0.06, +0.06], p=0.725 | +0.00 [-0.05, +0.05], p=0.655 |
| Feedback (selector) vs First candidate only | -0.06 [-0.17, +0.06], p=0.261 | +0.03 [-0.05, +0.12], p=0.552 | -0.02 [-0.08, +0.04], p=0.518 | +0.02 [+0.00, +0.05], p=0.317 |
| Feedback (native) vs First candidate only | -0.07 [-0.17, +0.05], p=0.223 | -0.00 [-0.09, +0.09], p=0.894 | +0.03 [-0.03, +0.09], p=0.491 | +0.03 [+0.00, +0.08], p=0.180 |


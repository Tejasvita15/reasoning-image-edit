# RQ2.3 Feedback vs resampling (Step1X-Edit-v1p2)

K-small (40 examples) x seeds 42, 1234, 7; thinking on (cached plan), guidance 6, 28 steps. Both strategies share candidate 1. Feedback: Step1X reflection with max_try_cnt=2. Resampling: a fresh draw from the same plan with new noise. A second candidate is made only when Step1X's reflection rejects candidate 1, in both arms, so candidate budgets match per example. Selector: EditReward-MiMo-VL-7B on the original KRIS instruction, never the judge. Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2), blind to arm and selector; every distinct image judged once.

Second attempts: 73/120 example-seeds (61%); feedback retry type: refine (edit candidate 1) 39, restart (redraw from source) 34.

Q on 0-100: per-example mean over the three seeds, then the mean over the 40 examples, with a 95% bootstrap interval. Judge MMD = 19.06 Q points. Latency is the mean per example-seed on one A100 node with model CPU offload: thinking (charged from the cache) + generation (+ reflection and native selection for feedback) + EditReward selection when there are two candidates. Feedback and resampling match in candidates and generator calls, not in latency.

| Strategy | Q [95% CI] | VC | VQ | IF | KP | Candidates | Generator calls | Think + generate (+ reflect) + select = total (s) | Kept candidate 2 |
|---|---|---|---|---|---|---|---|---|---|
| First candidate only | 53.98 [46.68, 61.39] | 3.17 | 4.17 | 2.47 | 2.48 | 1.00 | 56 | 2.0 + 65.6 + 0.0 = 67.6 | 0/120 |
| Resampling (selector) | 54.62 [47.33, 62.10] | 3.17 | 4.28 | 2.47 | 2.48 | 1.61 | 90 | 2.0 + 105.4 + 0.5 = 107.9 | 41/120 |
| Feedback (selector) | 53.59 [46.39, 61.06] | 3.12 | 4.20 | 2.45 | 2.50 | 1.61 | 90 | 2.0 + 137.9 + 0.5 = 140.4 | 35/120 |
| Feedback (native) | 53.61 [46.46, 60.87] | 3.11 | 4.17 | 2.49 | 2.52 | 1.61 | 90 | 2.0 + 137.9 + 0.0 = 139.9 | 48/120 |

## Did the second candidate beat the first? (judge Q, example-seeds with a second candidate)

"Kept the better" counts, over pairs the judge did not tie, how often the selection rule kept the candidate the judge scored higher.

| Candidate 2 | Subset | n | Beat / tie / lost to candidate 1 | Mean Q gain | Selector kept the better | Native kept the better |
|---|---|---|---|---|---|---|
| feedback | all retries | 73 | 15 / 30 / 28 | -3.51 | 28/43 | 26/43 |
| feedback | feedback retry = refine | 39 | 5 / 17 / 17 | -5.13 | 18/22 | 13/22 |
| feedback | feedback retry = restart | 34 | 10 / 13 / 11 | -1.65 | 10/21 | 13/21 |
| resampling | all retries | 73 | 24 / 24 / 25 | +0.03 | 27/49 | - |
| resampling | feedback retry = refine | 39 | 16 / 13 / 10 | +3.31 | 16/26 | - |
| resampling | feedback retry = restart | 34 | 8 / 11 / 15 | -3.74 | 11/23 | - |

The refine/restart split is by the feedback arm's retry type; for resampling it groups the same example-seeds (resampling itself always redraws).

## By knowledge type and seed (descriptive only)

| Strategy | Factual | Conceptual | Procedural | Seed 42 | Seed 1234 | Seed 7 |
|---|---|---|---|---|---|---|
| First candidate only | 50.20 | 64.10 | 47.92 | 53.18 | 59.01 | 49.74 |
| Resampling (selector) | 48.21 | 65.06 | 51.07 | 54.17 | 60.26 | 49.43 |
| Feedback (selector) | 48.41 | 63.94 | 48.82 | 52.66 | 58.49 | 49.64 |
| Feedback (native) | 48.81 | 63.94 | 48.45 | 52.71 | 58.02 | 50.10 |

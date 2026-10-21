# Extra 2: when does correction help? (RQ2.3 data)

Gains are judge Q (0-100) of a second candidate minus Q of the shared first candidate, per example and seed, using raw second-candidate scores (not the selector's pick). Existing RQ2.3 judge scores; no new runs or judging. 95% CIs bootstrap whole examples (all seeds together, 10,000 resamples). Terciles are cut on Q(first) values.

Caveat: low first scores tend to rise on any retry (regression to the mean), so a downward trend in either gain alone proves nothing. Feedback - resampling cancels that effect, because both arms share the first candidate.

## Step1X

Runs with a second candidate: 73 over 34 examples; runs without one (reflection accepted the first candidate, not plotted): 47.

| | Mean gain [95% CI] | Spearman ρ vs Q(first) [95% CI] | Q(first) 0-33 (n=27) | Q(first) 38-50 (n=28) | Q(first) 56-100 (n=18) |
|---|---|---|---|---|---|
| Feedback gain | -3.51 [-6.11, -0.83] | -0.54 [-0.69, -0.34] | +2.39 [-0.85, +5.44] | -1.64 [-5.31, +2.08] | -15.28 [-22.79, -9.79] |
| Resampling gain | +0.03 [-2.73, +2.87] | -0.37 [-0.58, -0.13] | +5.09 [+1.21, +9.84] | +1.12 [-2.97, +5.71] | -9.26 [-19.62, -0.61] |
| Feedback - resampling | -3.54 [-6.60, -0.79] | -0.13 [-0.36, +0.12] | -2.70 [-8.49, +1.84] | -2.75 [-7.17, +1.18] | -6.02 [-14.44, +3.33] |

Every tercile has at least 10 runs.


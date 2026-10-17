# RQ2.2 frozen configuration on K-test (Step1X-Edit-v1p2)

Configuration frozen on K-dev before any K-test run (highest mean Q on K-dev (guidance 6, thinking off/on x 20/28/50 steps)): thinking on, 28 steps, guidance 6, reflection off. K-test, seed 42. Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2). Reported separately from RQ1.1 (different settings).
Same configuration on K-dev: mean Q 59.49 (n=90). Judge MMD: 19.06 Q points.

Q on 0-100 with a 95% bootstrap interval (10,000 resamples over examples); components are mean judge scores on 1-5. Per-knowledge-type rows are descriptive only.

| Subset | n | Q | 95% CI | Visual consistency | Visual quality | Instruction following | Knowledge plausibility (n) |
|---|---|---|---|---|---|---|---|
| all | 300 | 58.8 | 55.8-61.6 | 3.57 | 4.17 | 2.79 | 2.29 (163) |
| factual | 100 | 62.2 | 57.6-67.2 | 3.29 | 4.49 | 2.69 | - |
| conceptual | 100 | 63.1 | 57.8-68.3 | 3.88 | 4.19 | 3.24 | 2.79 (100) |
| procedural | 100 | 50.9 | 46.2-55.7 | 3.53 | 3.82 | 2.45 | 1.49 (63) |

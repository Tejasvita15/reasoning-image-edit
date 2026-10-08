# RQ2.2 summary: steps x thinking on K-dev (Step1X-Edit-v1p2, guidance 6)

Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2). K-dev, 90 examples, seed 42, reflection off. 20/28 steps from RQ2.2, 50 steps from RQ2.1. Latency: end-to-end per image (preprocess + think + generate + postprocess), single-node timing run on an A100 (see rq2_2_latency.md).
Judge MMD (RQ2.1 calibration): 19.06 Q points.

Q on 0-100; components are mean judge scores on 1-5; knowledge plausibility over knowledge-family outputs only.

| Thinking | Steps | n | Latency (s) | Q | Visual consistency | Visual quality | Instruction following | Knowledge plausibility | Generator calls |
|---|---|---|---|---|---|---|---|---|---|
| off | 20 | 90 | 50.2 | 54.0 | 3.36 | 4.24 | 2.50 | 1.82 | 40 |
| off | 28 | 90 | 65.8 | 55.9 | 3.44 | 4.26 | 2.61 | 1.92 | 56 |
| off | 50 | 90 | 109.4 | 53.9 | 3.24 | 4.21 | 2.57 | 1.90 | 100 |
| on | 20 | 90 | 51.6 | 57.2 | 3.64 | 4.01 | 2.79 | 2.00 | 40 |
| on | 28 | 90 | 67.4 | 59.5 | 3.69 | 4.26 | 2.82 | 1.96 | 56 |
| on | 50 | 90 | 110.9 | 58.6 | 3.44 | 4.26 | 2.93 | 2.02 | 100 |

Frozen configuration for the single K-test run (highest mean Q on K-dev (guidance 6, thinking off/on x 20/28/50 steps)): thinking on, 28 steps, guidance 6 (mean Q 59.5).

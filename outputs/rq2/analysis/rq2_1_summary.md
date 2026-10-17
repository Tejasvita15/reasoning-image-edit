# RQ2.1 summary: guidance x thinking (Step1X-Edit-v1p2)

Judge: GPT-6 Astra Pro (ChatGPT UI, prompt version kris-chatgpt-ui-v2). Split: k_dev, seed 42, 50 steps, reflection off.
Judge repeatability (calibration, 20 outputs): MMD = 19.06 Q points, mean |dQ| = 5.31.

Q on 0-100; components are mean judge scores on 1-5; knowledge plausibility over knowledge-family outputs only.

| Thinking | Guidance | n | Q | Visual consistency | Visual quality | Instruction following | Knowledge plausibility (n) |
|---|---|---|---|---|---|---|---|
| off | 2 | 90 | 50.2 | 3.26 | 3.99 | 2.38 | 1.76 (49) |
| off | 6 | 90 | 53.9 | 3.24 | 4.21 | 2.57 | 1.90 (49) |
| off | 10 | 90 | 54.2 | 3.13 | 4.27 | 2.66 | 1.96 (49) |
| on | 2 | 90 | 53.8 | 3.38 | 4.03 | 2.70 | 1.78 (49) |
| on | 6 | 90 | 58.6 | 3.44 | 4.26 | 2.93 | 2.02 (49) |
| on | 10 | 90 | 56.5 | 3.36 | 4.08 | 2.88 | 2.02 (49) |

# RQ2.2 latency: steps x thinking (Step1X-Edit-v1p2)

Timed in one job on one node (x1000c0s2b0n1; NVIDIA A100-SXM4-40GB), offload: model_cpu_offload, after a warm-up generation. Each stage is the median over 10 K-dev examples; generation per example is the median of 3 repeats. Thinking is one thinking pass (timed on the same node), charged to every thinking-on generation. Total is the median of each example's preprocess + think + generate + postprocess, so it can differ slightly from the sum of the stage medians.

| Thinking | Steps | Preprocess (s) | Think (s) | Generate (s) | Postprocess (s) | Total (s) | Generator calls | Thinking tokens | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|---|
| off | 20 | 0.002 | 0.00 | 49.87 | 0.205 | 50.17 | 40 | 0 | 24.3 |
| off | 28 | 0.002 | 0.00 | 65.40 | 0.210 | 65.82 | 56 | 0 | 24.3 |
| off | 50 | 0.002 | 0.00 | 108.98 | 0.219 | 109.42 | 100 | 0 | 24.3 |
| on | 20 | 0.002 | 1.39 | 49.79 | 0.240 | 51.62 | 40 | 12 | 24.3 |
| on | 28 | 0.002 | 1.39 | 65.51 | 0.243 | 67.42 | 56 | 12 | 24.3 |
| on | 50 | 0.002 | 1.39 | 108.87 | 0.190 | 110.87 | 100 | 12 | 24.3 |

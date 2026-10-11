# RQ2 results summary: Step1X-Edit-v1p2

RQ2 asks how Step1X's inference-time settings trade quality against compute on KRIS-Bench:

- **RQ2.1:** does thinking change how sensitive editing is to guidance?
- **RQ2.2:** where is extra compute best spent, on more denoising steps or on thinking?
- **RQ2.3:** with one extra attempt, is self-correction (feedback) better than simply trying again (resampling)?

All three sub-questions are complete: generation, judging, statistics and figures. This file states what was measured, the result of each part, and where every file is. The design is in [RQ2_plan.md](RQ2_plan.md).

## Headline findings

1. **Thinking helps a little, consistently.** Turning on Step1X's thinking (the built-in instruction rewrite) raises quality by about +2 to +5 Q at every guidance value and step count tested. It costs only about 1.4 s per image. The gain is never large enough to rule out judge noise (MMD = 19.06 Q).
2. **Guidance 6 (the default) is the right choice.** Guidance 2 is worse. Guidance 10 is no better than 6 with thinking off, and slightly worse with thinking on. Thinking does not measurably change how sensitive the model is to guidance.
3. **More denoising steps do not buy quality.** 20, 28 and 50 steps give statistically indistinguishable quality, while latency rises from about 50 s to about 110 s. Compute is better spent on thinking (+1.4 s) than on steps.
4. **Frozen configuration: thinking on, 28 steps, guidance 6.** On the held-out K-test (300 examples) it scores Q = 58.8 [55.8, 61.6], matching its K-dev score of 59.5.
5. **Self-correction does not beat trying again.**
   - With the same number of candidates, feedback scores −1.0 Q [−2.5, +0.3] relative to resampling and is 32 s slower.
   - Neither kind of second attempt measurably improves on the first image.
   - When Step1X "refines" by editing its own first output, the result is usually worse.

None of the differences in RQ2 reaches the MMD. The judge's own re-scoring spread is about 19 Q, so the effects above are measurable trends, not decisive wins.

---

## Common setup (all of RQ2)

| Item | Setting |
|---|---|
| Model | `stepfun-ai/Step1X-Edit-v1p2`, bf16, diffusers branch `Peyton-Chen/diffusers@step1xedit_v1p2` |
| Hardware | ASPIRE2A, 1× A100 40 GB; model-level CPU offload for every condition; peak VRAM 24.3 GB |
| Thinking | Step1X's built-in thinker rewrites the instruction (about 12 tokens). Each rewrite is generated once, cached, then replayed as the prompt with pipeline thinking off, so every condition sees the identical thought. One thinking pass (about 1.4–1.9 s) is charged to every thinking-on image. |
| Noise | Same starting noise for every condition of an example (seed derived from base seed + example ID). PyTorch deterministic mode is on. The determinism check (S7) passed: reruns are bitwise identical across nodes. |
| Reflection | Off in RQ2.1 and RQ2.2. On only for the feedback arm of RQ2.3. |
| Splits | K-dev 90 (tuning), K-test 300 (used once, for the frozen configuration), K-small 40 (RQ2.3; a subset of K-test, but nothing was tuned on it) |
| Judge | The team's shared judge: "GPT-6 Astra Pro" in the ChatGPT UI, prompt version `kris-chatgpt-ui-v2`, 2 images per chat, fresh temporary chat with memory and tools off. Evaluation protocol hash `04d422b5…`, identical for every RQ2 bundle. |
| Quality score Q | For each image, the judge scores the applicable components on 1–5: visual consistency (VC), visual quality (VQ), instruction following (IF), and knowledge plausibility (KP, knowledge-type tasks only). Q = mean of 25 × (score − 1), on 0–100. |
| Judge noise (MMD) | Calibration: the same 20 images judged twice. MMD = 95th percentile of \|ΔQ\| = **19.06 Q**; mean \|ΔQ\| = 5.31. Following the plan, no difference smaller than the MMD is claimed as meaningful. |
| Statistics | Paired by example; mean of per-example differences with a 95% paired bootstrap CI (10,000 resamples); Wilcoxon signed-rank p as a check; no multiple-comparison correction. Per-knowledge-type results are descriptive only. |

**How to read the tables.** "A − B [CI]" is the mean paired difference with its 95% interval. If the interval excludes 0, the trend is consistent across examples. It still counts as meaningful only if it also reaches the MMD, and none in RQ2 does.

---

## RQ2.1 Does thinking change guidance sensitivity?

**What we ran:** K-dev (90), seed 42, 50 steps, reflection off. Thinking off/on × guidance (`true_cfg_scale`) 2, 6 and 10 gives 6 configurations and 540 judged images (270 chats).

**What we measured:** Q and its components for each configuration, the thinking effect at each guidance value, the guidance effect, and the interaction (does thinking flatten the response to guidance?). We also chose the guidance value for RQ2.2.

### Results

| Thinking | Guidance | Q | VC | VQ | IF | KP |
|---|---|---|---|---|---|---|
| off | 2 | 50.2 | 3.26 | 3.99 | 2.38 | 1.76 |
| off | 6 | 53.9 | 3.24 | 4.21 | 2.57 | 1.90 |
| off | 10 | 54.2 | 3.13 | 4.27 | 2.66 | 1.96 |
| on | 2 | 53.8 | 3.38 | 4.03 | 2.70 | 1.78 |
| on | 6 | **58.6** | 3.44 | 4.26 | 2.93 | 2.02 |
| on | 10 | 56.5 | 3.36 | 4.08 | 2.88 | 2.02 |

| Comparison | A − B [95% CI] | Wilcoxon p |
|---|---|---|
| Thinking on − off, guidance 2 | +3.59 [+0.16, +7.13] | 0.091 |
| Thinking on − off, guidance 6 | **+4.70 [+1.11, +8.47]** | 0.036 |
| Thinking on − off, guidance 10 | +2.31 [−1.83, +6.55] | 0.478 |
| Guidance 2 − 6, thinking off | −3.61 [−7.08, −0.37] | 0.045 |
| Guidance 2 − 6, thinking on | −4.72 [−8.59, −1.00] | 0.019 |
| Guidance 10 − 6, thinking off | +0.35 [−1.76, +2.50] | 0.952 |
| Guidance 10 − 6, thinking on | −2.04 [−4.84, +0.58] | 0.112 |
| Interaction: (on 10 − on 2) − (off 10 − off 2) | −1.27 [−5.16, +2.57] | 0.771 |

### What it means

- Thinking improves Q at every guidance value, most clearly at guidance 6.
- Low guidance (2) hurts in both modes. Above 6 there is no gain.
- The interaction is about 0: thinking does not make the model less sensitive to guidance.
- **Selected guidance for RQ2.2: 6 for both modes.** The rule was the highest mean Q, keeping the default unless beaten by at least the MMD. With thinking off, guidance 10 beat 6 by only 0.35.

**Files**
- `outputs/rq2/analysis/rq2_1_summary.md` / `.csv`: configuration means
- `outputs/rq2/analysis/rq2_1_stats.md` / `.csv`: all paired comparisons, including the components
- `outputs/rq2/analysis/figures/rq2_1_editing_curve.png`: instruction following vs guidance
- `outputs/rq2/analysis/figures/rq2_1_consistency_curve.png`: visual consistency vs guidance
- Judge scores: `outputs/rq2/judge/chatgpt_main/scores_main.jsonl`. Calibration: `outputs/rq2/judge/chatgpt_calibration/repeatability.json`
- Images and logs: `outputs/rq2/` and `logs/rq2/`. Thought cache: `outputs/rq2/thought_cache/k_dev_seed42.jsonl`

---

## RQ2.2 Where is extra compute best spent?

**What we ran:**
- **Quality runs:** K-dev (90), seed 42, guidance 6, reflection off; thinking off/on × 20, 28 and 50 steps. The 50-step images are reused from RQ2.1. That is 360 new judged images (180 chats).
- **Latency runs:** a separate timing job on one A100 node: 10 K-dev examples, all 6 configurations, after a warm-up, median of 3 repeats. Each image's time is split into preprocess, think, generate and postprocess.
- **Frozen configuration:** run once on K-test (300 images, 150 chats).

**What we measured:** end-to-end latency, generator calls and thinking tokens; Q per configuration; thinking at matched latency; the step-count effect; and the frozen configuration's held-out result.

### Latency (seconds per image, A100, model CPU offload)

| Thinking | Steps | Think | Generate | Total | Generator calls |
|---|---|---|---|---|---|
| off | 20 | 0 | 49.9 | 50.2 | 40 |
| off | 28 | 0 | 65.4 | 65.8 | 56 |
| off | 50 | 0 | 109.0 | 109.4 | 100 |
| on | 20 | 1.39 | 49.8 | 51.6 | 40 |
| on | 28 | 1.39 | 65.5 | 67.4 | 56 |
| on | 50 | 1.39 | 108.9 | 110.9 | 100 |

Thinking costs less than one denoising step (about 2 s), so the matched-latency pairs are thinking on vs off at the **same** step count.

### Quality (K-dev, guidance 6)

| Thinking | 20 steps | 28 steps | 50 steps |
|---|---|---|---|
| off | 54.0 | 55.9 | 53.9 |
| on | 57.2 | **59.5** | 58.6 |

| Comparison | A − B [95% CI] | Wilcoxon p | Latency A − B |
|---|---|---|---|
| Thinking on − off, 20 steps | +3.19 [−0.79, +7.29] | 0.088 | +1.5 s |
| Thinking on − off, 28 steps | +3.59 [+0.14, +7.13] | 0.110 | +1.6 s |
| Thinking on − off, 50 steps | +4.70 [+1.11, +8.47] | 0.036 | +1.5 s |
| 28 − 20 steps, thinking off / on | +1.92 / +2.31 (both CIs include 0) | 0.065 / 0.109 | +15.7 s |
| 50 − 28 steps, thinking off / on | −2.04 / −0.93 (both CIs include 0) | 0.072 / 0.668 | +43.5 s |
| Thinking on at 20 steps − off at 50 steps | +3.31 [−0.72, +7.36] | 0.114 | −57.8 s |

### Frozen configuration and K-test

- **Rule:** the highest mean K-dev Q, fixed before any K-test run. **Selected: thinking on, 28 steps, guidance 6, reflection off.** The margin over the runner-up (thinking on, 50 steps) is only 0.93, so it is not claimed to be significantly better.
- **K-test result (300 examples, run once):**

| Subset | n | Q [95% CI] | VC | VQ | IF | KP |
|---|---|---|---|---|---|---|
| All | 300 | **58.8 [55.8, 61.6]** | 3.57 | 4.17 | 2.79 | 2.29 |
| Factual | 100 | 62.2 [57.6, 67.2] | 3.29 | 4.49 | 2.69 | – |
| Conceptual | 100 | 63.1 [57.8, 68.3] | 3.88 | 4.19 | 3.24 | 2.79 |
| Procedural | 100 | 50.9 [46.2, 55.7] | 3.53 | 3.82 | 2.45 | 1.49 |

The K-test score (58.8) matches K-dev (59.5), so the selection did not overfit K-dev. Procedural tasks are the hardest (descriptive only). Report this separately from RQ1.1, which uses default settings.

### What it means

- The best use of extra compute is thinking: about 1.4 s for roughly +3 to +5 Q.
- Extra denoising steps add up to 60 s and no measurable quality. 28 steps is enough.

**Files**
- `outputs/rq2_2/analysis/rq2_2_latency.md` / `.csv`: latency breakdown, peak VRAM, generator calls, thinking tokens
- `outputs/rq2_2/analysis/rq2_2_summary.md` / `.csv`: configuration means
- `outputs/rq2_2/analysis/rq2_2_stats.md` / `.csv`: all paired comparisons
- `outputs/rq2_2/analysis/rq2_2_frozen_config.json`: the frozen configuration and the rule used
- `outputs/rq2_2/analysis/rq2_2_ktest_summary.md` / `.csv`: K-test result
- `outputs/rq2_2/analysis/figures/rq2_2_quality_latency.png`: quality vs latency, matched-latency pairs marked
- Judge scores: `outputs/rq2_2/judge/chatgpt_main/scores_main.jsonl` and `outputs/rq2_2_ktest/judge/chatgpt_main/scores_main.jsonl`
- Images and logs: `outputs/rq2_2/`, `outputs/rq2_2_timing/` and `outputs/rq2_2_ktest/`, with the matching `logs/` folders

---

## RQ2.3 Is feedback better than just trying again?

**What we ran:** K-small (40) × seeds 42, 1234 and 7 (120 example–seed runs), using the frozen setup: thinking on (cached plan), guidance 6, 28 steps.

- **Candidate 1** is a normal edit, shared by both strategies (verified pixel-identical).
- **Feedback:** Step1X's own reflection checks candidate 1, with `max_try_cnt=2`. If it rejects the image, Step1X makes candidate 2 in one of two ways:
  - **refine:** edit candidate 1 with a corrected prompt;
  - **restart:** redraw from the source image.
- **Resampling:** if (and only if) feedback made a second candidate, redraw from the same plan with new noise. This keeps the candidate count equal per example.
- **Selection:** in each strategy, EditReward-MiMo-VL-7B, a reward model from a different family than the judge, picks the better of its two candidates using the original KRIS instruction. Step1X's own internal pick is reported as a second view of feedback.
- **Judging:** the judge scored every distinct image once, without knowing which strategy made it: 266 images, 133 chats.

**What we measured:** the Q of each strategy's final pick; latency including thinking, reflection and selection; how often the second candidate beat the first; and how often each selection rule kept the image the judge preferred. Seeds are averaged within each example first, so n = 40.

### Results

Second attempts happened in 73 of 120 runs (61%): 39 refine and 34 restart.

| Strategy | Q [95% CI] | Candidates | Latency per example |
|---|---|---|---|
| First candidate only (no retry) | 53.98 [46.7, 61.4] | 1.00 | 67.6 s |
| Resampling (EditReward picks) | **54.62** [47.3, 62.1] | 1.61 | 107.9 s |
| Feedback (EditReward picks) | 53.59 [46.4, 61.1] | 1.61 | 140.4 s |
| Feedback (Step1X's own pick) | 53.61 [46.5, 60.9] | 1.61 | 139.9 s |

| Comparison (paired) | A − B [95% CI] | Wilcoxon p | Extra latency |
|---|---|---|---|
| Feedback − resampling (both EditReward) | −1.02 [−2.45, +0.26] | 0.217 | +32.5 s |
| Feedback (Step1X pick) − resampling | −1.01 [−2.85, +0.83] | 0.360 | +32.0 s |
| Resampling − first only | +0.64 [−0.71, +2.10] | 0.614 | +40.4 s |
| Feedback − first only | −0.38 [−1.55, +0.75] | 0.679 | +72.9 s |
| Feedback (Step1X pick) − first only | −0.36 [−1.70, +0.90] | 0.717 | +72.3 s |

**Did the second candidate beat the first?** Judge Q, over the 73 runs with a retry:

| Second candidate | Beat / tie / lost | Mean Q change | EditReward kept the better | Step1X kept the better |
|---|---|---|---|---|
| Feedback, all | 15 / 30 / 28 | −3.5 | 28 / 43 | 26 / 43 |
| Feedback, refine retries | 5 / 17 / 17 | −5.1 | 18 / 22 | 13 / 22 |
| Feedback, restart retries | 10 / 13 / 11 | −1.7 | 10 / 21 | 13 / 21 |
| Resampling, all | 24 / 24 / 25 | 0.0 | 27 / 49 | – |

"Kept the better" counts only pairs the judge did not tie.

### What it means

- **Self-correction is not better than trying again.** It is about 1 Q lower (not significant) and 32 s slower per example, because of the reflection rounds.
- **A second attempt barely changes quality** in either strategy. The thinking gain from RQ2.1/2.2 is larger than any second-attempt effect here.
- **Refining usually makes the image worse.** EditReward partly rescues this by keeping candidate 1 (18 of 22 times correct). Step1X's own pick is less reliable (13 of 22).
- **Both selection methods pick the judge's preferred image only about 55–65% of the time.**
- **Budget is matched, latency is not.** The two strategies match in number of candidates and generator calls, but feedback always pays for reflection, so they cannot match in latency. The latency gap is reported instead.

**Files**
- `outputs/rq2_3/analysis/rq2_3_summary.md` / `.csv`: strategy means, latency breakdown, by knowledge type and by seed
- `outputs/rq2_3/analysis/rq2_3_stats.md` / `.csv`: paired comparisons, including the components
- `outputs/rq2_3/analysis/rq2_3_second_candidate.csv`: did candidate 2 beat candidate 1
- `outputs/rq2_3/analysis/rq2_3_per_example.csv`: every example and seed (each candidate's Q, EditReward scores, picks, latency)
- `outputs/rq2_3/analysis/figures/rq2_3_paired_differences.png`: paired Q differences with extra latency
- Judge scores: `outputs/rq2_3/judge/chatgpt_{cand1,cand2}_seed{42,1234,7}/scores_main.jsonl`
- Images: `outputs/rq2_3/{cand1,feedback_cand2,resample_cand2}/`. Generation logs: `logs/rq2_3/`. Selector logs: `logs/rq2_3/selector/`

# **RQ2 — Guidance and Steps**

**Question RQ2 answers:** How do the two dials (guidance and number of steps) behave, does "thinking" change how you should set them, and when a second attempt is allowed, is fixing the first attempt better than just trying again?

RQ2 has three sub-questions: 2.1 (guidance), 2.2 (steps vs thinking), 2.3 (feedback vs resampling). Do them in that order, because 2.1 picks the guidance value that 2.2 needs.

---

## **Before running anything**

Confirm the shared setup exists:

1. **Splits exist and are frozen.** `splits.json` with K-dev (90, 30 per type), K-test (300), K-small (40), sampled with a fixed seed, committed, never regenerated. RQ2 lives on **K-dev** for 2.1 and 2.2, and K-small for 2.3.
2. **The pipeline runs.** Code that loads Step1X and UniReason, runs one edit, saves the image, and scores it with the frozen GPT-4o KRIS judge (S2).
3. **Logging is in place (S6).** One row per generation, capturing config, seeds, the thought text and its token count, per-stage timing, and peak VRAM. A missing field means re-running, so check before scaling.
4. **Determinism check passed (S7).** Same seed must give the same image. Every RQ2 comparison assumes this.
5. **Offload mode fixed (S5).** Model-level offloading, the same mode for every condition, stated in every timing table.

**The one mechanic that makes or breaks RQ2:** the _replay_. In 2.1 and 2.3, change only the dial being studied and hold everything else identical. For each example, generate the thought once, cache it (thought text plus the exact rewritten prompt), and reuse that same cached thought and the same starting noise across every guidance value and every arm. If the thought or the noise changes between runs, the comparison is worthless. Even when replaying a cached thought, still charge one thinking pass in the latency (S4, Shared efficiency).

---

## **RQ2.1 — Does thinking change guidance sensitivity?**

**Idea in one line:** turn the "obey the instruction" dial up and down, with thinking off vs on, and see whether thinking makes the model less dependent on a high dial.

**Fixed for this sub-question:** thinking on/off, correction/reflection **off**, **50 steps**, K-dev (90 examples), single seed 42\.

**Guidance grid:**

- Step1X: `true_cfg_scale` in {2, 6, 10}. Optional if time: add {4, 8}.
- UniReason: text guidance in {2, 4, 6}, image guidance pinned at 2\. Optional: add {3, 5}.

**Exact steps:**

1. For each of the 90 K-dev examples, generate the thinking rewrite **once** per model (thinking on) and cache it. Keep the raw instruction for the thinking-off runs.
2. For each model, run the diffusion edit at every guidance value, twice: once thinking-off (raw instruction) and once thinking-on (cached rewrite). Reuse the identical starting noise across all guidance values within an example.
    - That is 90 examples × 2 (thinking off/on) × 3 guidance \= 540 generations per model, \~1080 across both (more with the optional guidance points).
3. Score every output with the frozen judge. Record the primary KRIS aggregate **Q**, and separately Instruction Following, Visual Consistency, Visual Quality, and Knowledge Plausibility (S2).
4. Log one row per generation (S6).

**What it produces:**

- Two curves per model, plotted **separately**: an _editing_ curve (Instruction Following vs guidance) and a _consistency/preservation_ curve (Visual Consistency vs guidance). Draw thinking-off and thinking-on as separate lines on each.
- A short written read of each: as guidance rises, editing usually climbs and consistency usually drops. Does the thinking-on line sit higher, or flatter (less sensitive to the dial)?
- **The guidance value to carry into 2.2**, chosen on K-dev only. TOne value per thinking mode, i.e. two per model.(highest mean Q, and keep the default if the winner beats it by less than the MMD)

**Rules and caveats to state:**

- With only three points, this reports _sensitivity_ (steep or flat), not an optimal value.
- Guidance numbers are not comparable across the two models, so never treat "Step1X 6 vs UniReason 4" as equivalent.
- Record what each model's unconditional (guidance) branch contains. If the thought text is dropped together with the instruction in that branch, text guidance is scaling instruction \+ thought jointly. State this, because it changes what the dial means.

---

## **RQ2.2 — Where is extra compute best spent, steps or thinking?**

**Idea in one line:** both "more steps" and "turn on thinking" cost time and both can help. At equal time, which buys more quality?

**Fixed for this sub-question:** the guidance value selected in 2.1 (per model and thinking mode), correction off, K-dev.

**Step grid:** {20, 28, 50} steps. (28 is included to link with RQ3's step count;)

**Exact steps:**

1. For each model, run thinking-off and thinking-on at each step count, at the 2.1-selected guidance.
    - The 50-step runs at that guidance already exist from 2.1, so **reuse them**. Only newly generate 20-step and 28-step: roughly 90 × 2 (thinking) × 2 (new step counts) \= 360 new generations per model.
2. Measure end-to-end latency properly (Shared efficiency): warm up first, call `torch.cuda.synchronize()` before reading each timer, take the median of at least 3 timing repeats, and break time into preprocess \+ think \+ generate \+ postprocess. A 2x generator speedup is not a 2x pipeline speedup once thinking time is included.
3. **Freeze one configuration per model** (a specific thinking-mode \+ step-count), chosen on K-dev, then run that frozen config **once on K-test (300)**. This is the only place RQ2 touches K-test.

**What it produces:**

- A **quality vs end-to-end latency** plot, one point per setup (each model, thinking off/on, each step count). Quality is the KRIS aggregate Q.
- **Matched-latency comparison:** explicitly line up pairs that cost about the same wall-clock time, for example thinking-on @ 20 steps vs thinking-off @ 50 steps, and state which wins at equal cost. This is the actual answer to the sub-question.
- The frozen-config K-test number, reported **separately** from RQ1.1's number (different settings, so not the same result).
- Generator calls and reasoning-token counts alongside each point.

**Caveat to state:** fewer sampling steps is _not_ distillation. This trades steps for speed on the same model, not compression.

---

## **RQ2.3 — Is feedback better than just trying again? (Priority P2, do after 2.1/2.2)**

**Idea in one line:** given two attempts, is it better to look at the first result and correct it, or to ignore it and generate a fresh independent attempt, then keep the better of the two?

**Fixed for this sub-question:** K-small (40), thinking on with the frozen plan, same selected guidance.

**The two arms (both share the same first candidate):**

- **Feedback arm:** candidate 1, then the model's correction of candidate 1 as candidate 2\.
- **Resampling arm:** candidate 1, then a fresh redraw from the _same cached plan_ with a new diffusion seed as candidate 2\.

**Exact steps:**

1. Generate candidate 1 once per example (shared by both arms).
2. Feedback arm: run the model's native correction to get candidate 2\. Resampling arm: redraw from the same frozen plan with a new seed to get candidate 2\.
    - Budget-matching detail: when Step1X's reflection accepts candidate 1, both arms stop at one candidate, so their budgets match per example. UniReason always produces a second candidate in both arms.
3. Pick the better candidate in each arm using a **fixed selector model from a different model family than the GPT-4o judge (Can use EditReward-MiMo-VL-7B)**, to avoid correlated bias. Use the same selector for both arms.  
   [https://huggingface.co/TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508](https://huggingface.co/TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508)
4. Also report the feedback arm using its own native internal selection, as a second view.
5. Score the selected outputs with the frozen judge.

**What it produces:**

- A comparison of feedback vs resampling on quality, at **matched candidate budget and matched actual latency** (not just "two attempts"). Charge thinking, correction, and selection time.
- A note on how often each strategy's second candidate actually beat the first.

**Hard rule:** never select candidates using the final benchmark score. That is peeking at the answer key. The selector must be independent of the judge.

---

## **Deliverables checklist**

Plots:

- \[ \] 2.1 guidance curves: editing and consistency, separate, thinking off vs on, per model.
- \[ \] 2.2 quality vs end-to-end latency scatter, with matched-latency pairs called out.
- \[ \] (2.3) feedback vs resampling comparison at matched budget.

Tables:

- \[ \] 2.1 selected guidance value per model (chosen on K-dev).
- \[ \] 2.2 frozen-config K-test result, reported separately from RQ1.1.
- \[ \] Latency breakdown (preprocess / think / generate / postprocess) and peak VRAM for every configuration.

Stats (S8) on every headline comparison:

- \[ \] Per-example differences averaged, with 95% paired bootstrap CIs (10,000 resamples), Wilcoxon signed-rank as a check.
- \[ \] Do not claim any difference smaller than the MMD (the judge's re-scoring spread from Validation).
- \[ \] Per-category results are descriptive only, no category-level significance claims.

---

## **Non-negotiable rules (breaking any of these invalidates the numbers)**

1. Tune only on K-dev. Do not touch K-test until the config is frozen. K-test is used exactly once, in 2.2.
2. Change only the dial under study. Reuse the same cached thought and the same starting noise across guidance values (2.1) and across arms (2.3).
3. Always pass every flag explicitly: `enable_thinking_mode`, `enable_reflection_mode`, `max_try_cnt`, `num_inference_steps`, `true_cfg_scale`. Do not rely on defaults (Step1X defaults reflection to True and steps to 28).
4. Even when replaying a cached thought, charge one thinking pass in the latency.
5. Same judge, same offload mode, same seed (42 for single-seed runs) across every condition being compared.
6. One person owns the scoring script. Nobody computes scores with their own code.

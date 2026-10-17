# **Task: Knowledge-intensive, instruction-based image editing**

**Revision notes (v2).** **\[NEW\]** \= added, **\[REV\]** \= revised wording. Untagged text is unchanged from v1. Remove the tags before submission.

1\. New **Setup and protocol (S1–S8)** section: splits, judge, eligible categories, run flags and seeds, offloading, logging, determinism check, statistics. S4, S6 and S7 carry over Part 2 items 1.5–1.8 and 2.2.

2\. Row-level fixes to RQ1.1–Optional, based on the review and on checks of the Step1X, RegionE, UniReason and KRIS code.

3\. RQ3 details: pinned CFG and RegionE settings, equivalence margin, how to switch off each RegionE component, timing protocol.

4\. New coverage table mapping experiments to the required report categories.

 

# **Setup and protocol \[NEW\]**

Complete S1–S8 before any scaled run. S4, S6 and S7 carry over Tab 2 items 1.5–1.8 and 2.2.

## **S1 — Data splits**

 Eligible pool: KRIS-Bench after the exclusions in S3. **Primary grouping: the three knowledge types** (Factual, Conceptual, Procedural). Report the seven reasoning dimensions descriptively only.

| Split | Proposed size | Sampling | Used for |
| :---- | :---- | :---- | :---- |
| **K-dev** | 90 (30 per type) | Stratified; **disjoint from K-test** | Pilot, all guidance and step selection (RQ2.1, RQ2.2), S7 |
| **K-test** | 300 (100 per type) | Stratified | RQ1.1, frozen-configuration evaluation (RQ2.2) |
| **K-small** | 40 (13–14 per type) | Stratified subset of K-test | Three-seed runs, RQ1.2, RQ1.3, RQ2.3, RQ3.1, Validation |
| **RQ3.2 subset** | 20 | Stratified subset of K-small | Component ablation |

● If a type has too few eligible items, allocate proportionally and state it.

● Sample with a fixed seed. Save all IDs to splits.json, commit it, and never regenerate it.

● Choose K-small and the RQ3.2 subset before any outputs are seen.

● No tuning or selection on K-test.

## **S2 — Judge**

● Use the **official KRIS-Bench scripts** (mercurystraw/Kris\_Bench) with GPT-4o at a **pinned snapshot**, temperature 0, and the unmodified prompts.

● Use the same evaluator for both models. Do **not** use UniReason's modified "faster" evaluator.

● Save version, temperature and prompts in judge\_config.yaml; commit it and never change it.

● Store every raw judge response.

● If the snapshot becomes unavailable, choose one fixed substitute for the whole project and do not compare against published numbers.

● **Primary score Q** \= the official KRIS aggregate. Also report Visual Consistency, Visual Quality, Instruction Following and Knowledge Plausibility (where applicable) separately.

● Score all conditions of one experiment in the same batch.

## **S3 — Eligible categories**

Step1X accepts a single reference image. Exclude multi-image tasks (e.g., multi-element composition) for **both** models. Check every category's input count during setup, and report the excluded categories and the resulting pool size.

## **S4 — Run flags and seeds**

● **Base seeds:** 42, 1234, 7\. Use 42 for every single-seed run so single-seed results are directly comparable.

● **Derived seeds:** separate, deterministic seeds per stage (thinking, initial diffusion, each retry), derived from (base seed, example ID, stage).

● **Thinking is sampled, not greedy.** The Step1X thinker uses T \= 0.7, top-p 0.9 and ≤ 128 new tokens, and draws from the **global torch RNG**, not the pipeline generator. Call torch.manual\_seed(thinking\_seed) before each call. UniReason's thinking also samples (T \= 0.3); treat it the same way.

● **Replay:** cache every thought/rewrite. Wherever the design fixes the thought (RQ2.1, RQ2.3, RQ3.1), pass the cached rewrite as the prompt with thinking off, and charge one thinking pass in latency (see Shared efficiency analysis).

● **Explicit flags:** always pass enable\_thinking\_mode, enable\_reflection\_mode (defaults to True), max\_try\_cnt, num\_inference\_steps (code default 28; README uses 50\) and true\_cfg\_scale.

## **S5 — Deployment and offloading**

● Use **model-level** offloading (swap thinker ↔ generator), not per-layer sequential offloading, which makes latency measure PCIe traffic and can hide RegionE's speedup.

● Keep the generator resident on the GPU during diffusion if it fits.

● Time offload transfers separately.

● Use the same offload mode across all compared conditions, and state it in every latency table.

 

**S6 — Logging schema**

One row per generation. Build it before the first run; adding fields later means re-running.

● **Run:** example ID, model, condition, full config, base seed, derived seeds.

● **Candidates:** output paths for every candidate, selected-candidate index, attempt count, stop/retry decisions.

● **Text:** rewrite/thought text and its token count, reflection/critique text.

● **Timing:** per stage (preprocess, thinking, diffusion per attempt, reflection per round, postprocess, offload transfers) plus directly measured end-to-end time.

● **Resources:** peak VRAM.

● **RegionE (when on):** ρₜ, cache-reuse and refresh events, full-image and reduced-token generator calls.

● **Other:** failures and retries, raw judge output.

## **S7 — Determinism check**

● On 10 K-dev examples per model, regenerate with identical config and seeds.

● Enable PyTorch's deterministic algorithms.

● Require identical rewrites and bitwise-identical images. If bitwise identity is impossible on this hardware, require a mean absolute pixel difference \< 1 (0–255 scale) and state the tolerance.

● If the check fails, fix it before scaling: every paired comparison assumes that the same seed gives the same noise.

## **S8 — Statistics**

● Compute differences per example, then average.

● Report 95% paired bootstrap CIs (10,000 resamples over examples), with a Wilcoxon signed-rank test as a check.

● Three-seed K-small results: average over seeds within each example first, and report them separately from K-test.

● **Minimum meaningful difference (MMD):** the judge's re-scoring spread from Validation. Do not claim differences smaller than the MMD.

● **RQ3.1 equivalence margin δ:** fixed before test scoring, with δ ≥ MMD.

● Per-category results are descriptive; make no category-level significance claims.

# **Research Questions and Experiments**

| RQ / sub-RQ | Experiments: models, configurations and dataset | Priority | Remarks / analysis outputs |
| :---- | :---- | :---- | :---- |
| **Setup: Deployment and protocol** | **Both models:** pilot 20 K-dev examples, including thinking and correction. Start at **50 denoising steps**: Step1X true\_cfg\_scale=6; UniReason text guidance **4**, image guidance **2**. Disable RegionE. Pin revisions, precision, offloading and common supported resolution. [GitHub](https://github.com/stepfun-ai/Step1X-Edit) **\[NEW\]** Pass every step count and mode flag explicitly (Step1X's code default is 28 steps, not 50). Follow Setup S1–S8. | **P1** | Verify each mode on the **40GB GPU** before scaling. Measure full-pipeline latency before estimating GPU-hours. BAGEL remains a fallback, not another full experiment track. **\[NEW\]** Pass the determinism check (S7) and the judge repeatability check (Validation) before any K-test scoring. |
| **RQ1 — Reasoning contribution 1.1 Does thinking improve editing in both models?** | **Step1X \+ UniReason, K-test:** direct editing versus thinking-only; correction off. Keep checkpoint, guidance and sampling fixed within each model. One seed on K-test; **three seeds total on K-small** **\[REV\]** (42, 1234, 7; S4). **\[NEW\]** Set enable\_reflection\_mode=False explicitly; Step1X defaults it to True. | **P1** | Report paired thinking gains by category and a shared quality–cost table. Cross-model results compare systems; within-model ablations test inference-time thinking, not reasoning-related training. **\[NEW\]** Paired bootstrap CIs (S8). Per-category results are descriptive. This is the headline K-test result. |
| **1.2 Does post-edit correction help?** | **Both models, K-small:** thinking-only versus thinking plus one correction opportunity. Step1X: max\_try\_cnt=2 total attempts. UniReason: its refinement procedure, capped at two generated candidates. Save first, corrected and selected outputs. [GitHub](https://raw.githubusercontent.com/Peyton-Chen/diffusers/step1xedit_v1p2/src/diffusers/pipelines/step1x_edit/pipeline_step1x_edit_v1p2.py) | **P1** | Report correction/degradation rates, preservation, actual attempts and latency. Native correction/selection procedures differ. Select both set of examples where correction helped and did not. **\[NEW\]** Step1X retries only when its own reflection judges the first output a failure. The retry edits the *previous output* with a refined prompt, and the final image is picked by internal scores. UniReason's refinement always runs a second pass. Report the trigger rate, actual attempts, and how often the selected output ≠ the last output. |
| **1.3 Where does reasoning fail?** | Reuse **both models' K-small outputs**. Two reviewers label observable plans and images: wrong knowledge, omitted constraints, correct plan/wrong image, unnecessary changes and mistaken critique. Allow multiple labels. **\[NEW\]** Pilot a written codebook on 10 items before labelling. Judge plan correctness against the KRIS knowledge hints. Optional: extend plan/image labels to K-test thinking-only outputs (no new generations) to fill out the heatmap. | **P1** | No new generations. Produce a category heatmap and annotated success/failure grids. Distinguish interpretation failures from execution failures; **\[REV\]** report per-label Cohen's κ. **\[NEW\]** Step1X rewrites are capped at 128 tokens, so knowledge errors may be under-observable; state this. |
| **RQ2 — Guidance and compute 2.1 Does thinking change guidance sensitivity?** | **Both models, K-dev:** thinking off/on; correction off; 50 steps. Step1X CFG **{2,6,10}**; UniReason text guidance **{2,4,6}**, image guidance fixed at **2**. Hold noise fixed and replay the same thought/rewrite across guidance values. UniReason exposes separate text/image guidance controls. [GitHub](https://raw.githubusercontent.com/AlenjandroWang/UniReason/main/UniReason/eval/gen/gen_images_mp_kris.py) **\[NEW\]** If budget allows, add Step1X CFG {4, 8} and UniReason text guidance {3, 5}. Record what each model's unconditional (guidance) branch contains. | **P1** | Plot editing and consistency curves separately. These are proposed grids, not universal optima. Guidance values are not equivalent across architectures. Select settings on K-dev only. **\[NEW\]** With three points, report *sensitivity*, not a located optimum. If the thought text is dropped together with the instruction in the unconditional branch, text guidance scales instruction \+ thought jointly; state this. |
| **2.2 Where is extra compute best spent?** | **Both models, K-dev:** thinking off/on at **\[REV\] {20, 28, 50} steps** (28 links to RQ3), using corresponding development-selected guidance. Reuse identical prior runs. Freeze one selected configuration per model and evaluate on **K-test**. | **P1** | Plot quality versus end-to-end latency. Include correction configurations only on the common K-small subset. Report generator calls and reasoning tokens. Fewer sampling steps are not distillation. **\[NEW\]** Explicitly compare matched-latency pairs (e.g., thinking@20 vs direct@50). Report the frozen-configuration K-test result separately from RQ1.1's default-setting result. |
| **2.3 Is feedback better than extra attempts?** | **Both models, K-small:** compare a two-candidate correction pipeline with two independent draws from the same frozen plan/instruction. Apply the same fixed selector to both candidate pools, separate from the final evaluation judge. **\[NEW\]** Both arms share the *same first candidate*. The second candidate is either the correction (feedback arm) or a redraw from the same frozen plan with a new diffusion seed (resampling arm). When Step1X's reflection accepts the first candidate, both arms stop at one candidate, so budgets match per example. UniReason always produces a second candidate in both arms. Also report the feedback arm with its native internal selection. | **P2** | Control candidate budget and charge thinking, feedback and selection costs. Compare actual latency, not just attempts. Never select candidates using final benchmark scores. **\[NEW\]** Use a selector from a different model family than the judge to avoid correlated bias. |
| **RQ3 — Reasoning with acceleration 3.1 Does RegionE preserve thinking gains?** | **Step1X only, K-small:** thinking off/on × RegionE off/on—**four conditions**. Correction off; identical guidance and **28-step schedule** **\[NEW\]** (true\_cfg\_scale=6; RegionE set\_params() defaults). Fix thought and initial noise within each pair. The inspected RegionE implementation requires 28 steps **\[REV\]** (the code asserts num\_inference\_steps \== 28; its velocity-decay factors are calibrated for 28). [GitHub](https://raw.githubusercontent.com/Peyton-Chen/RegionE/main/RegionE/Step1XEditV1P2/utils.py) | **P2** | Compare absolute quality loss and the change in thinking gain, with paired intervals. Report generator and full-pipeline speedups. Similarity to vanilla alone is not edit correctness. **\[NEW\]** Report CI widths; expand to 80 examples if pilot latency allows. Equivalence margin fixed in advance (S8). |
| **3.2 Which acceleration components matter?** | **Step1X, 20 predetermined K-small examples:** thinking on; 28 steps. Compare no RegionE, full RegionE, **RIKVCache off**, and **AVDCache off**. Reuse compatible RQ3.1 outputs; log selected-token fraction, cache activity and generator calls. **\[NEW\]** AVDCache off: set cache\_threshold \< 0. RIKVCache off: no switch exists in the code, so it needs an attention-processor edit (see RQ3.2A); validate on 2–3 examples first. | **P2** | Show selected regions, output differences and failures. Add LPIPS/PSNR against vanilla alongside editing scores. Do not equate reasoning difficulty with edit size. |
| **Validation: Are the conclusions reliable?** | **Both models:** two reviewers blind-score outputs for **30 K-small examples** across core modes. Re-score 20 fixed outputs to assess judge repeatability. Report paired example-bootstrap intervals; analyse the three-seed subset separately. | **P1** | Reuse outputs. Freeze evaluator/version/rubric. Report human agreement, judge agreement and failures. Do not transfer small-subset error bars to full-test results. **\[NEW\]** Human–human: weighted κ. Human–judge: Spearman ρ and ICC. The judge's re-scoring spread sets the minimum meaningful difference (MMD, S8). Run the repeatability check before K-test scoring. |
| **Optional external validity** | **GEdit-Bench:** 80 English single-reference examples; both models direct/thinking-only, using frozen KRIS-selected settings. Alternatively, add **FLUX.2 Klein 9B at its supported four-step setting** on K-small. [GitHub](https://raw.githubusercontent.com/AlenjandroWang/UniReason/main/EVAL.md) | **P3** | Choose one extension first. FLUX provides external context, not proof of absent implicit reasoning. Discuss adaptive stopping or compatible LoRA distillation; implementation is not required. **\[NEW\]** Score GEdit-Bench with its own official evaluator. |

 

# **RQ3 — Can inference acceleration preserve the benefits of thinking?**

 

RQ3 connects understanding an edit with executing it efficiently. It asks whether acceleration preserves the improvements introduced by thinking, and which internal components explain the resulting quality–efficiency trade-off. 

The study uses a direct thinking × acceleration experiment, followed by targeted component ablations.

**Scope**: Step1X-Edit-v1p2 only. RQ1 and RQ2 provide the shared Step1X–UniReason comparison; RQ3 provides deeper analysis of one supported implementation. RegionE is a training-free acceleration framework with support for Step1X-v1p2.

**Priority**: P2 — high-value additional analysis, after the core two-model experiments.

 

## **RQ3.1 — Does RegionE preserve the performance gains from thinking?**

## **A. Experimental design**

Use K-small: 40 predetermined KRIS-Bench test examples covering **\[REV\]** all three knowledge types (S1). Evaluate all four configurations on exactly the same examples:

| Configuration | Thinking | RegionE | Purpose |
| :---- | :---- | :---- | :---- |
| **A** | Off | Off | Direct-editing reference |
| **B** | On | Off | Thinking-enabled reference |
| **C** | Off | On | Acceleration without thinking |
| **D** | On | On | Thinking and acceleration together |

**Disable reflection throughout**. This isolates the interaction between initial thinking and generator acceleration without introducing different numbers of editing attempts.

 

Keep the checkpoint, initial diffusion noise, resolution, guidance, precision, offloading and sampling schedule fixed. Generate each thinking-enabled instruction once and reuse it in B and D, so acceleration is the only difference within that pair.

 

Use 28 configured inference steps for every condition. The RegionE implementation used in this plan requires this setting because its velocity-decay calibration depends on the schedule. **\[NEW\]** The code enforces this with an assertion. The repository README shows RegionE with 50 steps; follow the code, not the README. Therefore, 50-step reference images from other experiments cannot serve as the unaccelerated controls here. **\[NEW\]** For the same reason, G₀ below is measured at 28 steps and is not directly comparable with the 50-step thinking gain from RQ1.1.

Freeze RegionE settings and a common CFG value before test evaluation. Do not separately optimize the four conditions on K-small. **\[NEW\]** Use true\_cfg\_scale=6 (the v1p2 default) and RegionE's set\_params() defaults for v1p2; record both in the config.

### 

### **B. Measure two different effects**

Let Q(A), Q(B), Q(C) and Q(D) denote the average editing scores for the four configurations. Higher scores indicate better performance. **\[NEW\]** Q is the primary KRIS score defined in S2.

Thinking gain without acceleration:

G₀ \= Q(B) − Q(A)

Thinking gain with acceleration:

G₁ \= Q(D) − Q(C)

 

The thinking × acceleration interaction is:

I \= G₁ − G₀

 

Equivalently:

I \= \[Q(D) − Q(C)\] − \[Q(B) − Q(A)\]

 

Interpretation:

I \< 0: Acceleration reduces the observed thinking gain.

I \> 0: The observed thinking gain is larger under acceleration.

I ≈ 0: An estimate near zero with a sufficiently narrow confidence interval suggests little interaction.

 

However, preserving the thinking gain is not the same as preserving overall quality. RegionE could reduce both thinking-off and thinking-on scores equally, giving I ≈ 0\.

 

Therefore, also measure the absolute quality effect of acceleration.

 

Quality change with thinking off:

ΔQ₀ \= Q(C) − Q(A)

 

Quality change with thinking on:

ΔQ₁ \= Q(D) − Q(B)

 

A negative value indicates lower quality after acceleration.

 

The principal questions are:

1\. Does thinking remain beneficial after acceleration?

2\. How much does acceleration change the quality of the thinking-enabled output?

 

Compute differences per example before averaging, and report paired confidence intervals. Predefine an acceptable quality-loss margin. A statistically nonsignificant difference alone does not establish retained quality. **\[NEW\]** Use the margin δ from S8 (δ ≥ MMD), fixed before test scoring. Conclude "quality retained" only if the lower bound of the 95% CI for ΔQ₁ is above −δ. Report the CI width for I; if it is wider than the effects of interest at n \= 40, expand to 80 examples or describe I as inconclusive.

 

If the unaccelerated thinking gain is absent or uncertain, do not describe the result as preserving a demonstrated benefit.

**C. Evaluate correctness, not just similarity**

Use the project's shared KRIS evaluation dimensions: Instruction Following, Visual Consistency, Visual Quality and Knowledge Plausibility where applicable. These distinguish whether the requested edit is correct from whether the result looks coherent. https://github.com/mercurystraw/Kris\_Bench

 

Additionally, compute PSNR, SSIM and LPIPS against the corresponding unaccelerated output:

 

● C against A: Acceleration with thinking off.

● D against B: Acceleration with thinking on.

 

These similarity measures quantify deviation from the reference pipeline; they do not establish edit correctness. A visually similar output may still contain an important error, such as an incorrect digit or spatial relationship.

 

## **RQ3.2 — Which acceleration components explain the quality–efficiency trade-off?**

### **A. Component ablation**

Use 20 predetermined examples within K-small. Keep thinking on, reflection off and the same 28-step configuration.

| Configuration | Change | Question addressed |
| :---- | :---- | :---- |
| **No RegionE** | Unaccelerated generator | What is the reference quality and cost? |
| **Full RegionE** | All acceleration components enabled | What trade-off does the complete method achieve? |
| **RegionE without RIKVCache** | Remove the cached contextual information used during regional generation | How important is retained context for accurate local editing? |
| **RegionE without AVDCache** | Disable temporal velocity-prediction reuse | How much acceleration and output deviation come from temporal reuse? |

Reuse the matching B and D outputs from RQ3.1 for Configurations 1 and 2\.

RIKVCache means Region-Instruction Key–Value Cache. It retains contextual information during regional generation.

 

AVDCache means Adaptive Velocity Decay Cache. It reuses adjusted velocity predictions across suitable denoising steps.

 

The paper studies their removal as separate ablations. In particular, its "without RIKVCache" condition removes contextual injection during local attention. It is not simply an implementation that recomputes the same context instead of caching it. Reproduce and validate this distinction. https://arxiv.org/html/2510.25590v1

 

**\[NEW\]** Implementation:

● **AVDCache off:** set cache\_threshold to a negative value. The code reuses a cached velocity only when its estimated error is ≤ the threshold, so a negative threshold disables reuse without other changes.

● **RIKVCache off:** the v1p2 code has no switch; the cached key/value tensors live in the attention processor. Implement the paper's "no contextual injection" variant there. Before running the 20 examples, check on 2–3 examples that only this component changed: the selected regions and AVDCache events should match full RegionE.

 

Hold other settings fixed. These experiments assess each component's contribution within the deployed configuration rather than independently optimizing every variant.

 

### **B. Inspect where computation is allocated**

Record the fraction of output-image tokens selected for iterative editing:

ρₜ \= Nₑ(t) / N

where:

ρₜ \= fraction of output-image tokens selected for iterative editing at stage t.

Nₑ(t) \= number of output-image tokens selected for iterative editing at stage t.

N \= total number of output-image tokens.

The implementation maintains edited and unedited token indices, allowing the selected regions to be inspected. Record the initial partition and subsequent changes rather than relying only on one final average. [https://raw.githubusercontent.com/Peyton-Chen/RegionE/main/RegionE/Step1XEditV1P2/utils.py](https://raw.githubusercontent.com/Peyton-Chen/RegionE/main/RegionE/Step1XEditV1P2/utils.py)

Alongside these regions, log cache-reuse events, refresh events, full-image and reduced-token generator calls, latency and editing-score changes.

RegionE can reuse predictions at some sampler updates, so identical configured step counts do not imply identical generator work.

The main qualitative visualization should show:

Source and instruction → Selected editing region → Unaccelerated output → Accelerated output → Difference map

This supports specific diagnoses. A required change might lie outside the selected region. Alternatively, the region may be appropriate, but cached context or predictions may be inadequate. Component ablations help distinguish these possibilities.

Treat ρₜ as an internal diagnostic, not a ground-truth edit mask. Its association with failures is informative but does not, by itself, prove their cause.

 

### **C. Separate reasoning difficulty from spatial extent**

Label reasoning requirements and expected edit extent separately, before inspecting generated outcomes.

Illustrative examples:

● "Correct the arithmetic on the sign" may require reasoning but change only one digit.

● "Turn the whole image monochrome" may be straightforward but affect almost every pixel.

 

Investigate whether failures relate to edit extent, reasoning requirements or both. Do not assume that reasoning-intensive edits are necessarily global.

 

With only 20 diagnostic examples, use these comparisons to explain observed cases and identify patterns rather than make strong category-level generalizations.

 

## **SHARED EFFICIENCY ANALYSIS**

Report generator latency and complete-pipeline latency separately, together with peak GPU memory.

 

Measure after warm-up under the same deployment settings. Include preprocessing, thinking, decoding and required transfers in the complete-pipeline accounting. **\[NEW\]** Call torch.cuda.synchronize() before reading each timer, report the median of ≥ 3 timing repeats, and state the offload mode (S5). For correction and selection pipelines, also charge reflection and selector time.

 

For thinking-enabled conditions:

 

T(total) \= T(preprocess) \+ T(think) \+ T(generate) \+ T(postprocess)

A generator speedup need not produce the same overall speedup.

 

Illustrative example:

Original pipeline:

T(original) \= 40 \+ 60 \= 100 seconds

 

After a twofold generator speedup:

T(accelerated) \= 40 \+ (60 / 2\) \= 70 seconds

 

End-to-end speedup:

S(total) \= 100 / 70 ≈ 1.43×

 

Thus, a 2× generator speedup produces only a 1.43× complete-pipeline speedup in this example.

 

When fixed rewritten instructions are replayed for experimental control, still charge one thinking pass to each thinking-enabled pipeline. Clearly distinguish reconstructed stage totals from directly measured end-to-end runs.

 

Plot editing performance against end-to-end latency for A–D. For the component study, add quality change and similarity-to-reference results. This shows whether savings remain useful after accounting for the reasoning overhead.

 

### **EXPECTED OUTPUTS AND CONTRIBUTION**

The main outputs are:

 1\. A thinking-gain and acceleration-loss comparison with confidence intervals.

2\. A quality–latency plot.

3\. An annotated region/cache failure analysis.

 

Proposed generation count:

 

RQ3.1:

40 examples × 4 configurations \= 160 generations

 

RQ3.2: 

20 examples × 2 additional component-removal variants \= 40 generations

 

Total:

160 \+ 40 \= 200 generations

 

This assumes exact reuse of the corresponding reference outputs. It excludes pilots, timing repetitions and additional seeds. **\[NEW\]** If RQ3.1 is expanded to 80 examples: 320 \+ 40 \= 360 generations.

 

The contribution is not simply that RegionE makes generation faster. It is a controlled investigation of whether computational reuse preserves the value of explicit thinking, which components are responsible for the trade-off, and when small output changes become important editing errors.

 

These findings remain useful whether acceleration retains quality, selectively harms certain edits or produces an uncertain result on the available subset. No new trained model is required; the value comes from the controlled comparison and the explanation of observed behaviour.

 

# **Coverage of required categories \[NEW\]**

 

| Required category | Experiments covering it |
| :---- | :---- |
| **Hyper-parameter analysis** | RQ2.1 (guidance), RQ2.2 (steps) |
| **Key component analysis** | RQ1.1 (thinking), RQ1.2 (correction), RQ3.2 (RIKVCache, AVDCache) |
| **Comparison between methods** | RQ1–RQ2 (Step1X vs UniReason), RQ2.3 (feedback vs resampling), RQ3.1 (accelerated vs unaccelerated), Optional (FLUX.2 / GEdit) |
| **Strengths and limitations** | RQ1.3 (failure taxonomy), RQ3.2B–C (region/cache diagnoses), Validation (human–judge agreement) |
| **Visualization of results** | RQ1.1 per-category gains; RQ1.3 heatmap and success/failure grids; RQ2.1 guidance curves; RQ2.2 and RQ3.1 quality–latency plots; RQ3.2 region/difference strips; Validation human–judge scatter |
| **Future extension** | Optional row: adaptive stopping or LoRA distillation (discussion only) |

 

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

**The one mechanic that makes or breaks RQ2:** the *replay*. In 2.1 and 2.3, change only the dial being studied and hold everything else identical. For each example, generate the thought once, cache it (thought text plus the exact rewritten prompt), and reuse that same cached thought and the same starting noise across every guidance value and every arm. If the thought or the noise changes between runs, the comparison is worthless. Even when replaying a cached thought, still charge one thinking pass in the latency (S4, Shared efficiency).

---

## **RQ2.1 — Does thinking change guidance sensitivity?**

**Idea in one line:** turn the "obey the instruction" dial up and down, with thinking off vs on, and see whether thinking makes the model less dependent on a high dial.

**Fixed for this sub-question:** correction/reflection **off**, **50 steps**, K-dev (90 examples), single seed 42\.

**Guidance grid:**

* Step1X: `true_cfg_scale` in {2, 6, 10}. Optional if time: add {4, 8}.  
* UniReason: text guidance in {2, 4, 6}, image guidance pinned at 2\. Optional: add {3, 5}.

**Exact steps:**

1. For each of the 90 K-dev examples, generate the thinking rewrite **once** per model (thinking on) and cache it. Keep the raw instruction for the thinking-off runs.  
2. For each model, run the diffusion edit at every guidance value, twice: once thinking-off (raw instruction) and once thinking-on (cached rewrite). Reuse the identical starting noise across all guidance values within an example.  
   * That is 90 examples × 2 (thinking off/on) × 3 guidance \= 540 generations per model, \~1080 across both (more with the optional guidance points).  
3. Score every output with the frozen judge. Record the primary KRIS aggregate **Q**, and separately Instruction Following, Visual Consistency, Visual Quality, and Knowledge Plausibility (S2).  
4. Log one row per generation (S6).

**What it produces:**

* Two curves per model, plotted **separately**: an *editing* curve (Instruction Following vs guidance) and a *consistency/preservation* curve (Visual Consistency vs guidance). Draw thinking-off and thinking-on as separate lines on each.  
* A short written read of each: as guidance rises, editing usually climbs and consistency usually drops. Does the thinking-on line sit higher, or flatter (less sensitive to the dial)?  
* **The guidance value to carry into 2.2**, chosen on K-dev only. TOne value per thinking mode, i.e. two per model.(highest mean Q, and keep the default if the winner beats it by less than the MMD)

**Rules and caveats to state:**

* With only three points, this reports *sensitivity* (steep or flat), not an optimal value.  
* Guidance numbers are not comparable across the two models, so never treat "Step1X 6 vs UniReason 4" as equivalent.  
* Record what each model's unconditional (guidance) branch contains. If the thought text is dropped together with the instruction in that branch, text guidance is scaling instruction \+ thought jointly. State this, because it changes what the dial means.

---

## **RQ2.2 — Where is extra compute best spent, steps or thinking?**

**Idea in one line:** both "more steps" and "turn on thinking" cost time and both can help. At equal time, which buys more quality?

**Fixed for this sub-question:** the guidance value selected in 2.1 (per model and thinking mode), correction off, K-dev.

**Step grid:** {20, 28, 50} steps. (28 is included to link with RQ3's step count;)

**Exact steps:**

1. For each model, run thinking-off and thinking-on at each step count, at the 2.1-selected guidance.  
   * The 50-step runs at that guidance already exist from 2.1, so **reuse them**. Only newly generate 20-step and 28-step: roughly 90 × 2 (thinking) × 2 (new step counts) \= 360 new generations per model.  
2. Measure end-to-end latency properly (Shared efficiency): warm up first, call `torch.cuda.synchronize()` before reading each timer, take the median of at least 3 timing repeats, and break time into preprocess \+ think \+ generate \+ postprocess. A 2x generator speedup is not a 2x pipeline speedup once thinking time is included.  
3. **Freeze one configuration per model** (a specific thinking-mode \+ step-count), chosen on K-dev, then run that frozen config **once on K-test (300)**. This is the only place RQ2 touches K-test.

**What it produces:**

* A **quality vs end-to-end latency** plot, one point per setup (each model, thinking off/on, each step count). Quality is the KRIS aggregate Q.  
* **Matched-latency comparison:** explicitly line up pairs that cost about the same wall-clock time, for example thinking-on @ 20 steps vs thinking-off @ 50 steps, and state which wins at equal cost. This is the actual answer to the sub-question.  
* The frozen-config K-test number, reported **separately** from RQ1.1's number (different settings, so not the same result).  
* Generator calls and reasoning-token counts alongside each point.

**Caveat to state:** fewer sampling steps is *not* distillation. This trades steps for speed on the same model, not compression.

---

## **RQ2.3 — Is feedback better than just trying again? (Priority P2, do after 2.1/2.2)**

**Idea in one line:** given two attempts, is it better to look at the first result and correct it, or to ignore it and generate a fresh independent attempt, then keep the better of the two?

**Fixed for this sub-question:** K-small (40), thinking on with the frozen plan, same selected guidance.

**The two arms (both share the same first candidate):**

* **Feedback arm:** candidate 1, then the model's correction of candidate 1 as candidate 2\.  
* **Resampling arm:** candidate 1, then a fresh redraw from the *same cached plan* with a new diffusion seed as candidate 2\.

**Exact steps:**

1. Generate candidate 1 once per example (shared by both arms).  
2. Feedback arm: run the model's native correction to get candidate 2\. Resampling arm: redraw from the same frozen plan with a new seed to get candidate 2\.  
   * Budget-matching detail: when Step1X's reflection accepts candidate 1, both arms stop at one candidate, so their budgets match per example. UniReason always produces a second candidate in both arms.  
3. Pick the better candidate in each arm using a **fixed selector model from a different model family than the GPT-4o judge (Can use EditReward-MiMo-VL-7B)**, to avoid correlated bias. Use the same selector for both arms.  
   [https://huggingface.co/TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508](https://huggingface.co/TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508)  
4. Also report the feedback arm using its own native internal selection, as a second view.  
5. Score the selected outputs with the frozen judge.

**What it produces:**

* A comparison of feedback vs resampling on quality, at **matched candidate budget and matched actual latency** (not just "two attempts"). Charge thinking, correction, and selection time.  
* A note on how often each strategy's second candidate actually beat the first.

**Hard rule:** never select candidates using the final benchmark score. That is peeking at the answer key. The selector must be independent of the judge.

---

## **Deliverables checklist**

Plots:

* \[ \] 2.1 guidance curves: editing and consistency, separate, thinking off vs on, per model.  
* \[ \] 2.2 quality vs end-to-end latency scatter, with matched-latency pairs called out.  
* \[ \] (2.3) feedback vs resampling comparison at matched budget.

Tables:

* \[ \] 2.1 selected guidance value per model (chosen on K-dev).  
* \[ \] 2.2 frozen-config K-test result, reported separately from RQ1.1.  
* \[ \] Latency breakdown (preprocess / think / generate / postprocess) and peak VRAM for every configuration.

Stats (S8) on every headline comparison:

* \[ \] Per-example differences averaged, with 95% paired bootstrap CIs (10,000 resamples), Wilcoxon signed-rank as a check.  
* \[ \] Do not claim any difference smaller than the MMD (the judge's re-scoring spread from Validation).  
* \[ \] Per-category results are descriptive only, no category-level significance claims.

---

## **Non-negotiable rules (breaking any of these invalidates the numbers)**

1. Tune only on K-dev. Do not touch K-test until the config is frozen. K-test is used exactly once, in 2.2.  
2. Change only the dial under study. Reuse the same cached thought and the same starting noise across guidance values (2.1) and across arms (2.3).  
3. Always pass every flag explicitly: `enable_thinking_mode`, `enable_reflection_mode`, `max_try_cnt`, `num_inference_steps`, `true_cfg_scale`. Do not rely on defaults (Step1X defaults reflection to True and steps to 28).  
4. Even when replaying a cached thought, charge one thinking pass in the latency.  
5. Same judge, same offload mode, same seed (42 for single-seed runs) across every condition being compared.  
6. One person owns the scoring script. Nobody computes scores with their own code.

## **~~Handoffs~~**

* ~~The guidance value selected in 2.1 should be the "fixed guidance" RQ1.1 uses on K-test.~~   
* ~~The 28-step runs feed RQ3, which is locked to 28 steps. Save them cleanly with full logging so they can be reused rather than re-run.~~

| Split | Size | Sliced from | Used for | Whose RQ |
| ----- | ----- | ----- | ----- | ----- |
| Evaluation set | \~350 | Full KRIS-Bench (\~1,267) | Main headline scores | RQ1, RQ3 (full-set runs) |
| Sweep set | \~100 | Evaluation set | Tuning/trying settings | RQ2  |
| K-small | \~40 | Evaluation set | Experiments run many times (multi-seed, reflection rounds) | RQ1 |
| RQ3.2 subset | \~20 | K-small | Deep acceleration component ablation | RQ3 |


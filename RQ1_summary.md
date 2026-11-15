# RQ1 Summary — Does Reasoning Help Image Editing?

**Project:** Reasoning-enhanced image editing on KRIS-Bench  
**Models:** Step1X-Edit v1.2 (ReasonEdit-S) and UniReason 1.0  
**Research scope:** RQ1.1 Thinking vs. Base; RQ1.2 Correction vs. Thinking; RQ1.3 qualitative diagnosis of reasoning failures  
**Status:** Main quantitative results, correction runtime/audit, and illustrative RQ1.3 pilot observations available. RQ1.3 is explicitly **qualitative only**; it is not a population-level human annotation study.  
**Prepared:** 15 November 2026

> **Bottom line:** Enabling inference-time Thinking improves Step1X editing quality by **+3.79/100** (95% CI **+1.78 to +5.84**), but yields no reliable aggregate change for UniReason (**+0.19**, CI **−2.01 to +2.43**). Additional Correction does **not** produce a reliable improvement for either model on the 40-example K-small subset (Step1X **−1.91**, UniReason **−1.56**). For Step1X, correction triggers a second generation in **69/120 runs** and raises mean logged inference time from **113.23 s to 223.48 s**, with a **2.00× mean paired runtime ratio**. Qualitative examples show interpretation, execution, and critique failures.

## 1. Research questions and experimental design

| Study | Question | Comparison | Data | Generation seeds |
|---|---|---|---|---|
| **RQ1.1** | Does thinking improve editing? | Base (thinking off; correction off) vs. Thinking (thinking on; correction off) | 300 K-test examples per model | 42 |
| **RQ1.2** | Does post-edit correction improve Thinking? | Thinking vs. Thinking + native Correction | 40 K-small examples per model | 42, 1234, 7 |
| **RQ1.3** | Where does reasoning fail? | Qualitative comparison of original, Base, Thinking, Correction, and available traces | Representative examples from K-small/blinded pilot; no prevalence study | Illustrations checked against saved examples |

### Models and what the switches actually mean

- **Step1X-Edit v1.2 / ReasonEdit-S:** a multimodal reasoner rewrites/interprets the edit request before the diffusion editor. In Correction mode its internal reflection can request one additional editing attempt; the system may select the first rather than the second candidate. It is therefore **not** a mandatory two-pass generator.
- **UniReason 1.0:** reasoning-enhanced editing with a separate released refinement procedure. Correction adds a refinement pass after the initially generated image. The two models’ correction mechanisms are **not algorithmically identical**.
- These are **within-model inference-time switches**. They do not isolate the effect of training models with versus without reasoning. Cross-model performance gaps compare two released systems and should not be interpreted as a controlled architecture ablation.

### Dataset construction and split integrity

- The KRIS-Bench subset was filtered to the examples eligible for the project’s single-reference image-editing setup (`get_rq1_eligible_dataset`); the exact internal filter predicates are defined in that loader.
- `src/create_splits.py` uses `random.Random(42)` and independently shuffles within the three knowledge strata: **factual**, **conceptual**, **procedural**. The chosen membership is then sorted and frozen in `data/splits.json`. This is **knowledge-type-stratified**, *not* category-stratified, sampling.
- **K-dev:** 90 distinct examples (30 per knowledge type), for development/setting selection; disjoint from K-test.
- **K-test:** 300 distinct held-out examples (100 per knowledge type), used for the primary RQ1.1 comparison.
- **K-small:** 40 randomly selected *from K-test* (14 factual, 13 conceptual, 13 procedural), used for RQ1.2 with three seeds. It is **not an independent held-out dataset**.
- Conditions within each model use the same example IDs, source images and original instructions with matched generation seeds. The same seed is a reproducibility control, not a guarantee of identical diffusion trajectories across different model/mode implementations.

### Generated outputs and judging coverage

| Component | Distinct final edited outputs |
|---|---:|
| K-test: 300 sources × 2 models × {Base, Thinking} × seed 42 | 1,200 |
| Additional K-small Base/Thinking runs: 40 × 2 models × 2 conditions × seeds {1234, 7} | 320 |
| K-small Correction: 40 × 2 models × 3 seeds | 240 |
| **Total frozen edited outputs** | **1,760** |

- The main judge archive has **380 complete batch JSON files**, containing **1,760** distinct ratings: **1,756 scored**, **4 unreadable**. There are also **10 repeat batches** (20 repeated ratings), so **390 batches** and **1,780 total ratings** counting repeats.
- Knowledge scoring was applicable to **932** main cases; **930** received valid knowledge scores. Unreadable outputs were excluded instead of being assigned arbitrary poor scores.

## 2. Scoring, statistics, and how to interpret them

The evaluation applied a KRIS-aligned rubric through **masked, manual ChatGPT UI judging**, documented as a GPT-6 Astra Pro workflow. It was a **substituted judge workflow**, not the original KRIS GPT-4o API evaluator and not directly numerically interchangeable with published KRIS results. Source image, original instruction, edit, and applicable knowledge references were provided; model/condition identifiers and generation settings were withheld from the judge. The source report flags that multiple outputs may occur in a batch, so within-batch influence remains possible.

Each applicable dimension is scored from **1 to 5**: visual consistency, instruction following, visual quality, and knowledge correctness (only on applicable items). The custom aggregate is:

```text
Q_i = mean_{d in applicable dimensions of i} [25 × (s_{i,d} − 1)]
```

Thus **Q ranges from 0 to 100 but is not percentage accuracy**, nor the benchmark’s published category-weighted aggregate. Changes in the four original dimensions are reported separately to explain aggregate effects.

**Primary inferential procedure:** comparisons are paired by source UID. For K-small, differences across available seeds are first averaged *within each example*, then examples—not 120 individual seeds—are the bootstrap resampling units. The analysis report records **12,000 example-level bootstrap resamples** for 95% percentile intervals and **50,000 two-sided paired sign-flip draws**, with **Holm correction across seven primary Q contrasts**. The sign-flip results are exploratory, particularly for small K-small. Note that a separate draft Methods section cited 10,000 replicates; use **12,000** when describing the analysis that generated the RQ1 intervals, unless rerun with different settings.

## 3. Primary quantitative results: all seven planned Q comparisons

| Comparison (left − right) | Split | Source UIDs | Valid paired outputs | ΔQ (points) | 95% CI | Holm-adjusted p |
| --- | --- | --- | --- | --- | --- | --- |
| Step1X: Thinking − Base | K-test | 299 | 299 | +3.79 | [+1.78, +5.84] | 0.00140 |
| UniReason: Thinking − Base | K-test | 300 | 300 | +0.19 | [-2.01, +2.43] | 0.87546 |
| Step1X: Correction − Thinking | K-small | 40 | 116 | -1.91 | [-4.51, +0.07] | 0.35723 |
| UniReason: Correction − Thinking | K-small | 40 | 120 | -1.56 | [-4.32, +0.90] | 0.52575 |
| Step1X − UniReason: Base | K-test | 300 | 300 | +8.17 | [+5.47, +10.79] | 0.00014 |
| Step1X − UniReason: Thinking | K-test | 299 | 299 | +11.82 | [+9.13, +14.49] | 0.00014 |
| Step1X − UniReason: Correction | K-small | 40 | 118 | +10.60 | [+3.88, +17.26] | 0.01384 |

**Reading this table:** positive ΔQ favors the left-hand condition/model. A confidence interval containing zero does not demonstrate a reliable advantage. For the two K-small within-model contrasts, “40 source UIDs” rather than “120 independent examples” determines the unit of uncertainty. Cross-model gaps reflect complete system differences (training/checkpoint/inference), not a clean causal comparison of their architectures.

### Paired outcome counts

| Contrast | Positive source examples | Negative source examples | Tied source examples | Total source examples |
| --- | --- | --- | --- | --- |
| Step1X: Thinking − Base | 114 | 75 | 110 | 299 |
| UniReason: Thinking − Base | 96 | 124 | 80 | 300 |
| Step1X: Correction − Thinking | 7 | 14 | 19 | 40 |
| UniReason: Correction − Thinking | 17 | 17 | 6 | 40 |
| Step1X − UniReason: Base | 175 | 80 | 45 | 300 |
| Step1X − UniReason: Thinking | 195 | 68 | 36 | 299 |
| Step1X − UniReason: Correction | 28 | 8 | 4 | 40 |

- For **RQ1.1**, each source contributes one paired comparison. For **RQ1.2**, counts are based on the sign of its **mean matched difference across available seeds**, not on individual seed-level ratings.
- Thus Step1X Correction improves **7/40** examples, degrades **14/40**, and ties **19/40**. UniReason Correction improves **17/40**, degrades **17/40**, and ties **6/40**. These score-based outcome categories are **not** human-coded failure mechanisms.

## 4. RQ1.1 — Thinking versus Base

### 4.1 Mean K-test scores and dimensions

| Model | Condition | n scored | Q /100 | Consistency /5 | Instruction /5 | Visual quality /5 | Knowledge /5 | Knowledge n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Step1X | Base | 300 | 53.38 | 3.03 | 2.65 | 4.18 | 2.15 | 163 |
| Step1X | Thinking | 299 | 57.18 | 3.17 | 2.87 | 4.25 | 2.33 | 163 |
| UniReason | Base | 300 | 45.21 | 2.60 | 2.49 | 3.81 | 1.82 | 163 |
| UniReason | Thinking | 300 | 45.40 | 2.33 | 2.83 | 3.52 | 2.31 | 163 |

**Interpretation.** Step1X Base averages **53.38** and Thinking **57.18**, but the *paired* estimate is **+3.79**, not simply the rounded unpaired difference, because the Thinking condition has one unreadable output. UniReason Base averages **45.21** and Thinking **45.40**, corresponding to **+0.19** paired points.

### 4.2 Mean paired change in every scored dimension

| Model | Rubric dimension | Valid pairs | Δ on 1–5 scale | 95% CI |
| --- | --- | --- | --- | --- |
| Step1X | Visual consistency | 299 | +0.14 | [+0.02, +0.26] |
| Step1X | Instruction following | 299 | +0.22 | [+0.09, +0.37] |
| Step1X | Visual quality | 299 | +0.07 | [-0.04, +0.19] |
| Step1X | Knowledge correctness | 163 | +0.18 | [+0.00, +0.38] |
| UniReason | Visual consistency | 300 | -0.27 | [-0.39, -0.15] |
| UniReason | Instruction following | 300 | +0.34 | [+0.20, +0.49] |
| UniReason | Visual quality | 300 | -0.29 | [-0.42, -0.16] |
| UniReason | Knowledge correctness | 163 | +0.50 | [+0.29, +0.70] |

**Step1X interpretation:** The reliable **+3.79 Q-point** improvement is accompanied by stronger **instruction following (+0.224)** and **visual consistency (+0.140)**. The observed visual-quality (+0.074) and knowledge (+0.184) changes have intervals that do not establish unambiguous positive effects at the same level. Thinking seems especially useful for *turning the requested edit into a more faithful change*, rather than universally increasing visual polish.

**UniReason interpretation:** Its higher **instruction following (+0.340)** and **knowledge scores (+0.497)** are offset by lower **visual consistency (−0.270)** and **visual quality (−0.290)**. This supports a **trade-off interpretation**: improved instruction/knowledge handling does not guarantee that the generated image preserves its source or remains visually high-quality. The aggregate **+0.19** CI spans zero, so do not claim either reliable improvement or reliable deterioration of the combined score.

**Published-study comparison:** Step1X reproduces the *direction* of the reported ReasonEdit thinking gain. The UniReason release does not show an overall inference-time-switch benefit under this project protocol. This is not necessarily contradictory to the UniReason publication: training-stage gains, different test samples, settings, refinement definitions, and the substituted evaluator do not constitute the same intervention. Do not compare published paper aggregates numerically to this project’s custom Q.

## 5. RQ1.2 — Correction versus Thinking

### 5.1 K-small descriptive means (all scored outputs)

| Model | Condition | n scored | n expected | Mean Q | Consistency | Instruction | Visual quality | Knowledge | Knowledge n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Step1X | Base | 120 | 120 | 48.82 | 3.03 | 2.23 | 4.05 | 1.97 | 60 |
| Step1X | Correction | 118 | 120 | 52.37 | 2.99 | 2.47 | 4.18 | 2.38 | 58 |
| Step1X | Thinking | 118 | 120 | 54.71 | 3.08 | 2.55 | 4.28 | 2.53 | 60 |
| UniReason | Base | 120 | 120 | 44.97 | 2.67 | 2.40 | 3.73 | 1.93 | 60 |
| UniReason | Correction | 120 | 120 | 41.81 | 2.23 | 2.51 | 3.53 | 2.12 | 60 |
| UniReason | Thinking | 120 | 120 | 43.37 | 2.19 | 2.63 | 3.55 | 2.35 | 60 |

**Crucial:** These are **unpaired descriptive means**, whereas the headline RQ1.2 results (**−1.91**, **−1.56**) are *paired, seed-within-example aggregated effects*. Do not subtract the two descriptive Step1X means and report that difference as the paired treatment effect; Step1X has two unreadable Correction outputs and two unreadable Thinking outputs on K-small.

### 5.2 Correction deltas for every rubric dimension

| Model | Rubric dimension | Valid seed pairs | Source UIDs | Δ on 1–5 scale | 95% CI |
| --- | --- | --- | --- | --- | --- |
| Step1X | Visual consistency | 116 | 40 | -0.09 | [-0.23, +0.02] |
| Step1X | Instruction following | 116 | 40 | -0.03 | [-0.17, +0.09] |
| Step1X | Visual quality | 116 | 40 | -0.07 | [-0.17, +0.01] |
| Step1X | Knowledge correctness | 58 | 20 | -0.13 | [-0.30, +0.00] |
| UniReason | Visual consistency | 120 | 40 | +0.04 | [-0.09, +0.17] |
| UniReason | Instruction following | 120 | 40 | -0.12 | [-0.33, +0.07] |
| UniReason | Visual quality | 120 | 40 | -0.02 | [-0.18, +0.14] |
| UniReason | Knowledge correctness | 60 | 20 | -0.23 | [-0.50, +0.00] |

- **Step1X:** correction Q **−1.91** (CI [−4.51, +0.07]); the point estimates for consistency, instruction, quality, and knowledge are all slightly negative. There is **no reliable aggregate gain**; the CI narrowly crosses zero. The score-based outcome counts favor ties/decreases over improvements (19 tied, 14 worse, 7 better).
- **UniReason:** correction Q **−1.56** (CI [−4.32, +0.90]); instruction and knowledge point estimates decline, consistency changes little, and quality is close to flat. No reliable aggregate gain; improvements and degradations occur in equal numbers (17 each).
- **Interpretation:** extra visual self-correction is not guaranteed to help and can damage successful edits. Since the mechanisms differ (conditional retry versus separate refinement), the two negative point estimates need not represent the same underlying failure mode. These K-small findings are not estimates over all 300 K-test sources.

### 5.3 Correction results by generation seed (descriptive)

| Model | Seed | Valid seed-level paired outputs | ΔQ (correction − thinking) |
| --- | --- | --- | --- |
| Step1X | 7 | 38 | -2.30 |
| Step1X | 42 | 38 | -1.15 |
| Step1X | 1234 | 40 | -2.40 |
| UniReason | 7 | 40 | +2.40 |
| UniReason | 42 | 40 | -2.24 |
| UniReason | 1234 | 40 | -4.84 |

**Interpretation:** Step1X has negative correction effects in all three seeds; UniReason changes sign across seeds (positive on seed 7 but negative on seeds 42 and 1234). Because there are only 40 sources and each seed is paired with those same sources, these seed-level results are descriptive rather than independent confirmations of efficacy.

### 5.4 Step1X reflection, candidate selection, and measured runtime

**Scope:** K-small, 40 unique UIDs × 3 seeds; **120/120 successful logged inference pairs** for runtime/attempt analysis, distinct from **116 valid scored pairs** used for Q.

| Step1X statistic | Observed value |
| --- | --- |
| Reflection-triggered second attempts | 69/120 (57.5%) |
| Single-attempt runs | 51/120 (42.5%) |
| Total generation attempts | 189 (51 × 1 + 69 × 2) |
| Mean generation attempts per run | 1.575 |
| Second attempts when first candidate was retained | 25/69 (36.2%) |
| Selected candidate index 0 / index 1 | 76 / 44 (includes 51 single-attempt outputs at index 0) |
| Final candidate selected as last candidate | 95/120 (79.2%); not last: 25/120 (20.8%) |
| Stop-reason counts | reflection_success 66; reflection_failed 24; max_attempts_reflection_requested 30 |
| Thinking mean / median logged end-to-end time | 113.23 s / 109.19 s |
| Correction mean / median logged end-to-end time | 223.48 s / 236.05 s |
| Mean / median paired additional time | +110.24 s / +126.98 s |
| Mean / median per-case Correction ÷ Thinking ratio | 2.00× / 2.18× |

**Interpretation:** Step1X’s conditional reflection triggered an extra image-generation attempt in **57.5%** of runs, yet the correction pipeline incurred an average **+110.24 seconds**, with a **2.00× mean within-pair time ratio**, without a statistically reliable Q gain. The longer runtime includes the costs actually captured in the logged end-to-end pipeline; it is **not** proof that every run had two diffusion attempts. In **25** of the **69** retried cases, the system kept the first candidate, so “correction enabled” is not synonymous with “a revised image is always selected”. The same-source score comparison includes cases where reflection has no beneficial effect.

**Timing arithmetic caution:** 2.00× is the **mean of each matched run’s time ratio**, not simply 223.48 ÷ 113.23, which is a different statistic. These time figures are Step1X logged inference measurements, not a directly comparable cross-model benchmark.

### 5.5 UniReason correction pairing audit and missing timing

- `outputs/rq1/correction_pairing_report.json` contains **120** records: 40 for each of seeds **42**, **1234**, and **7**.
- All **120/120** records report `initial_image_identical=true`, `reasoning_identical=true`, and `refined_image_valid=true`. The last check establishes **file/output validity**, **not correctness of the semantic edit**.
- UniReason applies the additional refinement pass, and all 120 paired outputs were scored. **Comparable per-example end-to-end runtime measurements were not recovered** from the saved RQ1 records; do **not** impute seconds or claim a measured UniReason runtime ratio.
- Therefore the defensible cost statement is **Step1X correction has measured substantial overhead without reliable Q improvement**; **UniReason correction adds refinement but its numerical runtime overhead remains unavailable**.

## 6. Cross-model system comparison

The complete released Step1X system scored higher than UniReason on matched outputs across the three compared conditions, under this judge/Q protocol:

| Condition | Subset | Source UIDs | Matched outputs | Step1X − UniReason ΔQ | 95% CI | Holm-adjusted p |
| --- | --- | --- | --- | --- | --- | --- |
| Base | K-test | 300 | 300 | +8.17 | [+5.47, +10.79] | 0.00014 |
| Thinking | K-test | 299 | 299 | +11.82 | [+9.13, +14.49] | 0.00014 |
| Correction | K-small | 40 | 118 | +10.60 | [+3.88, +17.26] | 0.01384 |

**Interpretation:** The Step1X system advantage increases from **+8.17** under Base to **+11.82** under Thinking on K-test. On the separate K-small correction subset the gap is **+10.60**. These are **complete-system comparisons**, not evidence that any single architectural component is superior, and the K-small correction gap should not be interpreted as a three-way matched K-test ranking.

### Other rubric dimensions in cross-model contrasts

| Mode | Dimension | n valid pairs | Step1X − UniReason Δ /5 | 95% CI |
| --- | --- | --- | --- | --- |
| Base | Visual consistency | 300 | +0.43 | [+0.23, +0.62] |
| Base | Instruction following | 300 | +0.15 | [-0.01, +0.31] |
| Base | Visual quality | 300 | +0.37 | [+0.22, +0.51] |
| Base | Knowledge correctness | 163 | +0.33 | [+0.10, +0.56] |
| Thinking | Visual consistency | 299 | +0.84 | [+0.68, +1.01] |
| Thinking | Instruction following | 299 | +0.04 | [-0.13, +0.21] |
| Thinking | Visual quality | 299 | +0.73 | [+0.58, +0.88] |
| Thinking | Knowledge correctness | 163 | +0.02 | [-0.20, +0.24] |
| Correction | Visual consistency | 118 | +0.75 | [+0.39, +1.12] |
| Correction | Instruction following | 118 | -0.02 | [-0.37, +0.31] |
| Correction | Visual quality | 118 | +0.65 | [+0.32, +0.98] |
| Correction | Knowledge correctness | 58 | +0.28 | [-0.15, +0.77] |

**Interpretation:** Under Thinking, the largest positive Step1X gaps are in **visual consistency (+0.843/5)** and **visual quality (+0.732/5)**. The Thinking instruction-following gap is small (+0.040) and its interval crosses zero. These patterns fit the interpretation that a more visually stable edit can score higher overall even when the two systems are not clearly separated on instruction compliance.

## 7. Exploratory breakdown by benchmark editing category

The category CSV reports **18 editing categories represented in the eligible/evaluated data**. The table below shows the mean **Thinking − Base ΔQ**, along with the valid source count for that contrast. These are **descriptive subgroup means**: per-category sample sizes can be small, intervals were not established for these individual means here, and selecting the largest differences invites multiple-comparison effects.

| Category | Step1X n | Step1X ΔQ | UniReason n | UniReason ΔQ |
| --- | --- | --- | --- | --- |
| abstract_reasoning | 30 | +7.29 | 30 | -7.92 |
| anomaly_correction | 16 | +13.54 | 16 | +2.60 |
| biology | 17 | -0.74 | 17 | +10.66 |
| chemistry | 14 | +7.59 | 14 | +11.16 |
| color_change | 16 | +0.52 | 16 | -3.12 |
| count_change | 16 | +6.77 | 16 | +0.52 |
| geography | 14 | +11.16 | 14 | +7.14 |
| humanities | 13 | +4.33 | 13 | +11.54 |
| mathematics | 12 | +7.29 | 12 | -0.52 |
| medicine | 11 | +15.91 | 11 | +5.68 |
| multi-instruction_execution | 37 | +2.25 | 37 | -0.90 |
| part_completion | 12 | +4.86 | 12 | -18.06 |
| physics | 9 | -1.39 | 9 | -3.47 |
| position_movement | 13 | +0.64 | 13 | +5.13 |
| practical_knowledge | 10 | -9.38 | 10 | +2.50 |
| rule-based_reasoning | 33 | -1.70 | 33 | -5.87 |
| size_adjustment | 16 | -2.08 | 17 | +1.96 |
| viewpoint_change | 10 | +5.83 | 10 | +0.00 |

- The largest positive observed Step1X category means include **medicine (+15.91; n=11)**, **anomaly_correction (+13.54; n=16)**, and **geography (+11.16; n=14)**. Its most negative listed category is **practical_knowledge (−9.38; n=10)**.
- For UniReason, positive observed means include **humanities (+11.54; n=13)**, **chemistry (+11.16; n=14)**, and **biology (+10.66; n=17)**, while **part_completion (−18.06; n=12)** and **abstract_reasoning (−7.92; n=30)** are negative. These contrasts help generate failure-analysis hypotheses, **not** new claims of category-specific significance.
- The denominator here is category membership *within K-test*, not 75 per class or the full original KRIS-Bench distribution; categories are not balanced by the knowledge-type stratifier.

### Exploratory K-small category correction effects

The following table is included for completeness only. A category may contain as few as one K-small source; **do not use these category-level correction effects as reliable generalizations**.

| Category | Step1X Correction − Thinking ΔQ | UniReason Correction − Thinking ΔQ |
| --- | --- | --- |
| abstract_reasoning | +0.00 (n=1) | +0.00 (n=1) |
| anomaly_correction | +1.39 (n=2) | +1.39 (n=2) |
| biology | +0.00 (n=1) | -4.17 (n=1) |
| chemistry | — | — |
| color_change | +0.00 (n=1) | +0.00 (n=1) |
| count_change | -13.89 (n=3) | -0.93 (n=3) |
| geography | +0.00 (n=2) | +4.17 (n=2) |
| humanities | -2.08 (n=3) | -4.17 (n=3) |
| mathematics | -4.17 (n=2) | -18.75 (n=2) |
| medicine | -22.92 (n=1) | -4.17 (n=1) |
| multi-instruction_execution | +0.46 (n=6) | -0.46 (n=6) |
| part_completion | — | — |
| physics | -2.08 (n=2) | +2.08 (n=2) |
| position_movement | -1.39 (n=2) | -18.06 (n=2) |
| practical_knowledge | -1.04 (n=2) | +4.17 (n=2) |
| rule-based_reasoning | +1.04 (n=6) | +1.39 (n=6) |
| size_adjustment | +0.00 (n=4) | +0.69 (n=4) |
| viewpoint_change | +0.00 (n=2) | +1.39 (n=2) |

## 8. RQ1.3 — Where reasoning fails (qualitative diagnostic extension only)

**Scope:** RQ1.3 is **representative qualitative examples only**, organized into **interpretation**, **execution**, and **critique** failures. It does **not** require 80-case labeling, a frequency heatmap, or Cohen’s κ. A previously generated 80-board review set and 10-case blinded pilot are useful *source material*, but they are not a completed population-level human annotation study.

| Failure stage | Diagnostic definition | Illustrative pilot evidence | Interpretation |
| --- | --- | --- | --- |
| **Interpretation** | Model misunderstands what property/object the user wants changed | **Anatomical anomaly:** source foot has extra toes; the reasoning focuses on skin tone or general foot size/proportions and the edits retain extra toes. | Misidentified edit target; greater reasoning effort does not repair the intended anomaly when the plan addresses the wrong issue. |
| **Execution** | Reasonable/correct identified goal is not realized in generated pixels | **Side-view framed object:** the rewrite explicitly calls for a side profile, but the image remains largely frontal. **Animal-product matching:** one trace correctly states yarn–sheep, honey–bee, eggs–chicken, milk–cow, but the image does not show the requested successful matches. | A correct verbal relation or spatial instruction is insufficient for successful visual editing. |
| **Critique** | Post-generation reflection accepts a bad output or otherwise evaluates it incorrectly | **4×4 Sudoku:** an output includes impossible values (e.g., 5/9 in a 1–4 puzzle), yet an available reflection declares successful completion. | False acceptance can prevent useful repair; careful distinction required from cases where reflection correctly detects failure. |

**Additional illustrative cases / cautions:**

- **Cup movement, `position_movement__34`, Step1X, K-test seed 42:** The original contains two cups, and the instruction is to move one onto another. The Base image introduces an extra cup, while Thinking repositions an existing one. This is a **success case supporting RQ1.1**, not itself an example of reasoning failure. The recorded case-level Q changed from approximately **58.3 to 83.3** (descriptive individual example, not a mean effect). Recommended labels: **Original | Step1X Base | Step1X Thinking**.
- **Seasonal fox:** both reasoning variants can change a winter-white fox toward seasonal fur, but some edits also alter the setting or the fox’s appearance. The official editing instruction/reference must be checked before claiming that environmental changes were prohibited. It is a useful *scope-preservation* illustration, not a verified failure-category prevalence figure.
- **Sudoku nuance:** one pilot case has an incorrect logical plan **and** an invalid image, while another has a short generic plan and a demonstrably false success reflection. Do not call a case “correct plan, wrong image” unless the recorded plan is actually specific and correct.
- **Critique availability:** the blinded 10-case pilot comprises five source images under two masked model conditions; approximately half of the correction-critique traces were placeholders. Do not infer a mistaken critique from **missing** logs. Also distinguish a **false acceptance** from a reflection that correctly calls an image wrong but fails to correct it.

**What these examples support:** reasoning may fail at the request-interpretation stage, at the model-to-image execution stage, or in the reflection/acceptance stage. The visible rationale/critique is only an *observable trace*, not a full explanation of hidden computation. **These examples do not estimate failure rates, establish model rankings for particular failure categories, or claim independent human annotation.**

**Suggested figures:**

1. `fig/rq1_cups.png`: original → Step1X Base → Step1X Thinking, demonstrating successful spatial editing (RQ1.1).
2. One RQ1.3 figure containing three *actual saved* failures: anatomical anomaly (interpretation), side-view frame (execution), Sudoku (critique); use source/output panels and include a short relevant reflection excerpt for critique. Verify source instructions, candidate images and the specific trace before final selection.
3. The optional fox teaser may illustrate preservation/scope trade-offs, but should not imply a proven model-wide tendency.

## 9. Judge repeatability and measurement limitations

The 10 repeat batches contained **20 identical final edited outputs re-rated** (10 Step1X Thinking plus 10 UniReason Correction images, from 10 source UIDs), compared against their main ratings. Original image identity was checked using SHA-256 and case metadata. **This is intra-judge re-scoring, not human–judge agreement.**

| Repeatability statistic | Result |
| --- | --- |
| Repeated images rated twice | 20 |
| Exact Q agreement | 15/20 = 75% |
| Mean absolute Q difference | 3.125 /100 |
| Bootstrap 95% interval for mean absolute difference | [0.625, 6.25] (exploratory; 20,000 resamples) |
| Median absolute Q difference | 0 |
| 95th percentile of absolute Q difference | 17.08 /100 (unstable at n=20) |
| Maximum absolute Q difference | 25 /100 |
| Mean signed second − first Q difference | −1.875 /100 |
| Consistency exact agreement | 16/20 (80%); mean absolute 1–5 difference = 0.25 |
| Instruction exact agreement | 18/20 (90%); mean absolute 1–5 difference = 0.10 |
| Visual-quality exact agreement | 17/20 (85%); mean absolute 1–5 difference = 0.15 |
| Knowledge exact agreement when applicable | 12/12 (100%); mean absolute 1–5 difference = 0.00 |

**Interpretation and cautions:** Repeat ratings are mostly stable, but several nontrivial Q differences occur. The small, nonrepresentative 20-image repeat set is **not** proof of overall judge validity, stability in every condition, or alignment with human experts. The repeat archive did not include complete independently auditable fresh-session/model-setting metadata. The 95th-percentile item-level re-score difference **must not** be treated as the detection threshold for mean treatment effects. More independent human checks would strengthen validity, but are outside the deliberately qualitative-only scope of RQ1.3.

### Unreadable scored outputs (all four)

| Example UID | Model | Condition | Seed | Category | Judge batch |
| --- | --- | --- | --- | --- | --- |
| geography__40 | Step1X | Correction | 42 | geography | main_b080 |
| rule-based_reasoning__75 | Step1X | Correction | 7 | rule-based_reasoning | main_b128 |
| size_adjustment__33 | Step1X | Thinking | 42 | size_adjustment | main_b138 |
| count_change__47 | Step1X | Thinking | 7 | count_change | main_b362 |

- The lone missing K-test pair for RQ1.1 is **`size_adjustment__33` under Step1X Thinking, seed 42**. The judge described the image as almost entirely white/unreadable; whether this reflects the saved image or a display/judging failure should be established by directly inspecting the actual output.
- **Step1X K-small scored Correction and Thinking:** 118/120 each, with unreadables leaving **116 mutually valid Correction − Thinking pairs**; Step1X runtime records nevertheless exist for **120/120** runs. **UniReason K-small** scores are valid for **120/120** runs in each of Thinking and Correction.
- Missing outputs were omitted from metric means and paired comparisons. This is a defensible primary analysis, but a sensitivity analysis could evaluate how much different treatments of unreadables change estimates.

## 10. Main conclusions and practical interpretation

1. **Thinking is effective for Step1X on K-test.** The paired improvement is **+3.79/100**, with its 95% CI entirely positive and a Holm-adjusted p-value of 0.0014. Gains are associated chiefly with instruction adherence and source-image consistency.
2. **Thinking does not reliably change UniReason’s aggregate Q.** Its nearly zero overall effect masks simultaneous benefits in instruction/knowledge handling and losses in visual preservation/quality. This is a trade-off, not “the model does not reason.”
3. **Native Correction fails to deliver a reliable mean Q gain on K-small for either editor.** The small sample and seed variability counsel against stronger claims such as “correction always hurts.” Some examples get better while others get worse.
4. **Step1X correction is expensive under the observed deployment.** Reflection caused extra generations in 57.5% of runs and raised logged inference time by an average 110.24 seconds, for a mean 2.00× matched-run runtime ratio, without measured reliable aggregate quality improvement.
5. **UniReason refinement was successfully paired and valid as image files**, but its per-case runtime overhead was not available, so the cost comparison is deliberately asymmetric.
6. **The complete Step1X system scores higher under this Q protocol** on matched Base, Thinking and Correction contrasts. This is a system-level performance observation, not a causal architecture-isolation result.
7. **RQ1.3 provides diagnostic explanations, not population statistics.** Examples illustrate incorrect interpretation, failure to realize a correct plan, and false acceptance during reflection. These offer plausible mechanisms behind the limited correction benefits.
8. **Conclusions are conditional on the judged data and evaluator.** The substituted manual-UI judge, four unreadables, overlapping K-test/K-small membership, differences between model-specific correction procedures, and uncertainty in small subsets limit external generalization.

## 11. Report/presentation-ready summaries

### One-paragraph results summary

On 300 held-out KRIS-Bench examples, enabling inference-time Thinking improved Step1X-Edit’s paired aggregate editing score by **3.79 points** (95% CI **1.78–5.84**, **299** valid pairs) but produced no reliable change for UniReason (**+0.19**, 95% CI **−2.01–2.43**, **300** pairs). Thinking enhanced Step1X instruction adherence and source-image consistency. For UniReason, improvements in instruction and knowledge scores were offset by deteriorations in consistency and visual quality. Adding native Correction on 40 K-small examples across three seeds yielded no reliable gain for either Step1X (**−1.91**, CI **−4.51–0.07**) or UniReason (**−1.56**, CI **−4.32–0.90**). Step1X reflection triggered a retry in **69/120** runs and increased its average logged time from **113.23** to **223.48 seconds**; UniReason’s pairing audit passed all 120 runs, but comparable timing was unavailable. Representative failures distinguish incorrect interpretation, unsuccessful execution, and incorrect critique. The results are specific to a custom KRIS-aligned score from a substituted ChatGPT UI judge, rather than published KRIS official metrics.

### Slide-ready interpretation

> **Thinking helps Step1X, but not UniReason’s aggregate Q; another correction pass does not reliably improve either model.** The cup example shows where Step1X Thinking helps; the anatomical, side-view, and Sudoku examples illustrate failures in interpretation, execution, and critique.

## 12. Reproduction and source artifacts

The numerical tables in this summary were reconstructed from the preserved RQ1 analysis files (the saved `RQ1_Results` analysis package). They should be regenerated from the frozen scores rather than retyped when preparing new figures.

| File / location | Purpose |
|---|---|
| `data/splits.json`; `src/create_splits.py` | Frozen knowledge-stratified K-dev/K-test/K-small IDs; split-generation seed |
| `reports/RQ1_Results/ANALYSIS_REPORT.md` (or corresponding extracted analysis folder) | Primary results, protocol and limitations |
| `paired_comparisons_all_metrics.csv` | Seven Q contrasts; per-rubric effects, intervals, counts and p-values |
| `descriptive_ktest.csv`; `descriptive_ksmall.csv` | Model/condition average scores by split |
| `paired_category_differences.csv` | Exploratory benchmark-category contrasts |
| `paired_seed_differences.csv` | RQ1.2 differences by generation seed |
| `unreadable.csv` | Excluded judgment cases and IDs |
| `REPEAT_CALIBRATION_REPORT.md`; `repeat_pair_scores_PRIVATE.csv` | Independent re-rating repeatability audit |
| `analysis/rq1_correction_runtime/step1x_summary.json` | Step1X K-small retry, selection and timing summary (reported from the NSCC terminal output) |
| UniReason `outputs/rq1/correction_pairing_report.json` | 120-case initial-image/reasoning/refinement validity audit (reported from NSCC) |
| `annotations/rq13_review_2026/panels/` and `traces/` | 80 prepared qualitative boards and available internal traces |
| `pilot_RQ13_BLINDED.zip`; `RQ13_Pilot_Review_Notes.md` | 10-case blinded pilot and AI-assisted diagnostic notes; **not** independent human annotations |
| `fig/rq1_cups.png` | Successful Step1X Base vs Thinking example, where present in the report repository |

**Privacy/blinding note:** `individual_scores_private.csv`, case-selection private CSVs and the mapping between blinded review IDs and model identities are internal artifacts. Do not expose the private maps to a still-blinded evaluator.

---

**Scope note:** This summary consolidates verified RQ1 score CSVs, the analysis report, uploaded qualitative pilot notes, and the Step1X/UniReason audit outputs supplied during the project discussion. Numerical Step1X timing was reported by the successful NSCC analysis; UniReason timing is intentionally left unavailable. Some subgroup observations and RQ1.3 diagnoses are explicitly exploratory. No new image generation, human scoring, or model inference was performed for this document.

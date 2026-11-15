# RQ3 findings: thinking under regional and temporal acceleration

## 1. Scope and contribution

RQ3 examines whether regional and temporal acceleration preserve thinking-related gains in **Step1X-Edit-v1p2**, and which acceleration components explain the observed quality–efficiency trade-off. Following the RQ3 plan in *6132 Group Project Writeup* and §4–6 of *6132 Suggested Positioning*, the principal contribution is a controlled **thinking × acceleration interaction analysis**. The component removals adapt existing RegionE ablations to this reasoning-enabled deployment; they are supporting experiments, not new acceleration algorithms. RQ3 supplies a focused Step1X study alongside the broader Step1X–UniReason comparisons in RQ1–RQ2. These two planning documents are separate project materials; the completed methods and evidence are linked below.

The completed experiment demonstrates substantial acceleration with limited average loss in the recorded editing scores. The original 40-example ChatGPT analysis does **not** establish a positive thinking benefit. Supplementary validation by two humans on 30 matched examples yields more positive thinking estimates, but weak human–judge agreement and reviewer-dependent uncertainty. Both humans support limited average acceleration loss under the original tolerance. Thus, the efficiency result and average score-retention conclusion are more consistent across evaluators than the estimated benefit of thinking; neither score retention nor reviewer agreement establishes complete editing correctness.

The numerical evidence below comes from [RESULTS.md](outputs/rq3/postprocessing/RESULTS.md), the [canonical analysis](outputs/rq3/postprocessing/canonical_report.json), the [human-validation results](outputs/rq3/human_validation/analysis/RESULTS.md), and the completed findings in [RQ3_experiment.md](RQ3_experiment.md#completed-experiment-and-analysis-findings). The supplied planning documents define the intended questions; original and supplementary analyses are distinguished below.

## 2. Experimental design and measurement

The main factorial experiment uses 40 predetermined KRIS-Bench examples across the three knowledge types. A predetermined 20-example subset supports the component study. All conditions use 28 denoising steps, CFG 6, seed 42, reflection disabled, bfloat16 precision, model-level offloading, and Step1X's aspect-preserving 1024 size setting on an NVIDIA A100-SXM4-40GB. Checkpoint and code revisions are frozen in the [protocol record](outputs/rq3/evidence/analysis/records/protocol.json). Thinking-enabled comparisons reuse each example's cached thought/rewrite and matched initial diffusion noise. RQ3 therefore estimates its own baselines independently of RQ1 and RQ2.

There are **200 quality outputs and 600 additional timing generations**. The inferential sample sizes are 40 and 20 examples, respectively; timing repetitions are not independent quality observations. Quality was assessed through a manual ChatGPT interface adaptation of the KRIS rubric, with operator-recorded model label `GPT-6 Astra Pro`. For example $i$,

$$
Q_i=\frac{1}{|\mathcal D_i|}\sum_{d\in\mathcal D_i}25(s_{id}-1),
$$

where $s_{id}\in\{1,\ldots,5\}$ and $\mathcal D_i$ contains applicable instruction-following, visual-consistency, visual-quality and knowledge-plausibility dimensions. This example-weighted, 0–100 subset score is **not the published KRIS aggregate or its pinned API evaluation**.

Paired effects are calculated within examples, then averaged. Confidence intervals use 10,000 paired example-bootstrap resamples. Timing estimates average each example's median over three repetitions after warm-up. **Charged total** combines measured replay with the measured cost of one thinking/offload pass for thinking-enabled conditions; it is reconstructed pipeline accounting, excluding model loading, warm-up and artifact writes. Memory reports mean per-example maximum PyTorch allocated VRAM across timing repeats, rather than total device memory.

## 3. RQ3.1: original ChatGPT analysis of thinking and acceleration

**Table 1. Main factorial conditions and deployment costs; all rows use the same 40 examples.**

| Condition | Thinking | RegionE | Mean Q ↑ | Generator (s) ↓ | Charged total (s) ↓ | Peak allocated VRAM (GiB) ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | Off | Off | 53.59 | 56.95 | 66.67 | 24.11 |
| B | On | Off | 51.98 | 56.89 | 72.07 | 24.11 |
| C | Off | On | 54.06 | 21.79 | 31.50 | 35.07 |
| D | On | On | 51.41 | 22.10 | 37.05 | 35.07 |

**Table 2. Paired quality contrasts answering RQ3.1.**

| Quantity | Definition | Mean difference in Q | 95% CI |
| --- | --- | ---: | --- |
| Thinking effect without acceleration, $G_0$ | B − A | −1.61 | [−5.94, 2.14] |
| Thinking effect with acceleration, $G_1$ | D − C | −2.66 | [−7.14, 1.41] |
| Thinking × acceleration interaction, $I$ | (D − C) − (B − A) | −1.04 | [−2.40, 0.00] |
| Acceleration effect with thinking, $\Delta Q_1$ | D − B | −0.57 | [−1.56, 0.00] |
| Acceleration effect without thinking, $\Delta Q_0$ | C − A | +0.47 | [0.00, 1.41] |

**Thinking benefit.** Under the original ChatGPT scoring, neither thinking contrast establishes a positive mean effect: both intervals include zero. Thus, these ratings do not establish a positive gain whose preservation can be demonstrated. This finding is configuration- and evaluator-specific; it cannot establish that thinking is generally ineffective. A gain measured at 50 steps in another RQ cannot substitute for the baseline benefit here. Section 5 examines sensitivity to human scoring on matched outputs.

**Interaction versus absolute retention.** The interaction interval has width 2.40 points and lies within the frozen ±5-point tolerance. Acceleration therefore produces a small average change in the measured thinking effect relative to that tolerance. Separately, the lower bound for D − B is −1.56, above the predefined acceptable-loss threshold of −5. The experiment meets its **average score-retention criterion**. This decision uses the lower endpoint of the existing two-sided 95% interval; it does not rely on a nonsignificant test establishing equivalence.

**Efficiency and memory.** D provides a mean paired generator speedup of **2.59×** and charged-total speedup of **1.95×** over B. The smaller overall gain reflects costs outside diffusion, including thinking and transfers. These are means of within-example speed ratios; dividing the aggregate times in Table 1 instead yields 2.57× and 1.94×. RegionE increases peak allocated VRAM from 24.11 to 35.07 GiB, demonstrating a latency–memory trade-off in this deployment.

**Individual outcomes.** D and B receive equal aggregate scores on 38/40 examples. The remaining differences are −16.67 for `position_movement__30` and −6.25 for `geography__6`. Average retention therefore does not guarantee retention for every example. Moreover, score ties limit sensitivity to subtle changes and should be interpreted alongside images and perceptual measurements.

## 4. RQ3.2: which acceleration components matter?

B and D are reused on the same 20 examples. E removes RIKVCache contextual injection during regional attention; this intervention changes available context rather than recomputing equivalent context. F disables AVDCache temporal velocity reuse while retaining regional processing. Development checks verified the intended switches and matched controls before scaling.

**Table 3. Component study; every row uses the predetermined 20-example subset.**

| Condition | Mean Q ↑ | Generator (s) ↓ | Charged total (s) ↓ | Peak allocated VRAM (GiB) ↓ |
| --- | ---: | ---: | ---: | ---: |
| B: no RegionE | 53.13 | 56.66 | 71.81 | 24.11 |
| D: full RegionE | 53.13 | 21.75 | 36.58 | 35.05 |
| E: without RIKV | 48.96 | 20.58 | 35.49 | 24.12 |
| F: without AVD | 53.13 | 34.42 | 49.27 | 35.05 |

**RIKV and contextual preservation.** Removing RIKV changes mean Q by **−4.17** relative to D, with 95% CI **[−8.85, 0.00]**. It reduces memory substantially but yields only a small additional latency reduction. The interval includes zero and extends below −5: neither a conclusive nonzero quality loss nor retention under the same tolerance is established. The estimated score loss, larger image deviations and observed contextual disruption jointly motivate retaining RIKV in this configuration, subject to further validation.

**AVD and temporal reuse.** Removing AVD increases generator time from 21.75 to 34.42 seconds without a detected score change: F − D = 0.00, CI [0.00, 0.00]. B, D and F have identical component-score vectors on all 20 examples, although their images differ. The degenerate score interval reflects observed ties, not exact perceptual equivalence or zero population uncertainty. F remains faster than B, consistent with savings from the remaining regional processing; the component effects should not be treated as an additive decomposition of runtime.

**Executed computation.** C/D/E record 15 AVD-reuse steps and 26 transformer calls, compared with 56 calls in A/B/F. Calls include both conditional and unconditional CFG branches. Every condition still uses 28 configured denoising steps, illustrating why scheduler steps alone do not quantify generator work. Per-step token fractions, cache events and refresh records explain the execution pattern. Spatial overlays retain the selected-token snapshot at zero-based step 5, not a complete sequence of region masks.

**Table 4. Supplementary image deviation, computed at original RGB geometry. Baselines are generated outputs, not ground-truth correct edits.**

| Comparison | Examples | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| C versus A | 40 | 32.64 | 0.9544 | 0.04244 |
| D versus B | 40 | 33.43 | 0.9543 | 0.03936 |
| E versus D | 20 | 24.45 | 0.8483 | 0.17371 |
| F versus D | 20 | 36.94 | 0.9715 | 0.02513 |
| E versus B | 20 | 24.38 | 0.8434 | 0.18052 |
| F versus B | 20 | 34.67 | 0.9660 | 0.03074 |

The matched 20-example E/D and F/D comparisons show substantially greater deviation after removing contextual injection than after removing temporal reuse. Similarity to a generated baseline remains supplementary evidence: preserving an incorrect image cannot establish successful editing. Metric details are retained in the [image diagnostics](outputs/rq3/postprocessing/qualitative/README.md) and [LPIPS results](outputs/rq3/postprocessing/perceptual/summary.json).

## 5. Per-example interpretation and validation limits

Four cases illustrate distinct explanations for the aggregate results. These are documented **assistant visual observations**, supported by the [case catalogue](outputs/rq3/postprocessing/qualitative/candidates.md) and [observation provenance](reports/rq3_case_observations.json); they are not independent human annotations.

| Example | Observed evidence | Interpretation for the report |
| --- | --- | --- |
| `count_change__48` | Eight forks should become six; the rewrite requests removal of four. A/C retain eight forks, whereas B/D are nearly blank. | Incorrect interpretation and failed execution already occur without acceleration and persist with RegionE in this example. |
| `mathematics__6` | B/D better retain the original circle layout and move the green circle towards the requested centre, but introduce an unwanted blue fill. | A positive thinking contrast can represent relative improvement without complete instruction satisfaction. |
| `count_change__47` | B/D/F receive Q = 83.33 despite a blank charger face where six interfaces are requested. E additionally alters the charger, plug and background. | Context removal visibly disrupts content, while high baseline scores expose a judge-validity concern. |
| `position_movement__30` | B and D both duplicate a cup rather than simply relocating it; they receive Q = 58.33 and 41.67. | The largest scored acceleration loss is also an evaluator-discrepancy case; its score difference alone does not establish an equally large semantic deterioration. |

The seven available case panels were selected using disclosed **post-hoc** rules, including the largest positive and negative score changes. They explain observed cases rather than estimate failure prevalence. All examples remain in the quantitative analysis. The planned separation of reasoning difficulty from expected spatial edit extent is **not complete**: knowledge-type labels and selected-token fractions do not replace independent edit-extent annotations. No category-level causal conclusion or error-frequency heatmap is justified by the present observations.

Calibration produced exact agreement on 20 repeated development-output scores, giving observed MMD = 0; the five-point acceptable-loss margin was fixed before test scoring. However, calibration session metadata was corrected retrospectively under operator attestation without independent transcript verification. Repeatability does not establish correctness, particularly given the counting and movement discrepancies above. The manual UI model lacks a controlled API snapshot and temperature, and the example-bootstrap intervals omit evaluator and generation-seed uncertainty. Three timing repetitions do not constitute a multi-seed robustness study.

### Completed human validation

Two reviewers completed **240 ratings** of 120 A–D outputs from a [fixed 30-example subset](data/rq3_human_validation_subset.json), balanced across the three knowledge types. Eight practice ratings were excluded. Selection used metadata after the original analysis and before human scoring. The [review packets](RQ3_human_validation.md) concealed condition labels, rewrites and judge scores and randomized presentation; independence and prior exposure are reviewer-attested. The analysis verifies image identities and compares each human with the existing ChatGPT scores for exactly these outputs. The secondary **human mean** averages the two reviewers' $Q$ values; it is neither an additional independent reviewer nor adjudicated ground truth.

**Table 5. Aggregate-score agreement on 30 examples; 95% example-cluster bootstrap intervals.**

| Rater pair | Spearman $\rho$ [95% CI] | ICC(A,1) [95% CI] | Mean absolute Q difference [95% CI] |
| --- | --- | --- | --- |
| Reviewer 1 / Reviewer 2 | 0.898 [0.813, 0.951] | 0.899 [0.819, 0.943] | 7.43 [5.26, 9.69] |
| Reviewer 1 / ChatGPT | 0.042 [−0.245, 0.318] | 0.027 [−0.186, 0.211] | 23.40 [18.37, 28.84] |
| Reviewer 2 / ChatGPT | 0.133 [−0.157, 0.397] | 0.134 [−0.074, 0.308] | 24.69 [20.30, 29.48] |

The reviewers agree closely on aggregate $Q$, whereas neither human–judge comparison demonstrates close agreement. Human–human quadratically weighted $\kappa$ is 0.901 [0.790, 0.967] for instruction following, 0.844 [0.733, 0.931] for visual quality and 0.831 [0.671, 0.942] for knowledge; consistency agreement is lower at 0.498 [0.226, 0.719]. Knowledge applies to 64 outputs from 16 examples. Mean ChatGPT-minus-human differences are −2.22 and −4.55 points, with intervals containing zero, despite absolute differences exceeding 23 points. Disagreement therefore cannot be summarized as uniform judge overrating: discrepancies occur in both directions and can cancel in the mean.

**Table 6. Evaluator sensitivity of RQ3.1 on the same 30 examples. Values are paired mean Q differences [95% CI].**

| Contrast | Reviewer 1 | Reviewer 2 | ChatGPT | Human mean (secondary) |
| --- | --- | --- | --- | --- |
| $G_0=B-A$ | +8.19 [2.43, 14.03] | +6.18 [−2.22, 14.17] | −1.18 [−6.94, 3.54] | +7.19 [0.45, 13.72] |
| $G_1=D-C$ | +7.29 [1.39, 13.40] | +6.25 [−2.09, 14.38] | −1.81 [−7.71, 3.06] | +6.77 [−0.03, 13.40] |
| $I=G_1-G_0$ | −0.90 [−3.89, 2.08] | +0.07 [−1.81, 2.01] | −0.62 [−1.88, 0.00] | −0.42 [−2.67, 1.74] |
| $\Delta Q_1=D-B$ | −1.46 [−3.96, 0.69] | −0.83 [−2.22, 0.21] | 0.00 [0.00, 0.00] | −1.15 [−3.02, 0.31] |

Both humans estimate positive thinking effects, but Reviewer 2's intervals include zero. The secondary human mean establishes a positive unaccelerated contrast under its interval, while the accelerated contrast remains uncertain. Consequently, the original absence of an established thinking benefit is sensitive to the evaluator. Each human's D − B lower bound remains above −5, and each interaction interval lies within ±5. This supports limited average acceleration loss under human scoring as a **supplementary sensitivity analysis**, without establishing universal preservation of a positive thinking benefit. The subset excludes the two examples with original D/B score losses; its tied ChatGPT D/B scores therefore do not contradict the original 40-example estimate.

All intervals use 10,000 bootstrap resamples of the 30 examples, retaining all conditions and raters within each cluster. ICC(A,1) denotes a two-way mixed-effects, absolute-agreement, single-measurement model for each fixed rater pair. The high-level methods preceded joint analysis, but implementation followed submission and included access to point estimates before finalization; this is not preregistered evidence. The findings concern two named reviewers, one generation seed and A–D only. They do not validate the full 40-example results or E/F component effects. The [analysis plan](reports/rq3_human_analysis_plan.json) and [machine-readable report](outputs/rq3/human_validation/analysis/report.json) retain these decisions and preserve original scores.

The positioning statement can therefore read: **“A Step1X-specific study investigates whether regional and temporal acceleration preserve thinking-related gains. Per-example failure analysis and human–judge validation reveal evaluator-sensitive thinking effects, while supporting limited average acceleration loss under both recorded judge and human scores.”** Human validation supports this qualified interpretation; it does not confirm the judge's scoring validity.

## 6. Recommended report figures and tables

Use four principal figures and Tables 1, 3 and 5 in the main text. Tables 2, 4 and 6 provide exact effects and supplementary metrics and can move to the appendix if space is limited. Human validation is essential to interpreting the thinking result. Existing assets are linked below; combining panels into the proposed report layouts is a presentation step, not an additional experiment.

### Figure 1 — Interaction and practical trade-off

Combine the [paired-effect forest plot](outputs/rq3/postprocessing/figures/main_effects.pdf) with the [quality–latency plot](outputs/rq3/postprocessing/figures/quality_latency_intervals.pdf). The forest plot answers the interaction and retention questions; the latency panel shows their deployment implications. Retain the zero line, the −5 retention threshold and the ±5 interaction tolerance. Use paired contrasts for inference rather than overlap between marginal condition intervals.

*Proposed caption:* **Thinking × RegionE on 40 matched examples under original ChatGPT scoring.** Left: paired mean quality effects with 95% example-bootstrap intervals. Right: mean editing score against reconstructed charged-total latency. RegionE accelerates thinking-enabled editing within the predefined average score-loss tolerance, while neither thinking contrast establishes a positive benefit under these ratings. Results use one generation seed.

### Figure 2 — Component contributions to efficiency

Use the existing [ablation effects and efficiency figure](outputs/rq3/postprocessing/figures/ablation_effects_efficiency.pdf), accompanied by Table 3 so that the memory cost is visible. Place [per-step activity](outputs/rq3/postprocessing/figures/region_step_activity.pdf) in the appendix as the execution-level explanation.

*Proposed caption:* **RIKV and AVD removals on 20 matched examples.** Removing contextual injection produces an uncertain negative score effect and lower memory use. Removing temporal reuse increases latency without changing the recorded ratings. Speedups use B on this same subset; zero-width quality intervals arise from tied ratings rather than identical images.

### Figure 3 — Interpretation, execution and evaluation failures

Assemble two principal rows from the [fork-count A–D panel](outputs/rq3/postprocessing/qualitative/lowest_thinking_effect__count_change__48__ABCD.png) and [charger B/D/E/F panel](outputs/rq3/postprocessing/qualitative/lowest_no_rikv_effect__count_change__47__BDEF.png). For the charger case, include its [selected-region overlay](outputs/rq3/postprocessing/qualitative/lowest_no_rikv_effect__count_change__47__selected_regions.png) and [difference maps](outputs/rq3/postprocessing/qualitative/lowest_no_rikv_effect__count_change__47__differences.png). Display the original instruction, relevant rewrite, condition labels and original Q scores. Put the [relative thinking improvement](outputs/rq3/postprocessing/qualitative/highest_thinking_effect__mathematics__6__ABCD.png) and [cup-score discrepancy](outputs/rq3/postprocessing/qualitative/lowest_acceleration_effect__position_movement__30__ABCD.png) in the appendix to show both favorable and unfavorable cases.

*Proposed caption:* **Post-hoc examples separating interpretation, execution and evaluator limitations.** The fork-count error precedes acceleration. The charger example exhibits additional context disruption without RIKV and a mismatch between high judge scores and the requested edit. Overlays show token selection at step 5, not attention or ground-truth edit masks; difference maps indicate image changes rather than correctness. Cases do not estimate failure prevalence.

### Figure 4 — Human agreement and sensitivity of the thinking effect

Combine the [aggregate agreement plots](outputs/rq3/human_validation/analysis/figures/agreement_q.pdf) with the [matched condition-effect plot](outputs/rq3/human_validation/analysis/figures/condition_effects.pdf). The former contrasts human–human and human–judge agreement; the latter shows the consequence for RQ3.1 on identical examples. Retain each reviewer separately, label the human mean as secondary, and report exact intervals in Tables 5–6. Place [dimension-wise human agreement](outputs/rq3/human_validation/analysis/figures/human_human_kappa.pdf) in the appendix.

*Proposed caption:* **Supplementary validation on 120 outputs from 30 examples.** Humans agree closely on aggregate quality but weakly with the recorded ChatGPT judge. Human scores yield more positive thinking estimates with reviewer-dependent uncertainty, while average acceleration loss remains within the original five-point tolerance. Intervals resample examples with all conditions and raters retained. The secondary human mean is not an independent rater; the study does not include E/F or all 40 original examples.

Additional appendix material should include [dimension scores](outputs/rq3/postprocessing/figures/score_dimensions.pdf), [memory](outputs/rq3/postprocessing/figures/peak_memory.pdf), and [individual quality changes versus speedup](outputs/rq3/postprocessing/figures/quality_change_vs_speedup.pdf). The [knowledge-type plot](outputs/rq3/postprocessing/figures/knowledge_type_exploratory.pdf) is descriptive only. Figures are browsable in the [original experiment gallery](outputs/rq3/postprocessing/index.html) and [human-validation gallery](outputs/rq3/human_validation/analysis/index.html).

## 7. Key lessons for the discussion

1. **Interaction and benefit are separate, evaluator-sensitive questions.** A small thinking × acceleration interaction does not establish a positive thinking benefit. Human scores yield more positive thinking estimates than ChatGPT on matched examples, but uncertainty differs between reviewers. The factorial design separates this evaluator dependence from the acceleration effect.
2. **Useful acceleration is a pipeline and resource trade-off.** The 2.59× generator gain becomes 1.95× under charged-total accounting and requires more allocated GPU memory. Deployment recommendations should report all three quantities.
3. **Context retention and temporal reuse play different roles.** The component study associates RIKV removal with larger image deviations and uncertain quality loss, while AVD accounts for substantial runtime savings at the available score resolution. These conclusions concern the fixed deployment, not universally optimal component settings.
4. **Repeatability is insufficient evidence of evaluator validity.** Exact repeated ChatGPT scores coexist with weak agreement with both humans. Mean bias obscures large output-level discrepancies. Report agreement, absolute errors and representative disagreements alongside aggregate effects; human consensus also remains fallible.
5. **Average retention is more consistent than the thinking-benefit conclusion.** Both humans and the judge support limited average D/B loss within their respective samples, while disagreeing about the thinking effect. This supports a qualified deployment finding, rather than a claim that RegionE universally preserves reasoning or editing correctness.

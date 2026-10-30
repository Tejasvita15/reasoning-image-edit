# RQ1 ChatGPT evaluation analysis

> **Internal/private analysis.** Individual case CSVs include model identities and should NOT be sent to a blinded judge.

## Data validation

- Main responses: 380 complete batch JSON files, 1760 distinct case IDs.
- Scored: 1756; unreadable: 4 (omitted from metric means and paired comparisons).
- Knowledge metric applicable to 932 items; scored on 930 items.
- Every main case is matched to its private manifest record; expected batch sizes and score ranges verified.
- The subsequent `responses(1).zip` contains **all 390** batch JSONs (380 main + 10 repeat). All 20 repeat ratings are scored, and all 1,760 main ratings match the initial `chatgpt_main.zip` exactly. Repeatability statistics are summarized below.

## Definition of aggregate Q

For each scored image, compute `Q = mean(25 × (score − 1))` over its applicable consistency, instruction, quality and, where applicable, knowledge scores. The resulting Q is on a 0–100 scale. This is an explicitly defined *manual UI aggregate*, NOT a claim of official KRIS API equivalence. Per-metric results are reported separately.

## K-test (seed 42, 300 original examples)

This is the primary 300-example Base-vs-Thinking comparison; correction is evaluated only on K-small.

| Model | Condition | n scored | Mean Q | Consistency | Instruction | Quality | Knowledge (applicable) |
|---|---|---:|---:|---:|---:|---:|---:|
| step1x | base | 300 | 53.38 | 3.03 | 2.65 | 4.18 | 2.15 |
| step1x | thinking | 299 | 57.18 | 3.17 | 2.87 | 4.25 | 2.33 |
| unireason | base | 300 | 45.21 | 2.60 | 2.49 | 3.81 | 1.82 |
| unireason | thinking | 300 | 45.40 | 2.33 | 2.83 | 3.52 | 2.31 |

## Paired comparisons

Mean differences and bootstrap intervals are paired by original example UID. For 3-seed K-small, seed-level paired differences are averaged *within UID* before resampling/testing, so 120 correlated outputs are not treated as 120 independent examples. CI: 12,000 example-level bootstrap resamples; p: 50,000 two-sided paired sign-flip draws, with Holm correction across the seven Q comparisons. p-values depend on exchangeability/symmetry assumptions and are exploratory, especially for K-small.

| Comparison (LHS − RHS) | Set | n UIDs | matched outputs | ΔQ | 95% CI | p (Holm across 7) |
|---|---|---:|---:|---:|---|---:|
| Step1X: Thinking − Base | k_test | 299 | 299 | +3.79 | [+1.78, +5.84] | 0.0014 |
| UniReason: Thinking − Base | k_test | 300 | 300 | +0.19 | [-2.01, +2.43] | 0.8755 |
| Step1X: Correction − Thinking | k_small | 40 | 116 | -1.91 | [-4.51, +0.07] | 0.3572 |
| UniReason: Correction − Thinking | k_small | 40 | 120 | -1.56 | [-4.32, +0.90] | 0.5257 |
| Step1X − UniReason: Base | k_test | 300 | 300 | +8.17 | [+5.47, +10.79] | 0.00014 |
| Step1X − UniReason: Thinking | k_test | 299 | 299 | +11.82 | [+9.13, +14.49] | 0.00014 |
| Step1X − UniReason: Correction | k_small | 40 | 118 | +10.60 | [+3.88, +17.26] | 0.01384 |

## Repeat-rating calibration (10 fresh batches, 20 repeated outputs)

All 20 repeats are valid and matched to an original main-rating case using `repeat_of` and the same image SHA-256, UID, model, condition and seed. The 380 main response JSONs are unchanged from the previously analyzed archive.

- **Mean absolute rescore difference (aggregate Q, 0–100): 3.125 points** (bootstrap 95% interval for mean absolute difference: **0.625–6.25**; 20,000 resamples, exploratory).
- **95th percentile of absolute rescore differences: 17.08 points** (empirical linear percentile; highly sensitive to n=20).
- **Exact aggregate Q agreement: 15/20 (75%)**; median absolute difference **0**, maximum absolute difference **25**.
- Mean signed second-minus-first difference: **−1.875** points; this small sample does not establish systematic judging bias.

| KRIS dimension | Paired cases | Exact agreement | Mean absolute 1–5 difference |
|---|---:|---:|---:|
| Visual consistency | 20 | 16 (80%) | 0.25 |
| Instruction following | 20 | 18 (90%) | 0.10 |
| Visual quality | 20 | 17 (85%) | 0.15 |
| Knowledge correctness (applicable only) | 12 | 12 (100%) | 0.00 |

**Scope and interpretation.** The selection is not stratified across all model/condition groups: it contains 10 Step1X Thinking and 10 UniReason Correction results, over **10 source UIDs**. Therefore these diagnostic agreement rates cannot establish calibration quality for every RQ1 condition. No repeat-session metadata was included in the uploaded ZIP, so the displayed model label/fresh-chat settings cannot be independently audited from these responses. The 17.08-point per-image 95th percentile is **not** a cutoff for determining significance of a comparison of group means. Paired bootstrap confidence intervals and paired tests remain the appropriate main analysis, with measurement error as a limitation.

## Research interpretation and limitations

- These are masked, manual ChatGPT UI ratings; prompts group multiple edited images. Blinding helps, but within-batch comparisons may still influence scores.
- Inspect individual rubric dimensions before asserting that reasoning improves instruction compliance or knowledge correctness; Q combines scores with different applicability.
- The K-test sample contains 300 UIDs for Base/Thinking; K-small contains 40 UIDs × 3 seeds for Correction/Thinking; never generalize correction results to the entire K-test without qualification.
- Missing/unreadable outputs are excluded rather than assigned an arbitrary 1. Check sensitivity to missingness and include the unreadable-case table in reporting.
- Category and individual-case tables are descriptive/exploratory, not human-verified failure taxonomy labels. Human RQ1.3 annotation remains outstanding.
- An independent re-rating of 20 selected images is now available. It gives a preliminary within-judge repeatability check, not proof of overall judging validity or an inter-judge reliability estimate. The archive does not contain repeat-session setting logs, so fresh-chat provenance must be confirmed separately.

## Files

- `descriptive_ktest.csv`, `descriptive_ksmall.csv`, `descriptive_all_available.csv`, `descriptive_by_seed.csv`
- `paired_comparisons_all_metrics.csv`: the seven planned contrasts for all four rubrics plus Q
- `paired_category_differences.csv`, `paired_seed_differences.csv`, `unreadable.csv`
- `individual_scores_private.csv`, `case_selection_for_human_review_PRIVATE.csv`: model-ID revealed; internal only
- Five figures: K-test aggregate, K-test component rubrics, paired Q effects, per-category thinking effects, repeatability scatter
- `analyze_rq1.py`: reproducible CPU analysis; `rq1_repeat_analysis.py`: reproducible CPU repeatability audit

**New outputs:** `repeat_pair_scores_PRIVATE.csv`, `repeatability_summary.json`, `repeatability_scatter.png`. Treat case-ID-bearing files as private until judging is fully unblinded.

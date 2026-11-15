# RQ3 results and discussion

This report uses the frozen 40-example Step1X experiment at 28 denoising steps, CFG 6 and seed 42. The component-removal comparisons use the predetermined 20-example subset. There are 200 judged quality outputs and 600 additional timing generations; the sample sizes are 40 and 20 examples, respectively.

Under the recorded manual-judge scores, RegionE with thinking enabled changes mean Q by -0.57 points (95% CI [-1.56, 0.00]) and provides 1.95× mean paired charged-total speedup. The frozen 5-point acceptable-loss rule is met. The unaccelerated thinking contrast is -1.61 points (95% CI [-5.94, 2.14]); the original 40-example ChatGPT evaluation does not establish a positive thinking benefit. Its conclusion concerns accelerated thinking-enabled editing under this configuration. A separate human-scored subset, when available below, assesses sensitivity to the evaluator without replacing these original estimates.

## Main conditions

| Condition | Mean Q | Generator seconds | Charged-total seconds | Mean peak allocated VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: |
| A | 53.59 | 56.95 | 66.67 | 24.11 |
| B | 51.98 | 56.89 | 72.07 | 24.11 |
| C | 54.06 | 21.79 | 31.50 | 35.07 |
| D | 51.41 | 22.10 | 37.05 | 35.07 |

A/B: RegionE off, thinking off/on. C/D: RegionE on, thinking off/on.

Each timing value averages within-example medians of three NSCC repetitions. Charged total reconstructs replay plus one cached thinking/offload pass; it excludes model loading, warmup and artifact writes. Peak memory is the per-example maximum over timing repeats, then averaged over examples. It is PyTorch allocated memory, not total board memory.

## Paired quality effects

| Contrast | Mean Q difference | 95% example-bootstrap CI |
| --- | ---: | --- |
| Thinking without RegionE: B−A | -1.61 | [-5.94, 2.14] |
| Thinking with RegionE: D−C | -2.66 | [-7.14, 1.41] |
| Interaction: (D−C)−(B−A) | -1.04 | [-2.40, 0.00] |
| Acceleration with thinking: D−B | -0.57 | [-1.56, 0.00] |
| Acceleration without thinking: C−A | 0.47 | [0.00, 1.41] |

The interaction CI [-2.40, 0.00] lies within ±5 under the frozen rule. This is a small average interaction in the recorded scores; it does not demonstrate that thinking improves editing. The retention decision uses the lower endpoint of the existing two-sided 95% interval, as predeclared. It is not a newly calculated one-sided 95% interval.

38/40 D/B pairs have tied aggregate ratings. Inspect individual outcomes: average retention does not mean every example stays within five points. Confidence intervals resample examples 10,000 times with seed 42; they do not include generation-seed or evaluator uncertainty. No multiplicity correction was added after observing results.

## Component analysis

| Contrast, same 20 examples | Mean Q difference | 95% CI |
| --- | ---: | --- |
| D−B | 0.00 | [0.00, 0.00] |
| E−D: remove RIKV | -4.17 | [-8.85, 0.00] |
| F−D: remove AVD | 0.00 | [0.00, 0.00] |

B, D and F have identical component-score vectors across these 20 examples. Their images are not identical. The zero-width empirical score intervals arise from these ties; they do not establish zero population variability or exact perceptual equivalence.

Removing AVD increases generator time from 21.75s (D) to 34.42s (F), with no detected change in the coarse recorded ratings. All C/D/E examples log 15 AVD-reuse steps and 26 transformer calls versus 56 calls in A/B/F; the call counts include conditional and unconditional CFG branches and are not scheduler-step counts.

Removing RIKV reduces mean peak allocated memory from 35.05 to 24.12 GiB and charged total from 36.58 to 35.49s. Its quality interval extends below −5, so the same retention rule would not establish retention for that removal. The interval also reaches zero; describe the estimated loss and uncertainty rather than claiming a conclusive nonzero effect.

## Image differences

These compare generated outputs against other generated outputs. They measure deviation, not correctness against a ground-truth edit. Images are compared as full-resolution RGB without metric resizing.

| Comparison | Examples | Mean MAE (0–255) | Mean PSNR (dB) | Mean RGB SSIM |
| --- | ---: | ---: | ---: | ---: |
| C-A | 40 | 3.362 | 32.637 | 0.9544 |
| D-B | 40 | 3.431 | 33.433 | 0.9543 |
| E-D | 20 | 8.852 | 24.453 | 0.8483 |
| F-D | 20 | 2.239 | 36.942 | 0.9715 |
| E-B | 20 | 9.181 | 24.382 | 0.8434 |
| F-B | 20 | 2.629 | 34.666 | 0.9660 |

SSIM uses an 11×11 Gaussian window, sigma 1.5, population covariance and a mean across RGB channels. See [image-diagnostic methods](qualitative/README.md) and the saved known-answer checks.

LPIPS was computed separately on the same 160 pairs with the calibrated AlexNet metric, version 0.1, on a CPU. Inputs are full-resolution RGB scaled to [−1, 1]. Package versions, pretrained-weight hashes and per-pair results are saved in the `perceptual/` directory. [Metric reference](https://github.com/richzhang/PerceptualSimilarity).

| Comparison | Examples | Mean LPIPS (lower = less deviation) |
| --- | ---: | ---: |
| C-A | 40 | 0.04244 |
| D-B | 40 | 0.03936 |
| E-D | 20 | 0.17371 |
| F-D | 20 | 0.02513 |
| E-B | 20 | 0.18052 |
| F-B | 20 | 0.03074 |

## Case selection and visual observations

Seven cases were selected by recorded post-hoc rules: typical score/speed, fast with a small score change, largest acceleration loss, largest thinking loss and gain, largest RIKV-removal loss, and greatest F/D pixel variation among tied ratings. All examples remain in the quantitative estimates. See [the case catalogue](qualitative/candidates.md) for exact rules, scores, generated rewrites and attributed judge rationales.

**count_change__48:** The source shows eight forks and requests six. The saved rewrite says to remove four forks, which is an incorrect count adjustment. A/C retain eight visible forks; B/D are nearly blank. This illustrates both a rewrite error and failure to execute even that rewrite, rather than an acceleration-specific failure.

**mathematics__6:** A/C replace the original large-and-small-circle layout with several overlapping circles. B/D preserve the large circle and move the green circle toward its center, but also introduce a blue filled patch. This is a relative improvement consistent with the higher judge scores, not an error-free edit.

**count_change__47:** The requested output has six interfaces, but B/D/F show a blank charger face without the requested visible ports despite Q=83.33. E also changes the charger, plug and background substantially. The removal-of-RIKV example supports visible context disruption while exposing a mismatch between the judge rating and exact counting correctness.

**position_movement__30:** B and D both show three cups where the source has two: the left cup remains while another cup appears above the right one. Their broadly similar failure receives different scores (B=58.33, D=41.67). Treat this as a score-selected discrepancy case, not independent visual proof that RegionE caused a 16.67-point semantic loss.

These notes are assistant visual inspection of the displayed source/output grids, not an independent human review or replacement scoring pass. The original ratings are unchanged. The charger example in particular makes the validity of the manual evaluator an important limitation, even where repeatability is high.

## Supplementary human validation: 30 examples

Two human reviewers independently rated the same 120 A–D outputs from a fixed 30-example subset (10 examples per knowledge type). The subset was selected after the original experiment and before human scoring. Eight practice ratings were excluded. These human results supplement the original 40-example ChatGPT analysis; they do not replace its scores or increase its sample size.

Human–human Q agreement was ICC(A,1) = 0.899 (95% CI [0.819, 0.943]). Human–ChatGPT agreement was weak: ICC(A,1) = 0.027 and 0.134 for Reviewers 1 and 2, respectively. Thus, agreement between humans does not validate the automated judge, and agreement itself is not proof of objective correctness.

| Same 30 examples | Reviewer 1: Q difference [95% CI] | Reviewer 2: Q difference [95% CI] |
| --- | ---: | ---: |
| Thinking without RegionE: B−A | 8.19 [2.43, 14.03] | 6.18 [-2.22, 14.17] |
| Thinking with RegionE: D−C | 7.29 [1.39, 13.40] | 6.25 [-2.09, 14.38] |
| Interaction: (D−C)−(B−A) | -0.90 [-3.89, 2.08] | 0.07 [-1.81, 2.01] |
| Acceleration with thinking: D−B | -1.46 [-3.96, 0.69] | -0.83 [-2.22, 0.21] |

Thinking effects are evaluator-sensitive: both humans give positive point estimates, whereas the matched ChatGPT subset gives negative estimates. Reviewer 1’s thinking intervals exclude zero; Reviewer 2’s include zero. This does not establish a robust positive thinking effect across evaluators. Both human D−B intervals retain lower bounds above −5, supporting limited average loss under the original tolerance as a supplementary sensitivity check.

The intervals resample 30 example clusters 10,000 times, preserving all four conditions and reviewer pairings. These are two fixed reviewers, with independence and prior exposure recorded by attestation. The equal-weight human mean is secondary, not adjudicated ground truth. The supplementary analysis does not validate E/F component ablations, the remaining 10 examples, or additional generation seeds.

[Human-validation results](../human_validation/analysis/RESULTS.md) · [Human-validation charts](../human_validation/analysis/index.html) · [Human-validation report and provenance](../human_validation/analysis/report.json)

## Limits and reporting scope

- One generation seed and 40 examples; the component subset has 20 examples. Timing repeats do not test seed robustness.
- One manual ChatGPT judging pass, an operator-recorded UI model label and coarse 1–5 component ratings. This adaptation is not the published KRIS aggregate/API evaluation.
- Calibration agreement was exact on 20 paired re-scores, but its session metadata was corrected retrospectively under operator attestation, without independent transcript verification. Repeatability does not establish correctness.
- Human–judge validation covers only 30 A–D examples and two fixed reviewers. Human–ChatGPT agreement is weak; thinking effects depend on the evaluator. No human E/F validation or complete 40-example human analysis is available.
- Region overlays show saved token-selection snapshots at zero-based step 5, not attention maps or a complete spatial trajectory.
- Category and case analyses are exploratory. The data do not support transferring these conclusions to RQ1, other step counts, checkpoints or deployment settings.

## Reproducibility

Generation protocol: `a2001fe4612dfee6ae489c243b0fb204b1f04ba39f4877bf0f53853fe7f1438c`.

Evaluation protocol: `095c775e5aae19c2469c5647e717d9209b14a9887ae43ec7410f50657e6af2e2`.

Original records are preserved under `outputs/rq3/evidence/`. The local image lookup is separate from their unchanged NSCC paths. `run_manifest.json` records input and postprocessor hashes and local package versions. The canonical main and component effect intervals reproduce locally.

[Browse all figures](index.html) · [Original canonical report](canonical_report.json) · [Derived tables](tables/) · [Qualitative cases](qualitative/candidates.md)

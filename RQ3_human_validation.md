# RQ3 human–judge validation

## Purpose and materials

This validation measures agreement between two independent human reviewers and the existing ChatGPT judge on the same generated outputs. It also checks whether the RQ3 interpretation changes under human scoring, while retaining the original judge results. Manual review requires no new generations, API key, NSCC allocation or Conda environment; the completed statistical analysis uses the CPU `ai6132` analysis environment.

The subset is recorded in [data/rq3_human_validation_subset.json](data/rq3_human_validation_subset.json): **30 examples, with 10 from each knowledge type**, selected deterministically using seed 42 and sorted pools from the frozen 40-example RQ3 set. Selection uses metadata, not scores or observed failures. This subset was selected **after generation and analysis, before human scoring**; it must not be described as preregistered before the experiment. The existing [rq3_subset.json](data/rq3_subset.json) contains the separate **20-example component-ablation subset**.

Each reviewer assesses conditions A–D for all 30 examples: **120 outputs per reviewer, 240 human assessments in total**. Conditions E/F are excluded, so this validation does not directly validate the component-ablation findings. The original project plan proposed human validation across both models; this packet implements its Step1X RQ3 portion only.

| Reviewer | Packet to open | Ratings to complete |
| --- | --- | --- |
| 1 | [reviewer_1/index.html](outputs/rq3/human_validation/reviewer_1/index.html) | `reviewer_1/ratings.csv` |
| 2 | [reviewer_2/index.html](outputs/rq3/human_validation/reviewer_2/index.html) | `reviewer_2/ratings.csv` |

Each packet contains its own instructions, anonymized cases, independently ordered outputs, `practice_ratings.csv` and `reviewer_metadata.json`. It can be opened locally in a browser. Case pages contain the frozen per-item judging rubric, with only identifiers and attachment filenames changed for the human packet. Of the 120 main outputs, 64 require knowledge scores and four have a viewpoint reference image. These applicability flags are prefilled in the packets.

From the repository root, open either packet in the default browser:

```bash
python3 -c "from pathlib import Path; import webbrowser; webbrowser.open(Path('outputs/rq3/human_validation/reviewer_1/index.html').resolve().as_uri())"
python3 -c "from pathlib import Path; import webbrowser; webbrowser.open(Path('outputs/rq3/human_validation/reviewer_2/index.html').resolve().as_uri())"
```

Packets have already been generated. The reproducible builder is [scripts/prepare_rq3_human_validation.py](scripts/prepare_rq3_human_validation.py); `python3 scripts/prepare_rq3_human_validation.py` recreates them only when the destination is absent. It refuses to overwrite a review folder or change an existing subset. The completed packets, original submissions and analysis are included in this branch. Keep a separate durable backup; caches and unrelated run directories under `outputs/` remain ignored.

## Coordinator instructions

1. Freeze the subset and packet versions before either reviewer starts. Retain the private mapping, generation/evaluation identities and source-image hashes. Do not change membership after inspecting human scores.
2. Send each reviewer **only their individual reviewer folder**. Keep the private mapping, condition labels, ChatGPT responses, rewrites, latency records, selected-region plots and existing findings out of the packet. Do not give reviewers the complete repository or analysis bundle for this task.
3. Select two reviewers who can apply the rubric independently. Ask each to disclose previous exposure to the outputs, findings or judge scores in `reviewer_metadata.json`; prior exposure cannot be removed by anonymization.
4. Run the four practice cases before the main ratings. They use one existing A output from each of four calibration examples disjoint from the 30-example subset: `abstract_reasoning__22`, `anomaly_correction__60`, `biology__13` and `color_change__18`. Reviewers may discuss rubric interpretation during practice. Record clarifications and apply the same instructions to both reviewers. Practice scores are excluded from validation statistics.
5. After practice, collect the main ratings independently. Reviewers must not discuss cases or see each other's scores until both original submissions are preserved. Any later adjudication is a separate record, not a replacement for independent ratings.

## Reviewer instructions

Open your packet's `index.html`, read its `README.md`, and inspect cases in the supplied order. First complete the four practice rows. Then complete all 120 main rows in `ratings.csv`. Excel, Numbers or another spreadsheet editor is sufficient; export the result as **CSV UTF-8**, retaining the column names, case IDs and row count. Save regularly and take breaks after approximately 20 outputs. Allow roughly **4–8 hours per reviewer** at 2–4 minutes per output; difficult cases may take longer.

For each case, inspect the original source, the edited output, the original instruction, and any supplied knowledge explanation or reference image. Zoom in when counting objects or checking small details. Judge the visible result against the request; a visually changed image does not necessarily satisfy the instruction. Assess each output independently on the absolute rubric, without ranking it against other cases. Do not use ChatGPT or another model to supply your human ratings.

Enter integer scores from **1 to 5**, following the full anchors on the case page:

| Field | What to assess |
| --- | --- |
| `consistency_score` | Preservation of elements not requested to change; do not penalize the requested modification itself. |
| `instruction_score` | Presence and correctness of the requested changes; distinguish an attempted edit from a correct result. |
| `quality_score` | The edited image's coherence, naturalness and artifacts. Assess this dimension from the edited image alone. |
| `knowledge_score` | Compliance with the supplied knowledge rubric, only where the packet marks it applicable. |

Knowledge applicability follows the **official rubric family**, not the example's factual/conceptual/procedural grouping. Leave `knowledge_score` blank when it is not applicable. For applicable cases, if `instruction_score` is 1 or 2, the frozen rubric requires **`knowledge_score = 1`**. Preserve this rule even if you would design a different rubric. Viewpoint cases include a ground-truth image: use it for instruction scoring only.

Use `status=ok` for a scoreable case and provide every applicable score. Use `status=unreadable` if a required image is missing or unreadable enough to prevent scoring; leave **all** score fields blank and explain the problem. A poor edit is not an unreadable case. Enter an evidence-based `rationale` covering the applicable dimensions. Optional `confidence` values are 1 (low), 2 (moderate), or 3 (high); record ambiguous instructions, absent knowledge explanations or expertise limitations in `ambiguity_note`. Do not invent missing reference information or use confidence to alter the scoring scale.

Complete `reviewer_metadata.json`, including your reviewer ID, background, prior exposure, practice completion time, main start/completion times and confirmation of independent rating. Return **`ratings.csv`, `practice_ratings.csv` and `reviewer_metadata.json`**. Keep a copy of your original submission.

## Completed validation and analysis

Both submissions are complete: **240 main human ratings** of the same 120 A–D outputs from 30 examples, with **eight practice ratings excluded**. There are no missing or unreadable main outputs. The [Reviewer 1 audit](outputs/rq3/human_validation/validation/reviewer_1_check.md) and [Reviewer 2 audit](outputs/rq3/human_validation/validation/reviewer_2_check.md) preserve the original submissions and document their checks. Reviewer 1's practice timestamp was corrected by the user, with both metadata versions retained. Reviewer 2's uppercase applicability flags and two encoding-damaged optional notes are nonblocking: parsing accepts boolean case and displays invalid bytes as explicit escapes in derived text. Submitted ratings remain unchanged.

Each human's $Q$ is the mean of $25(s-1)$ over applicable dimensions. The analysis uses quadratically weighted Cohen's $\kappa$, Spearman's $\rho$, and **ICC(A,1), two-way mixed effects, absolute agreement, single measurement**, for the named fixed rater pair. Confidence intervals use 10,000 example-cluster bootstrap replicates (seed 42), retaining all four conditions and both reviewers within each sampled example. The 120 outputs are not 120 independent examples. These high-level measures were specified before joint analysis; the executable implementation was completed after submissions, and a method auditor inspected point estimates before code finalization. This was not preregistered or an implementation completed without access to outcomes. The exact decisions are recorded in the [analysis plan](reports/rq3_human_analysis_plan.json).

| Comparison | Result |
| --- | --- |
| Human–human ordinal agreement | Quadratic $\kappa$: consistency 0.498, instruction 0.901, visual quality 0.844, knowledge 0.831 |
| Human–human aggregate agreement | $Q$ ICC(A,1) = 0.899, 95% CI [0.819, 0.943] |
| Reviewer 1 versus ChatGPT | $\rho=0.042$; ICC(A,1) = 0.027 |
| Reviewer 2 versus ChatGPT | $\rho=0.133$; ICC(A,1) = 0.134 |

The human–judge comparisons do **not** establish strong agreement with the recorded judge. Human–human agreement also does not establish objective correctness. On these same 30 examples, the secondary mean of the two human $Q$ scores gives $B-A=7.19$ [0.45, 13.72] and $D-B=-1.15$ [−3.02, 0.31]. Reviewer 2's individual thinking-effect interval includes zero, so the positive thinking estimate is not uniformly established across reviewers. Each human's $D-B$ lower confidence bound exceeds −5, supporting acceptable average acceleration loss under the original margin as a supplementary sensitivity analysis. Preserve the original 40-example judge estimates separately; these results neither replace them nor validate the E/F component ablations. They apply to these two fixed reviewers and the selected examples.

Read the [full results and confidence intervals](outputs/rq3/human_validation/analysis/RESULTS.md), [machine-readable report](outputs/rq3/human_validation/analysis/report.json), [agreement figure](outputs/rq3/human_validation/analysis/figures/agreement_q.pdf), [dimension-agreement figure](outputs/rq3/human_validation/analysis/figures/human_human_kappa.pdf), and [paired-effect figure](outputs/rq3/human_validation/analysis/figures/condition_effects.pdf). Largest-disagreement cases are post-hoc diagnostic selections, not exclusions or estimates of failure prevalence.

The implementation is [src/rq3_human](src/rq3_human/__main__.py). To reproduce the analysis locally from the repository root, use a **new output directory**: the default `outputs/rq3/human_validation/analysis` already contains the completed run and the command refuses to overwrite it.

```bash
conda activate ai6132
python -m src.rq3_human \
  --output-dir "outputs/rq3/human_validation/analysis_rerun_$(date +%Y%m%d_%H%M%S)"
```

This command checks preserved packet identities and submissions, recomputes statistics and figures locally, and leaves original human ratings, ChatGPT judgments, the five-point margin and generation records unchanged.

Protocol basis: the Validation row of *6132 Group Project Writeup.md* (the project plan), the frozen [manual rubric implementation](src/rq3_manual/prompts.py), and [RQ3 findings](RQ3_findings.md). This procedure validates the recorded manual ChatGPT evaluation; it does not reproduce the original benchmark's separate API scoring contexts.

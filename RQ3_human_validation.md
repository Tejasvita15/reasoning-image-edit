# RQ3 human–judge validation

## Purpose and materials

This validation measures agreement between two independent human reviewers and the existing ChatGPT judge on the same generated outputs. It also checks whether the RQ3 interpretation changes when human scores replace automated scores. It requires no new generations, API key, NSCC allocation or Conda environment.

The subset is recorded in [data/rq3_human_validation_subset.json](data/rq3_human_validation_subset.json): **30 examples, with 10 from each knowledge type**, selected deterministically using seed 42 and sorted pools from the frozen 40-example RQ3 set. Selection uses metadata, not scores or observed failures. This subset was selected **after generation and analysis, before human scoring**; it must not be described as preregistered before the experiment. The existing [rq3_subset.json](data/rq3_subset.json) contains the separate **20-example component-ablation subset**.

Each reviewer assesses conditions A–D for all 30 examples: **120 outputs per reviewer, 240 human assessments in total**. Conditions E/F are excluded, so this validation does not directly validate the component-ablation findings. The original project plan proposed human validation across both models; this packet implements its Step1X RQ3 portion only.

| Reviewer | Packet to open | Ratings to complete |
| --- | --- | --- |
| 1 | [reviewer_1/index.html](outputs/rq3/human_validation/reviewer_1/index.html) | `reviewer_1/ratings.csv` |
| 2 | [reviewer_2/index.html](outputs/rq3/human_validation/reviewer_2/index.html) | `reviewer_2/ratings.csv` |

Each packet contains its own instructions, anonymized cases, independently ordered outputs, `practice_ratings.csv` and `reviewer_metadata.json`. It can be opened locally in a browser. Case pages contain the frozen per-item judging rubric, with only identifiers and attachment filenames changed for the human packet. Of the 120 main outputs, 64 require knowledge scores and four have a viewpoint reference image. These applicability flags are prefilled in the packets.

On your Mac, open either packet with:

```bash
cd /Users/aloysiustan/Development/ai6132_group
open outputs/rq3/human_validation/reviewer_1/index.html
open outputs/rq3/human_validation/reviewer_2/index.html
```

Packets have already been generated. The reproducible builder is [scripts/prepare_rq3_human_validation.py](scripts/prepare_rq3_human_validation.py); `python3 scripts/prepare_rq3_human_validation.py` recreates them only when the destination is absent. It refuses to overwrite a review folder or change an existing subset. Keep the generated packets and returned ratings backed up: `outputs/` is ignored by Git.

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

## Analysis after both submissions

Human analysis is a subsequent step; the packet does not contain human results. Preserve the original submissions, validate IDs and applicability, and investigate missing/unreadable cases without silently dropping difficult examples. Compute each human's $Q$ using the existing rule: the mean of $25(s-1)$ over applicable dimensions. Do not replace the original ChatGPT scores or adjust the frozen quality margin in response to human disagreement.

Prespecify **quadratically weighted Cohen's $\kappa$** per ordinal dimension for human–human agreement. Compare each human with the judge using Spearman's $\rho$ and an explicitly documented absolute-agreement ICC on $Q$; the proposed ICC is **ICC(A,1), two-way mixed effects, single measurement**, treating the named human and judge as the fixed pair. Freeze and record the implementation before calculating results. Report score differences and agreement failures alongside these statistics; correlation alone does not establish agreement.

Use paired confidence intervals that resample the **30 examples as clusters**, preserving their four conditions and reviewer pairings. The 120 outputs are not 120 independent examples. Report applicability and missingness counts, and inspect human-scored thinking effects, acceleration effects and their interaction on this subset. Keep subset conclusions distinct from the original 40-example estimates. Human–judge disagreement cases may support qualitative discussion after scoring, but cannot be selected retrospectively into the validation sample.

Protocol basis: the Validation row of [6132 Group Project Writeup.md](</Users/aloysiustan/Downloads/6132 Group Project Writeup.md>), the frozen [manual rubric implementation](src/rq3_manual/prompts.py), and [RQ3 findings](RQ3_findings.md). This procedure validates the recorded manual ChatGPT evaluation; it does not reproduce the original benchmark's separate API scoring contexts.

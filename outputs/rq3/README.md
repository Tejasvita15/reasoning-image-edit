# RQ3 results and evidence

This branch includes the completed Step1X–RegionE experiment and supplementary
human validation. All links below refer to files in the repository.

| Start here | Contents |
| --- | --- |
| [Report findings](../../RQ3_findings.md) | RQ3 answers, interpretation, limitations and proposed report figures |
| [Experiment results](postprocessing/RESULTS.md) | Original 40-example factorial and 20-example component analyses, with the human-validation supplement |
| [Human-validation results](human_validation/analysis/RESULTS.md) | Agreement, matched condition effects and disagreement examples for two reviewers on 30 examples |
| [All-output gallery](index.html) | All 200 quality outputs, grouped by example and condition, with original and available human scores |
| [Experiment figures](postprocessing/index.html) | Statistical plots, selected cases, region overlays and image differences |
| [Human-validation figures](human_validation/analysis/index.html) | Agreement plots and condition-effect comparisons |
| [Reproduction commands](../../RQ3_experiment.md) | Environment, NSCC generation, judging and local analysis |

GitHub displays Markdown, PNG figures, and CSV/JSON data. To use the interactive
HTML galleries, check out this branch and open `outputs/rq3/index.html` in a
browser. HTML pages and images work offline without a model, API key or Python
environment. If needed, serve the repository with standard Python:

```bash
# Run from the repository root, then visit http://localhost:8000/outputs/rq3/.
python -m http.server 8000 --bind 127.0.0.1
```

## Included material

- `evidence/chatgpt_main/`: all 200 quality images, source/reference images,
  judging prompts, 100 original response files, session metadata, validated
  scores and frozen condition mappings.
- `evidence/analysis/`: the original NSCC analysis and its 76 inventoried files,
  including all 800 generation records: 200 quality runs and 600 timing repeats.
- `evidence/calibration/`: both calibration judging passes and their audit.
- `human_validation/`: two complete review packets, original submissions,
  metadata correction history, validation checks and joint statistics.
- `postprocessing/`: report-ready PNG/PDF figures, tables, perceptual diagnostics
  and documented case observations.

A–D cover 40 examples; E/F cover the predetermined 20-example component subset.
Human ratings cover A–D for a separate 30-example subset. The human mean is a
secondary summary, not ground truth or a third independent reviewer. The
original judge results remain available even where human scores disagree.

## Integrity and scope

Run this standard-library check from the repository root after checkout:

```bash
python scripts/verify_rq3_outputs.py
```

The [sharing manifest](../../reports/rq3_shared_outputs.json) records current
file sizes and SHA-256 hashes. Original scientific records, scores, images and
protocol identities are preserved. Historical import inventories retain the
paths and hashes recorded at import time; documented wording-only edits are
listed separately in the sharing manifest. Historical machine paths inside
JSON/CSV provenance are not paths that teammates must recreate. Analysis
resolves the images using repository-relative bundle paths.

The checkout is approximately 715 MiB; identical image copies account for much
of that size and Git stores identical blobs once (approximately 293 MiB unique
content). Model weights, downloaded dependencies, caches, duplicate transfer
folders and future experiment runs remain ignored. This is sufficient for the
completed analysis, but is not a full backup of NSCC pilots or every timing-run
image. Both human reviews are complete; the shared archive exposes condition
identities and scores and must not be used as a blinded packet for new reviewers.

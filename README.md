# AI6132 group experiments

The existing `src/run_experiment.py` runner and `configs/base.yaml`,
`thinking.yaml`, and `correction.yaml` implement the RQ1 Step1X experiments.

RQ3 is a standalone Step1X–RegionE workflow. See
[RQ3_experiment.md](RQ3_experiment.md) for the protocol audit, Conda/CUDA setup,
development pilots, generation, evaluation and analysis commands.

```bash
conda activate ai6132
python -m src.rq3.run --dry-run
python -m pytest -q tests
```

The dry run plans 200 quality outputs without loading models or using an API.
Real generation requires Linux/CUDA and validated development pilots. Reuse the
committed `data/splits.json`; coordinate the proposed shared settings in
`configs/rq3.yaml` and `configs/judge_config.yaml` before test evaluation.

## Completed RQ3 outputs

This branch includes the completed RQ3 reports, figures, selected-case galleries,
judging evidence and human-validation outputs. Start with:

- [RQ3 findings](RQ3_findings.md): report-ready results and interpretation.
- [Output guide and integrity check](outputs/rq3/README.md), including an
  [offline gallery of all 200 outputs](outputs/rq3/index.html).
- [Experiment results and discussion](outputs/rq3/postprocessing/RESULTS.md):
  quantitative results, image comparisons and qualitative cases.
- [Human-validation results](outputs/rq3/human_validation/analysis/RESULTS.md):
  agreement statistics and sensitivity of the findings to human scores.
- [Experiment figure gallery](outputs/rq3/postprocessing/index.html) and
  [human-validation gallery](outputs/rq3/human_validation/analysis/index.html).

Markdown reports and images can be viewed directly on GitHub. For the HTML
galleries, clone the branch and open the corresponding `index.html` in a browser;
GitHub's file view does not render these pages as a website. No GPU, model
checkpoint or API key is required to view the completed outputs.

The Git allowlist includes the completed evidence, post-processing and human
validation directories. Model caches, installed dependencies, intermediate
transfer directories and unrelated run outputs remain ignored. Keep a separate
durable backup of the original NSCC research archive.

## Running on NSCC

NSCC ASPIRE-2A uses PBS. See [the NSCC runbook](RQ3_experiment.md#9-nscc-aspire-2a-pbs-workflow)
and `jobs/nscc/` for setup, offline preparation, resumable GPU arrays and CPU
analysis using an allocated NSCC project. Bulk files use the existing NSCC
`$HOME/scratch` allocation; keep code and small reports in HOME. Submit from
the repository root on NSCC. Set `NSCC_PROJECT` to a project allocated to the
current account; the example otherwise reads the checked-in script default:

```bash
export NSCC_PROJECT="${NSCC_PROJECT:-$(awk '/^#PBS -P / {print $3; exit}' jobs/nscc/gpu.pbs)}"
source jobs/nscc/common.sh
rq3_paths
myquota
qsub -P "$NSCC_PROJECT" -o "$RQ3_LOG_DIR/" jobs/nscc/bootstrap.pbs
```

Complete the documented pilots before submitting production shards.
The runbook includes a CPU storage audit and backup commands for scratch results.

RQ3 generation/scoring and NSCC setup/GPU/finish jobs support Telegram alerts
using the same `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_CHAT_ID` and
`TELEGRAM_NOTIFICATIONS` settings as AI6101. Keep them in a private `.env`;
see [.env.example](.env.example) and the [notification setup](RQ3_experiment.md#telegram-notifications-using-the-ai6101-settings).
Use `--notify` / `--no-notify` for generation or executed scoring. Dry runs and
scoring previews remain silent. `python -m src.notifications --check-config`
checks settings without sending a message or displaying credentials.

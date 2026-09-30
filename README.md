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

NSCC ASPIRE-2A uses PBS. See [the NSCC runbook](RQ3_experiment.md#9-nscc-aspire-2a-pbs-workflow)
and `jobs/nscc/` for setup, offline preparation, resumable GPU arrays and CPU
analysis using project `personal-rtan134`. Bulk files use the existing NSCC
`$HOME/scratch` allocation; keep code and small reports in HOME. Submit from
the repository root on NSCC:

```bash
source jobs/nscc/common.sh
rq3_paths
myquota
qsub -o "$RQ3_LOG_DIR/" jobs/nscc/bootstrap.pbs
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

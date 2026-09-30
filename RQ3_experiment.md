# RQ3: standalone Step1X–RegionE experiment

Branch: `rq3_implementation`. Entry point: `python -m src.rq3.run`.

**Running on NSCC ASPIRE-2A:** start with [Section 9](#9-nscc-aspire-2a-pbs-workflow).
It provides the PBS setup, pilot, array, resume and analysis commands for
`personal-rtan134`, `miniforge3` and `cuda/12.2.2`.

RQ3 generates its own baselines, caches its own thinking rewrites, logs its own
outputs, and performs its own paired analysis. It does not consume RQ1/RQ2
results, selected guidance, UniReason, or 50-step images. The user's revised
independent design takes precedence over the writeup's older RQ2 handoff and
conflicting final split table.

## 1. Coordinate the shared protocol once

The repository already shares **data membership and seed derivation**, but did
not have a complete frozen deployment/evaluation protocol before this change.

| Item | Existing codebase | RQ3 implementation / coordination needed |
| --- | --- | --- |
| Dataset | `Liang0223/KRIS_Bench`; `src/dataset.py` loads an unpinned revision | RQ3 pins a dataset revision; agree on it across RQs |
| IDs | Committed `data/splits.json`: K-dev 90 (30/type), K-test 300 (100/type), K-small 40 (14 factual, 13 conceptual, 13 procedural) | Reuse these exact IDs; hashes are checked. Never regenerate the shared file |
| Eligibility | Exactly one source image; excludes `multi-element_composition` | Reuse eligibility and taxonomy; validate actual inputs when loading |
| Legacy pool | `data/evaluation_set.json` includes excluded multi-image composition examples | Do not use it for this protocol |
| RQ3.2 subset | Absent | `data/rq3_subset.json`: 20 preselected K-small IDs (7 factual, 7 conceptual, 6 procedural), chosen before outputs |
| Checkpoint | Step1X v1p2 name, no revision | Pinned below; all RQs should agree on checkpoint and inference code |
| Resolution/preprocessing | RQ1 leaves library defaults implicit | RQ3: EXIF transpose, RGB, native `size_level=1024`, roughly 1024² pixels with dimensions rounded to multiples of 16; record actual dimensions. This is not a forced 1024×1024 crop |
| Precision/offload | RQ1 uses bfloat16, `cpu_offload: false` | RQ3 proposes bfloat16 + whole-model transfers; coordinate this difference before comparing latency |
| Sampling | RQ1 uses 50 steps, CFG 6 | RQ3 fixes **28 steps, CFG 6**, reflection off, one attempt for every arm |
| Seeds | SHA-256 derivation; global thinker RNG and separate `diffusion_0` seed | Same derivation; 42 primary; optional 1234 and 7 reported separately. Actual initial-noise hashes verify pairing |
| Judge | No evaluator or judge config | Selected: manual ChatGPT judging with pinned KRIS rubric text and a separate recorded UI policy; `configs/judge_config.yaml` retains the optional GPT-4o API route |
| Statistics | No shared implementation | 10,000 paired example-bootstrap resamples, 95% CI, Wilcoxon check. Freeze MMD and equivalence margin δ before test scoring |

Commit the agreed configs, split/subset files and this code revision before
scaling. These settings are a concrete proposal, **not evidence that the other
RQ owners have agreed**. Coordinate exclusive GPU slots for latency measurement.
RQ3 establishes the 28-step thinking gain itself; a benefit at 50 steps in RQ1
cannot supply a missing benefit here.

Current pins (recorded in YAML, resolved while implementing):

| Artifact | Revision |
| --- | --- |
| Step1X v1p2 checkpoint | `ca85b97fd19f2235dc0d6fd3633d1319f169e149` |
| KRIS dataset | `0748003488a5d39a81d2f40290a86d787824b92d` |
| Step1X repository (provenance; actual inference comes from fork below) | `4cca05dfcd6c32dc810d0b4db84c2bf0682a5aeb` |
| Peyton-Chen diffusers | `f5f1c98fa00cb4d0479af1b1b1c17d724345963a` |
| RegionE | `df66d2e60710f66ce801e8b7c151d22c3239d114` |
| Official KRIS evaluator | `cf1475253a13f5e5666826c10d8967e38917468a` |

Sources: [Step1X](https://github.com/stepfun-ai/Step1X-Edit),
[diffusers fork](https://github.com/Peyton-Chen/diffusers/tree/f5f1c98fa00cb4d0479af1b1b1c17d724345963a),
[RegionE code](https://github.com/Peyton-Chen/RegionE/tree/df66d2e60710f66ce801e8b7c151d22c3239d114),
[official KRIS evaluator](https://github.com/mercurystraw/Kris_Bench/tree/cf1475253a13f5e5666826c10d8967e38917468a).

## 2. Environment

Run commands from the repository root. An `ai6132` environment was created on
the current Mac with Python 3.10 and development/analysis packages. Activate it:

```bash
conda activate ai6132
python -m src.rq3.run --dry-run
python -m pytest -q tests
```

On another machine, create the portable base environment:

```bash
conda env create -f environment.yml
conda activate ai6132
```

**GPU deployment is a separate Linux/CUDA installation.** The Mac environment
does not run Step1X–RegionE generation. NSCC setup is automated in Section 9;
the equivalent manual installation below uses the proposed NSCC stack.
PyTorch's packaged CUDA 12.1 runtime is separate from the loaded CUDA 12.2.2
toolkit. Use the same validated wheel/driver for all compared conditions.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-rq3.txt
mkdir -p external
git clone https://github.com/Peyton-Chen/diffusers.git external/diffusers
git -C external/diffusers checkout f5f1c98fa00cb4d0479af1b1b1c17d724345963a
git clone https://github.com/Peyton-Chen/RegionE.git external/RegionE
git -C external/RegionE checkout df66d2e60710f66ce801e8b7c151d22c3239d114
git clone https://github.com/mercurystraw/Kris_Bench.git external/Kris_Bench
git -C external/Kris_Bench checkout cf1475253a13f5e5666826c10d8967e38917468a
python -m pip install -e external/diffusers --no-deps
python -m pip install -e external/RegionE --no-deps
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

The checkouts must stay clean. Local fixes live in `src/rq3`, leaving upstream
files unchanged. RegionE's package metadata pins unrelated dependencies,
including its own Triton version: `--no-deps` preserves the Triton supplied by
the CUDA PyTorch installation. If `pip check` reports these upstream metadata
conflicts, inspect them against imports actually used by v1p2; do not blindly
replace PyTorch's Triton. The CUDA pilot is still required.

Do not install vanilla `diffusers` over the fork. Do not reuse an RQ1 environment
without verifying these revisions. Model and dataset weights download on the
first run; allow sufficient CPU RAM/disk for Step1X's generator and thinker.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
mkdir -p logs reports
python -m pip freeze > reports/rq3_pip_freeze.txt
conda env export > reports/rq3_conda_resolved.yml
nvidia-smi > reports/rq3_gpu.txt
git rev-parse HEAD > reports/rq3_project_revision.txt
git diff > reports/rq3_project_changes.patch
```

Archive those files with results. Changing CUDA, PyTorch, attention backend,
precision or offload mode requires a fresh pilot and separate output directory.

## 3. Conditions and generation budget

| Condition | Thinking | RegionE | Scope |
| --- | --- | --- | --- |
| A | Off | Off | 40 K-small |
| B | On, cached rewrite | Off | 40 K-small |
| C | Off | Full | 40 K-small |
| D | On, same rewrite as B | Full | 40 K-small |
| E | On, same rewrite as B/D | No RIKV contextual injection | 20 frozen subset |
| F | On, same rewrite as B/D | No AVD velocity reuse | 20 frozen subset |

Total: **160 + 40 = 200 quality generations per seed**. B/D are reused within
RQ3.2. Pilots, warmups, repeated timing runs and optional additional seeds are
extra. `--timing-repeats 3` adds 600 unscored generations to the full design.

The backend calls thinking once per example/seed, seeds its global torch RNG,
and caches the exact rewrite. Diffusion always receives an explicit prompt with
native thinking/reflection off and `max_try_cnt=1`. A/C share the raw instruction;
B/D/E/F share the rewrite. Every arm resets the same initial diffusion seed.
Thought caches include protocol, source hash, instruction, seed and token count.

For PyTorch 2.5.1 strict determinism, the native Transformers top-p filter runs
on CPU and returns its scores to the original sampling device. Its processor
position and parameters are preserved, including preceding repetition penalty,
temperature and top-k filtering. The model and seeded token sampling remain on
their original device. The CPU filter and transfers count toward thinking time;
runtime metadata records `thinking_top_p_backend: transformers_cpu`.

RegionE's pinned defaults are `warmup_step=6`, `post_step=2`, `refresh_step='16'`,
`threshold=0.88`, `cache_threshold=0.02`, `erosion_dilation=true`, 28 steps.
Both native embedded `guidance_scale=6` and `true_cfg_scale=6` are explicit;
the negative prompt is the empty string in every arm.
F sets only `cache_threshold=-1`. E uses native attention on the current reduced
stream, removing cached reference/non-selected context; it does not recompute
the full context. The adapter fixes an upstream missing-flash-attention fallback
by choosing SDPA consistently across arms. See `src/rq3/regione.py`.

```bash
python -m src.rq3.run --phase all --dry-run > reports/rq3_plan.json
python -m src.rq3.run --phase rq3_1 --dry-run > reports/rq3_1_plan.json
python -m src.rq3.run --phase rq3_2 --dry-run > reports/rq3_2_plan.json
```

Planning requires no GPU, model download or API key. It checks split hashes,
membership, counts and fixed settings. Test ID truncation is deliberately absent;
`--limit` applies only to development pilots.

## 4. Development pilot and judge calibration

First verify three deterministic, stratified K-dev examples in all six arms:

```bash
python -m src.rq3.run --phase pilot --limit 3 --output outputs/rq3/smoke
python -m src.rq3.verify components --run outputs/rq3/smoke --output reports/rq3_smoke_components.json
```

Inspect images, thoughts, selected regions, initial-noise hashes and cache events.
E should retain D's selected regions and AVD event pattern; F must show no AVD
reuse. The validator reports a failure rather than concealing differing events.
After removing context, trajectories can diverge; investigate any discrepancy
on development examples before scaling. Do not tune on K-small.

**Selected evaluation path: ChatGPT interface, without an API key.** Use
`python -m src.rq3_manual` and the NSCC commands under
[Manual ChatGPT judging](#manual-chatgpt-judging-no-api-key) below. A completed
60-output calibration run is reusable: the exporter verifies its archived
protocol, original images and output hashes, then selects the first 20 distinct
quality outputs in the predetermined generation order. It creates two blinded
passes over the same images. No GPU regeneration is needed for this export.

Subset Q remains `mean(25 * (score - 1))` over consistency, instruction following,
visual quality, and knowledge plausibility where applicable. This is a 0–100
subset aggregate, not the published category-weighted full-benchmark score.
Manual judging preserves the applicable official rubric text, but combines the
three metric prompts in a ChatGPT session. The UI model label, settings, prompt
hashes and raw answers are recorded as a separate evaluation protocol. Coordinate
that choice across RQs before comparing their numerical scores.

The legacy `configs/judge_config.yaml` and `src.rq3.evaluate score` describe the
optional GPT-4o API route. Leave `aggregate_confirmed` unchanged for the manual
route: it does not read that API approval flag or claim API-equivalent ratings.
Do not send manual scores to `src.rq3.analyze`; use `src.rq3_manual analyze`.

The repeatability report proposes MMD as the 95th percentile of absolute repeated
Q differences. Freeze `statistics.minimum_meaningful_difference` at least that
large, and a positive `statistics.equivalence_margin >= MMD` in
`configs/rq3.yaml` before final pilots and main generation. Never choose these
values from test scores. Preserve the exported bundle, both raw judgment passes,
and the session log.

## 5. Determinism and component validation under the final frozen config

Changing judge settings/MMD/δ changes the protocol hash. Use fresh directories
for the final pilots; cached thoughts from the two independent runs must not be
copied between directories.

```bash
python -m src.rq3.run --phase pilot --limit 10 --output outputs/rq3/pilot_first
python -m src.rq3.run --phase pilot --limit 10 --output outputs/rq3/pilot_second
python -m src.rq3.verify determinism --first outputs/rq3/pilot_first --second outputs/rq3/pilot_second --output reports/rq3_determinism.json
python -m src.rq3.verify components --run outputs/rq3/pilot_first --output reports/rq3_components.json
```

The first check requires identical rewrite/thoughts, actual noise hashes and
bitwise-identical image pixels. If the hardware cannot achieve bitwise identity,
document that limitation and rerun the **comparison** with `--tolerance 1`;
every image must have MAE strictly below 1 on the 0–255 scale. Do not relax
PyTorch deterministic algorithms or quietly accept failed checks.

PyTorch 2.5.1 rejects floating-point CUDA `cumsum` under strict determinism;
Transformers 4.55.4 uses it inside nucleus (top-p) filtering. RQ3 therefore
delegates that native filter to CPU through an encoder-instance override,
restored after each thinking call. CPU and CUDA filtering may differ at
floating-point boundaries or ties: this is a fixed implementation choice for
all RQ3 runs, not a claim that old CUDA-filter outputs would be bitwise equal.
Keep the independent rewrite/image checks above. The restriction is documented
in [PyTorch 2.5.1's deterministic-algorithms definition](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/__init__.py)
and the filter in [Transformers 4.55.4's source](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/generation/logits_process.py).

Scaled generation checks the passing reports and their protocol hashes. A changed
protocol needs fresh reports. Reports are measurements, not manually filled
approval flags.

## 6. Generate the standalone experiment

Either run all conditions together (recommended for matched outputs):

```bash
python -m src.rq3.run --phase all --output outputs/rq3/main
```

Or run the two phases into the same output directory, reusing B/D and thoughts:

```bash
python -m src.rq3.run --phase rq3_1 --output outputs/rq3/main
python -m src.rq3.run --phase rq3_2 --output outputs/rq3/main --resume
```

For a restart, rerun the same command with `--resume`. Successful image hashes
are verified before skipping. Errors are logged; failed generations are retried
on resume. A changed protocol or corrupted completed image is rejected. Two
processes cannot write into the same run directory concurrently.

Measure latency on an otherwise idle GPU after warmup:

```bash
python -m src.rq3.run --phase all --output outputs/rq3/main --resume --timing-repeats 3
```

Every launch warms the required arms on a K-dev example, outside the recorded
test budget. Timing repeats use the same prompt and noise and are never scored.
Report median timings over repeats 1–3, not the initial quality run (repeat 0).
The quality rewrite is sampled once for experimental control. Timing repeats
also rerun the identically seeded thinker three times, check that it reproduces
the cached rewrite, and share each measurement across B/D/E/F. Each repeated
pipeline cost charges exactly one corresponding thinking measurement.

`measured_replay_seconds` is directly measured replay latency including native
pipeline stages/transfers and source preprocessing. `charged_total_seconds`
adds one cached thinking pass and its required transfers; it is a reconstructed
full-pipeline cost, **not** a directly measured thinking-plus-edit invocation.
Output file writing and telemetry serialization are outside inference latency.
Use both generator and charged total speedups, and state the offload mode.

Optional three-seed results must be kept separate from the primary single-seed
table. The following command reuses seed 42 in the existing directory and adds
400 quality generations for seeds 1234 and 7:

```bash
python -m src.rq3.run --phase all --seeds 42 1234 7 --output outputs/rq3/main --resume --timing-repeats 3
```

The original SLURM helper is retained for other clusters. **Do not use `sbatch`
or `jobs/runrq3.sh` on NSCC**; use Section 9's PBS scripts. On a SLURM cluster:

```bash
mkdir -p logs
sbatch jobs/runrq3.sh --phase all --output outputs/rq3/main --resume --timing-repeats 3
```

Adjust partition, memory and time to the actual cluster. The script uses
`conda activate ai6132` and `SLURM_SUBMIT_DIR`; it has no hard-coded personal
repository path. Schedule one experiment per GPU for timing.

## 7. Score and analyse

For the selected manual route, follow the NSCC export/import/analysis commands
below. The following commands are retained only for an explicitly chosen API
judge, with its configuration confirmed before generation. Score every condition
using that same frozen API judge. Official
annotations, instructions and prompt templates are loaded from the pinned
checkout. Raw responses (including failed requests) are retained next to scores.

```bash
python -m src.rq3.evaluate score --config configs/rq3.yaml --generations outputs/rq3/main/generations.jsonl --output outputs/rq3/main/scores.jsonl
python -m src.rq3.evaluate score --config configs/rq3.yaml --generations outputs/rq3/main/generations.jsonl --output outputs/rq3/main/scores.jsonl --execute
python -m src.rq3.analyze --config configs/rq3.yaml --scores outputs/rq3/main/scores.jsonl --generations outputs/rq3/main/generations.jsonl --output-dir outputs/rq3/main/analysis --similarity
```

LPIPS may download its perceptual-network weights on first use. The component
diagnostics compare D/E/F with B using PSNR/LPIPS; these measure deviation, not edit correctness.
An identical-image PSNR is infinite and is handled explicitly in output.

The headline contrasts are paired within each example:

```text
G0 = Q(B) - Q(A)
G1 = Q(D) - Q(C)
interaction = Q(D) - Q(C) - Q(B) + Q(A)
acceleration effect with thinking = Q(D) - Q(B)
```

Average seeds within each example before bootstrap for a three-seed analysis:

```bash
python -m src.rq3.evaluate score --generations outputs/rq3/main/generations.jsonl --output outputs/rq3/main/scores_three_seed.jsonl --seeds 42 1234 7 --execute
python -m src.rq3.analyze --scores outputs/rq3/main/scores_three_seed.jsonl --generations outputs/rq3/main/generations.jsonl --output-dir outputs/rq3/main/analysis_three_seed --seeds 42 1234 7 --similarity
```

Report confidence-interval widths and uncertainty. A nonsignificant difference
does not establish equivalence; the chosen interval must fall inside the frozen
margin. If RQ3 has no meaningful unaccelerated thinking gain, describe the effect
of acceleration on thinking-enabled editing without claiming preservation of a
benefit established elsewhere. Knowledge-type/category tables are descriptive.

For qualitative analysis, label expected edit extent and reasoning requirements
separately before looking at generated outcomes. Inspect source + instruction,
selected token regions, B, D/E/F and difference maps. Treat selected regions as
an internal diagnostic, not ground-truth edit masks. Record wrong knowledge,
omitted constraints, correct plan/wrong image and unnecessary edits with a
written codebook; do not infer spatial extent from reasoning difficulty.

## 8. Artifacts and validation status

Each run contains `protocol.json`, the complete `protocol_snapshot.json`,
`generations.jsonl`, sources, per-seed thoughts and images. Generation records
include full config, all seeds, actual noise hash, candidate/stop information,
stage timings, peak VRAM, token partitions, cache events and full/reduced
generator calls. Scoring adds components, Q and raw-response sidecars. Retain
environment/GPU metadata and development validation reports with the results.
The protocol hash includes local implementation/dependency file hashes, as well
as the frozen dataset, subset and evaluator config; changing code invalidates
old readiness reports and prevents an unsafe resume.
`runtime.json` records actual library/CUDA/GPU settings and is checked on resume.

CPU tests cover the experiment plan, split integrity, pairing/replay, resume
integrity, ablation instrumentation and paired analysis. This implementation
was developed on an Apple Silicon Mac: **CUDA model loading, the 40GB deployment,
upstream kernel compatibility, image quality and real speedups require the
documented GPU pilots. No research generations or paid judge calls have been
performed as part of implementation.**

The original implementation passed 40 tests, including numerical attention
subtests and mocked calls through the pinned official evaluator. After the NSCC
additions, the current local suite passes 75 tests; two optional upstream-source
integration tests are skipped when those checkouts are absent.
The two upstream-source integration tests are optional in a fresh environment;
to run them after cloning the dependencies above:

```bash
RQ3_DIFFUSERS_SOURCE=external/diffusers RQ3_REGIONE_SOURCE=external/RegionE RQ3_OFFICIAL_CHECKOUT=external/Kris_Bench python -m pytest -q tests
```

## 9. NSCC ASPIRE-2A PBS workflow

### Resources and scheduling choices

ASPIRE-2A has A100 40GB GPUs. The scripts use `normal` with
`select=1:ngpus=1`; NSCC assigns CPU/RAM according to its GPU policy. Public
pages currently differ on the exact RAM ratio, so do not override it with
an arbitrary `mem` request. ASPIRE-2A+ has different hardware and allocations.
See [ASPIRE-2A specifications](https://nsccsg.github.io/systems/aspire2a/system-specifications/)
and [GPU allocation guidance](https://nsccsg.github.io/quickstart/launch-an-interactive-job/).

All PBS scripts include `#PBS -P personal-rtan134`, `#PBS -q normal`, and
`#PBS -j oe`. Submit from the repository directory; scripts use `PBS_O_WORKDIR`.
This follows the [NSCC batch-job guide](https://nsccsg.github.io/quickstart/submit-a-batch-job/).

| Stage | Script / mode | Request | Initial walltime proposal |
| --- | --- | --- | --- |
| Install and stage everything once | `jobs/nscc/bootstrap.pbs` | 4 CPU cores, 32GB RAM; no GPU | 4 hours |
| Technical smoke test | `gpu.pbs`, `smoke` | 1 GPU; 3 K-dev × 6 conditions | 2 hours |
| Judge calibration images | `gpu.pbs`, `calibration` | 1 GPU; 10 K-dev × 6 conditions | 4 hours |
| Final first pilot, including timing repeats | `gpu.pbs`, `pilot_first`, repeats=3 | 1 GPU; 60 quality + 180 timing generations | 8 hours |
| Independent determinism rerun | `gpu.pbs`, `pilot_second`, repeats=0 | 1 GPU; 60 quality generations | 4 hours |
| Quality generation | `gpu.pbs`, `main` | 4 shards; 1 GPU each; at most 2 simultaneous | 12 hours per shard initially; shorten from pilot estimate |
| Timing repetitions | same shards, repeats=3 | 1 GPU each; at most 1 of your shards at once | Estimate from pilot; default 12 hours |
| Validate, budget, merge, analyse | `jobs/nscc/finish.pbs` | 4 CPU cores, 16GB RAM; no GPU | 2 hours |
| Storage audit, no environment needed | `jobs/nscc/storage.pbs` | 1 CPU core, 2GB RAM; no GPU | 30 minutes |

These are starting requests, **not measured ASPIRE runtimes or guarantees**.
The current FAQ describes automatic routing by requested resources/time and a
120-hour maximum for normal GPU jobs. Do not request that maximum by default.
Queue policy and project eligibility can change; check `qstat -Q` and the
[current NSCC FAQ, job scheduling section](https://help.nscc.sg/aspire2a/faqs/).

The two-hour smoke request routes to `gdev`, which currently permits at most
16 GPUs running across all users. `would exceed overall limit on resource
ngpus in queue gdev` means the job is waiting for queue capacity. Keep it queued;
its `afterok` validation job normally stays in state `H` until smoke succeeds.
Monitor with `qstat -ans`. A new one-GPU request for three hours routes through
`normal` to `g1`, but does not guarantee an earlier start. If choosing to replace
the queued smoke job, cancel its old dependent checker too and submit a new
checker with the new smoke job ID. Avoid duplicate submissions to the same
output directory. See the FAQ's queue routing and limits in section 6.8.

One worker processes an example's A/B/C/D and, where applicable, E/F sequentially.
All conditions, seeds and timing repetitions for an ID stay in one shard, sharing
its rewrite cache. Four shards cover exactly the frozen 40 examples: 10 IDs each,
with 50, 50, 48 and 52 quality generations respectively for the current frozen
subset. This is
independent-example parallelism; it is not multi-GPU model parallelism or MPI.

Use `-J '0-3%2'` for at most two quality workers; `%1` schedules the same four
shards serially if only one GPU is available. Array limits/dependencies are
documented in [NSCC's advanced PBS training, job arrays](https://help.nscc.sg/wp-content/uploads/ASPIRE2A-ADVANCED-JOB-MANAGEMENT-TRAINING_GUIDE.pdf#page=128).
Each worker owns a different output directory and GPU allocation. Do not launch
multiple Python workers on one GPU, split conditions across jobs, or run two
arrays against the same shard directories simultaneously.

For final latency claims, arrange a quiet allocation/reservation and coordinate
with the other RQ teams. `%1` only limits **your** concurrent jobs; it does not
make a shared node exclusive. Host-memory/PCIe activity can affect model-offload
timing even when GPUs differ. No unverified `place=excl` policy is assumed by
these scripts. NSCC documents reservations in the
[same training guide](https://help.nscc.sg/wp-content/uploads/ASPIRE2A-ADVANCED-JOB-MANAGEMENT-TRAINING_GUIDE.pdf#page=135).

### Copy the repository and check the account

On your Mac, replace the SSH destination with your assigned ASPIRE-2A login:

```bash
cd /Users/aloysiustan/Development/ai6132_group
export NSCC_DEST='YOUR_NSCC_USER@YOUR_ASPIRE2A_LOGIN'
ssh "$NSCC_DEST" 'mkdir -p "$HOME/ai6132_group"'
rsync -av --exclude='.nscc/' --exclude='.conda/' --exclude='external' \
  --include='.env.example' --exclude='.env*' \
  --exclude='outputs/' --exclude='logs/' --exclude='__pycache__/' \
  ./ "${NSCC_DEST}:ai6132_group/"
ssh "$NSCC_DEST"
```

Set `NSCC_DEST` to plain `user@hostname` text, without Markdown or `mailto:`.
Keep the braces in `${NSCC_DEST}:...`; macOS zsh otherwise interprets the colon
as a variable modifier and can turn the remote destination into a local path.

On NSCC:

```bash
cd "$HOME/ai6132_group"
module avail miniforge3
module avail cuda/12.2.2
module show cuda/12.2.2
myprojects
myquota
myquota -p personal-rtan134
qstat -Q
source jobs/nscc/common.sh
rq3_paths
mkdir -p reports
printf 'Scratch: %s\nLogs: %s\nResults: %s\n' "$RQ3_STORAGE_ROOT" "$RQ3_LOG_DIR" "$RQ3_OUTPUT_ROOT"
```

`cuda/12.2.2` is the module requested for this deployment; confirm its availability
on **your 2A account**. The scripts fail if it cannot be loaded. A site-approved
module override can be passed via `qsub -v RQ3_CUDA_MODULE=...`; use it consistently
for setup, pilots and production. `miniforge3` is loaded by every job.
Do not run installations, model loading, inference or analysis on login nodes;
keep those in PBS allocations. [NSCC usage guidance](https://nsccsg.github.io/quickstart/dos-and-donts/)

### Storage layout: 50GB HOME, 100TB scratch

Keep the code/configuration checkout and small validation reports in HOME.
All bulk writes default to `$HOME/scratch/ai6132`, where `$HOME/scratch` is
NSCC's existing link to `/scratch/users/<org>/<uid>`. The scripts resolve that
link and refuse to create a fake scratch directory in HOME. They also reject
inherited cache/output overrides resolving outside your scratch allocation.
Unset a conflicting override or point it to scratch before submitting jobs.
The `personal-rtan134` compute project ID does not establish a project-storage
allocation; check `myquota -p personal-rtan134` separately.

| Artifact | Default location |
| --- | --- |
| Linux Conda environment | `$RQ3_STORAGE_ROOT/state/envs/ai6132` |
| Shared HF model/dataset cache | `$RQ3_STORAGE_ROOT/state/cache/huggingface` |
| Conda packages, Torch and other caches | `$RQ3_STORAGE_ROOT/state/cache/` |
| Materialized, validated dataset | `$RQ3_STORAGE_ROOT/state/data/rq3` |
| Pinned editable source checkouts | `$RQ3_STORAGE_ROOT/external/`, linked from repository `external/` |
| Environment stamp and package records | `state/envs/ai6132/.rq3-ready.json`, `state/logs/` under scratch root |
| Scheduler and application logs | `$RQ3_LOG_DIR` = `$RQ3_STORAGE_ROOT/logs/nscc/` |
| Results and resumable shards | `$RQ3_OUTPUT_ROOT/main/shard_0` through `shard_3` |
| Validation and budget reports | Repository `reports/` in HOME |

This avoids depending on the 50GB home quota for weights, packages or outputs.
The pinned checkpoint's required files alone occupy about **38.96 GiB**; the
environment and dataset need additional space. The checkpoint filter omits
only about 9.69 MiB of demos/docs at this revision, so it is not the main saving.
See the [pinned model files](https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2/tree/ca85b97fd19f2235dc0d6fd3633d1319f169e149)
and [NSCC storage allocations](https://nsccsg.github.io/systems/aspire2a/system-specifications/).

All shards share one environment, one model/dataset cache and one prepared
dataset. Pip wheel caching is disabled. The merge references shard images
without copying them. Byte-identical timing PNGs share their matching quality
PNG's disk blocks through hardlinks; different outputs remain independent.
Every requested generation and timing measurement still runs. Compilation
caches use an isolated job temporary directory and are removed on normal exit;
a hard kill can leave scratch temporary files for later inspection.

Scratch is **temporary, not a backup**: files unused for more than 30 days may
be removed without notice. Archive completed results promptly to your Mac or
approved durable project storage. Scheduler-local storage, when provided, is
used only for temporary files; resume state stays on shared scratch.
[NSCC retention policy](https://nsccsg.github.io/policies/data-management-and-retention-policy/)

If `$HOME/scratch` is missing, set `RQ3_SCRATCH_ROOT` to your existing allocated
scratch directory before sourcing `common.sh`. Pass that same variable with
`qsub -v RQ3_SCRATCH_ROOT` on **every** submission; combine it into the existing
`-v` list, for example `-v RQ3_SCRATCH_ROOT,RQ3_MODE=smoke`. Apply the same rule
for a custom `RQ3_STORAGE_ROOT`. Do not change roots or environment prefixes
between attempts. The commands below use the default site scratch link.

Always pass `-o "$RQ3_LOG_DIR/"` as shown below: otherwise PBS can place its own
stdout copy in the HOME submission directory even though application logs use
scratch. This directory must exist before submission (`rq3_paths` creates it).

Check account usage on the login node, then schedule the recursive size audit:

```bash
myquota
STORAGE_JOB=$(qsub -o "$RQ3_LOG_DIR/" jobs/nscc/storage.pbs)
qstat -f "$STORAGE_JOB"
# After completion; substitute the actual PBS job ID:
cat "$RQ3_LOG_DIR/storage_JOB_ID.log"
```

The audit requires no Conda setup, reports HOME and RQ3 directory usage, and
deletes nothing. `df` reports filesystem capacity, **not your personal quota**;
use `myquota` for both block and inode limits. Bootstrap and GPU logs also record
`myquota` when available. These reports do not reserve quota or guarantee future
space: leave headroom in HOME for scheduler output/metadata and other work.
NSCC says not to run recursive `du` scans on login nodes.
[NSCC quota guidance](https://nsccsg.github.io/quickstart/check-quota-and-usage/)

After setup, optional cleanup **inside an allocated CPU job**, with no bootstrap
or package installation running, can remove downloaded Conda package archives:

```bash
source jobs/nscc/common.sh
rq3_activate
conda clean --tarballs --dry-run
conda clean --tarballs --yes
```

Keep extracted Conda packages, the environment, HF model blobs/snapshot links,
prepared data and editable sources throughout the experiment. Do not use
`conda clean --force-pkgs-dirs`, delete HF blobs by hand, or clean active runs.
These cleanup options are documented in the [Conda command reference](https://docs.conda.io/projects/conda/en/latest/commands/clean.html).
Old scratch temporary directories can be inspected in the audit and removed
only after their jobs have ended. Nothing automatically deletes research data.

**Existing installations:** these defaults do not move or delete your old
`.nscc/`, `external/`, `outputs/`, home caches or Conda environment. Recreate the
Linux environment at its final scratch prefix; Conda/editable installs embed
absolute paths. If repository `external/` already lives in HOME, bootstrap
stops without replacing it: use a fresh code-only checkout using the transfer
exclusions above. Preserve old results separately. Storage-related Python
changes affect the protocol hash, so use fresh development pilots before a new
production run; do not append these runs to logs from the earlier code version.
An old prepared-data manifest also needs a fresh `RQ3_PREPARED_DATA` directory
with this code version; pass the new path in every job's `-v` list. Preparation
reuses existing pinned HF blobs, so it does not need to redownload the weights.

### Telegram notifications using the AI6101 settings

RQ3 reuses the notification settings from `ai6101/escalation-budget-gate`:

```dotenv
TELEGRAM_BOT_TOKEN=YOUR_EXISTING_BOT_TOKEN
TELEGRAM_BOT_CHAT_ID=YOUR_EXISTING_CHAT_ID
TELEGRAM_NOTIFICATIONS=auto
```

Keep these in repository `.env`, which is ignored by Git; `.env.example` is a
blank template. The helper reads only these three keys, without executing shell
commands or interpolating variables. Exported environment variables take
precedence, including deliberately empty values. It does not import unrelated
AI6101 API credentials or settings. No credentials are written into experiment
configs, protocol manifests or scheduler arguments.

The local private `.env` can be copied separately to NSCC. From your Mac, after
the code-only transfer above and with `NSCC_DEST` already set:

```bash
scp .env "${NSCC_DEST}:ai6132_group/.env"
ssh "$NSCC_DEST" 'chmod 600 "$HOME/ai6132_group/.env"'
```

On NSCC, check that the settings are present without sending a message or
displaying their values. This helper uses only Python's standard library:

```bash
cd "$HOME/ai6132_group"
module load miniforge3
python -m src.notifications --check-config
```

`configured: true` confirms both credential fields are populated; it does not
validate the bot, destination or network connectivity. Jobs read `.env` directly,
so the normal PBS commands need no additional `-v` credentials. For a file kept
elsewhere, export its absolute path as `TELEGRAM_ENV_FILE` and include
`TELEGRAM_ENV_FILE` in each job's existing `qsub -v` list. Do not `source .env`
or use `qsub -V` to transfer secrets.

`TELEGRAM_NOTIFICATIONS=auto` follows the reference project's one-hour threshold.
For generation, the initial estimate is 180 seconds per planned image, including
timing repeats; scoring uses 60 seconds per planned quality output. These are
notification heuristics, not measured runtimes or PBS walltime recommendations.
`on` enables all runs and `off` disables them. Direct generation and executed
scoring also accept `--notify` / `--no-notify`, which override the mode:

```bash
# This plan remains silent even with --notify; no models or messages are sent.
python -m src.rq3.run --dry-run --notify

# Enable a short smoke job through the scheduler; secrets stay in .env.
qsub -o "$RQ3_LOG_DIR/" -l walltime=02:00:00 \
  -v RQ3_MODE=smoke,TELEGRAM_NOTIFICATIONS=on jobs/nscc/gpu.pbs

# Disable notifications for a submission, including wrapper failure alerts.
qsub -o "$RQ3_LOG_DIR/" \
  -v RQ3_ACTION=budget,TELEGRAM_NOTIFICATIONS=off jobs/nscc/finish.pbs
```

Generation and executed scoring report start/resume, crossed 12.5% milestones,
and one terminal status. Counts include only the selected shard/seed workload
and verified saved outputs; ETA uses newly completed work. Messages include the
job ID, array index, host and output path. Prompts, images, raw exceptions and
judge responses are not sent. Milestones follow persisted JSONL records, and
generation completion follows the shard completion marker. Notification calls
are outside recorded inference timings. Score previews and repeatability checks
send nothing. All tests use mocked or disabled transport.

PBS setup and finish jobs send lifecycle alerts using their default walltime
requests for `auto`. GPU wrappers report early startup failures and catchable
interruptions if Python has not already reported the terminal status. A private
per-attempt marker prevents duplicate terminal alerts. `jobs/runrq3.sh` provides
the same fallback for Slurm. Direct `bootstrap.sh`, the storage audit and the
legacy RQ1 `runbase.sh` have no automatic notifications.

Telegram delivery is best-effort: requests use HTTPS with a five-second socket
timeout and no redirects/retries. Unavailable credentials, blocked egress or API
errors generate a generic warning and leave the workload's exit status intact.
NSCC must permit access to `api.telegram.org`; Hugging Face offline mode does not
disable Telegram. A node crash, SIGKILL, or failure before a compatible Python
interpreter is available cannot reliably send an alert. Check PBS status and
artifact completion markers for the authoritative result. See the
[Telegram sendMessage API](https://core.telegram.org/bots/api#sendmessage).

If a delivery test reports `configured: true`, `ConnectionResetError`, and
`delivered: false`, the credentials were loaded but the HTTPS exchange failed
before a usable Telegram response. This does not validate the token/chat or
prove a firewall block. NSCC documents outbound domain filtering; a reset can
also come from a proxy, remote endpoint or transient network issue.

Compare from your Mac using its existing `.env`. This explicitly sends one
test message, regardless of the `auto` notification threshold:

```bash
cd /Users/aloysiustan/Development/ai6132_group
conda activate ai6132
python - <<'PY'
from src.notifications import load_notification_env, TelegramSender
load_notification_env()
sender = TelegramSender()
print({"configured": sender.configured})
print({"delivered": sender("[ai6132] Mac Telegram delivery test")})
PY
```

If the same bot/chat succeeds from your Mac but repeatedly resets on NSCC,
ask `help@nscc.sg` to check HTTPS access to `api.telegram.org:443` from your
ASPIRE-2A login/compute host and confirm any approved proxy configuration.
Provide the hostname, UTC time, PBS job ID and exception class, without
credentials or token-bearing URLs. The sender already uses urllib's default
proxy handling; do not borrow proxy addresses documented for ASPIRE-2A+
containers. See [NSCC's filtering announcement](https://help.nscc.sg/aspire2a/urgent-scheduled-maintenance-for-aspire-2a-on-22-january-2025-9am-to-6-30pm-2-2/).
Experiments can continue while delivery is investigated; monitor PBS and logs.

Install this code before final development pilots: changes to the RQ3 runner or
scorer change the frozen implementation hash. Changing `.env` settings or
notification CLI flags does not change the scientific protocol or model seeds.

### Submit setup and the smoke test

```bash
SETUP_JOB=$(qsub -o "$RQ3_LOG_DIR/" jobs/nscc/bootstrap.pbs)
printf 'Setup: %s\n' "$SETUP_JOB"
SMOKE_JOB=$(qsub -o "$RQ3_LOG_DIR/" -W "depend=afterok:$SETUP_JOB" -l walltime=02:00:00 \
  -v RQ3_MODE=smoke jobs/nscc/gpu.pbs)
SMOKE_CHECK_JOB=$(qsub -o "$RQ3_LOG_DIR/" -W "depend=afterok:$SMOKE_JOB" \
  -v RQ3_ACTION=smoke_check jobs/nscc/finish.pbs)
qstat -ans
```

Setup creates a **Linux** environment with Python 3.10 and Git from conda-forge
using Miniforge, pins PyTorch 2.5.1/torchvision 0.20.1 with CUDA 12.1 wheels,
installs the project requirements and editable pinned checkouts, stages the
checkpoint and dataset, and checks the frozen IDs. It does not copy the Mac's
environment. PyTorch
publishes this wheel combination in its
[previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).

The requested CUDA 12.2.2 toolkit module does not imply a `cu122` PyTorch wheel.
The wrapper checks the installed runtime and uses a conservative NVIDIA driver
floor of 535.54.03. There is no automatic precision, checkpoint or CUDA fallback.
This stack still requires the real GPU pilot; the Mac's tests used CPU PyTorch.

Environment setup uses a lock and a ready stamp. Matching reruns skip package
installation; interrupted setup can resume. An incompatible existing environment
is rejected. RegionE's unrelated strict package metadata is audited separately
from the dependencies actually used; unexpected dependency errors remain fatal.
GPU jobs do not install packages or download data. They validate the staged
images and start with Hugging Face offline mode enabled.

If an older bootstrap stops at `--stage-sources` with
`FileNotFoundError: ... 'git'`, install Git into the interrupted environment
and resume from the same NSCC CPU session:

```bash
cd "$HOME/ai6132_group"
module load miniforge3 cuda/12.2.2
source jobs/nscc/common.sh
rq3_paths
conda install --yes --prefix "$RQ3_ENV_PREFIX" \
  --override-channels --channel conda-forge --freeze-installed git
"$RQ3_ENV_PREFIX/bin/git" --version
bash jobs/nscc/bootstrap.sh --prepare
```

The current bootstrap includes Git for fresh environments and repairs this
missing dependency when resuming interrupted setup. Git is needed both to
stage the pinned sources and to verify their revisions on compute nodes.
The existing Python packages can be reused; do not delete the environment or
create a readiness stamp manually.

If smoke fails with `cumsum_cuda_kernel does not have a deterministic
implementation`, update to the backend containing `_CPUTopPLogitsWarper`.
After publishing the updated branch to GitHub, run the following from the
NSCC login node. First check that the failed smoke job has stopped with
`qstat -ans`; cancel its old held validation job with `qdel JOB_ID` if it still
exists. Then:

```bash
cd "$HOME/ai6132_group"
source jobs/nscc/common.sh
rq3_paths
"$RQ3_ENV_PREFIX/bin/git" pull --ff-only origin rq3_implementation

# Preserve the failed attempt; the backend change has a new generation hash.
if [[ -d "$RQ3_OUTPUT_ROOT/smoke" ]]; then
  RQ3_FAILED_SMOKE=$(mktemp -d "$RQ3_OUTPUT_ROOT/smoke_before_top_p_fix.XXXXXXXX")
  mv -- "$RQ3_OUTPUT_ROOT/smoke" "$RQ3_FAILED_SMOKE/run"
fi

SMOKE_JOB=$(qsub -N rq3_smoke -o "$RQ3_LOG_DIR/" -l walltime=02:00:00 \
  -v RQ3_MODE=smoke,TELEGRAM_NOTIFICATIONS=on jobs/nscc/gpu.pbs)
SMOKE_CHECK_JOB=$(qsub -N rq3_smoke_check -o "$RQ3_LOG_DIR/" \
  -W "depend=afterok:${SMOKE_JOB}" \
  -v RQ3_ACTION=smoke_check,TELEGRAM_NOTIFICATIONS=on jobs/nscc/finish.pbs)
printf 'Smoke: %s; validation: %s\n' "$SMOKE_JOB" "$SMOKE_CHECK_JOB"
qstat -ans
```

This fix changes only the generation implementation. The ready environment,
prepared examples and downloaded weights remain valid; setup need not run
again. Preserve the archive for diagnosis; archived manifests may contain old
absolute paths and are not inputs to the new run. The checker expects the new
smoke run in its default `smoke/` directory. Keep determinism strict and rerun
the independent development pilots before collecting test results.

HTTP(S) access on NSCC is filtered, so a blocked GitHub/Hugging Face download
may need staging from an allowed machine and transfer to NSCC; it is not fixed
by requesting more GPUs or retrying on a login node. Preserve model cache
directory structure and exact source revisions. Do not transfer the Mac Conda
prefix as a substitute for the Linux environment. [NSCC network FAQ](https://help.nscc.sg/aspire2a/faqs/)

Inspect the smoke logs and `reports/rq3_smoke_components.json`. A passing PBS
exit code does not replace visual inspection of outputs and RegionE traces.

### Manual ChatGPT judging (no API key)

If calibration has already finished with `completed: 60, failures: 0`, use those
outputs. Otherwise, submit the calibration image job once from the login node:

```bash
CALIBRATION_JOB=$(qsub -o "$RQ3_LOG_DIR/" -l walltime=04:00:00 \
  -v RQ3_MODE=calibration jobs/nscc/gpu.pbs)
qstat -f "$CALIBRATION_JOB"
```

Export after the job finishes. This is a small CPU task: two CPUs, 8 GB memory,
30 minutes, no GPU or API access. Use the updated branch on NSCC first. Select
one image-capable model in ChatGPT, then enter its actual displayed label below;
keep that model and the UI settings fixed across both passes and main judging.
The model label is operator metadata, not the model's self-reported identity.

```bash
qsub -I -q normal -P personal-rtan134 \
  -l select=1:ncpus=2:mem=8gb -l walltime=00:30:00
cd "$HOME/ai6132_group"
source jobs/nscc/common.sh
rq3_activate
read -r -p 'Exact model label selected in ChatGPT: ' RQ3_JUDGE_LABEL
python -m src.rq3_manual export \
  --generations "$RQ3_OUTPUT_ROOT/calibration/generations.jsonl" \
  --kind calibration --limit 20 --batch-size 2 \
  --judge-label "$RQ3_JUDGE_LABEL" \
  --output-dir "$RQ3_OUTPUT_ROOT/chatgpt_calibration"
exit
```

This creates `chatgpt_calibration.zip` alongside the bundle directory in scratch.
It copies only the selected source/output images, any required official viewpoint
references, prompts and provenance. It does not copy model weights or the Conda
environment. Existing bundles are never overwritten. Resume manual work using
that bundle; an interrupted export before publication can be rerun. Inspect any
leftover export lock before removing it after an abrupt process termination.

Download and extract **on your Mac**, using NSCC's approved login/transfer route
and VPN if required:

```bash
export NSCC_DEST='rtan134@aspire2antu.nscc.sg'
scp "${NSCC_DEST}:scratch/ai6132/outputs/rq3/nscc/chatgpt_calibration.zip" "$HOME/Downloads/"
mkdir -p "$HOME/Downloads/chatgpt_calibration"
unzip "$HOME/Downloads/chatgpt_calibration.zip" -d "$HOME/Downloads/chatgpt_calibration"
cd "$HOME/Downloads/chatgpt_calibration"
mkdir -p responses
cp -n session_log_template.json session_log.json
open .
```

The ZIP is a transport file for you. **Extract it first; upload only one batch's
image files and paste that batch's `prompt.md` into ChatGPT.** Do not upload the
whole ZIP, `private/`, this runbook, generation logs or prior scores: they expose
condition assignments. Keep image filenames intact. ChatGPT supports image
attachments; this workflow uses direct images rather than relying on reading
images inside another file format. [Official attachment guidance](https://learn.chatgpt.com/docs/use-chatgpt)

With the default batch size there are 10 batches in `upload/pass_1/` and 10 in
`upload/pass_2/`: 20 outputs judged twice, in 20 independent chats. Each batch
has two edited outputs and their source images, plus a reference image if its
rubric requires one. The order and opaque image IDs differ between passes.

1. Start a fresh temporary chat for each batch, with memory/custom instructions
   disabled and tools off. Use the same selected model and settings throughout.
2. Attach **all image files** in that batch folder, paste its complete `prompt.md`,
   and submit once. The prompt asks for independent absolute ratings and strict
   JSON. Do not reveal A–F, thinking traces, seeds or which output you prefer.
3. Save the complete JSON answer unchanged as `responses/p1_b001.json`, etc.
   The filename must equal the batch ID. Do not fill scores yourself, average
   answers or retry to obtain a preferred rating. Save the chat transcript for
   audit. If an image is unreadable, the prompt requires null scores; resolve the
   attachment problem and document it before repeating the same fixed batch.
4. Fill that batch's entry in `session_log.json`: actual model/settings,
   `judged_at` with timezone (e.g. `2026-09-30T18:00:00+08:00`), and a unique
   conversation URL or your own session ID. A temporary chat can use an operator
   ID linked to its saved transcript. Never claim an API snapshot or temperature
   that the UI does not expose.
5. Finish pass 1, then repeat in new chats for pass 2 without supplying pass-1
   scores. Keep saved responses so work can resume batch by batch. The importer
   requires a complete pass and rejects missing, duplicate or invalid ratings.

Return the responses and session log **from your Mac**:

```bash
export NSCC_DEST='rtan134@aspire2antu.nscc.sg'
rsync -av "$HOME/Downloads/chatgpt_calibration/responses/" \
  "${NSCC_DEST}:scratch/ai6132/outputs/rq3/nscc/chatgpt_calibration/responses/"
scp "$HOME/Downloads/chatgpt_calibration/session_log.json" \
  "${NSCC_DEST}:scratch/ai6132/outputs/rq3/nscc/chatgpt_calibration/session_log.json"
```

Import and calculate repeatability on NSCC (the same commands also work locally
with the bundle path changed and the project's CPU dependencies installed):

```bash
qsub -I -q normal -P personal-rtan134 \
  -l select=1:ncpus=2:mem=8gb -l walltime=00:30:00
cd "$HOME/ai6132_group"
source jobs/nscc/common.sh
rq3_activate
RQ3_BUNDLE="$RQ3_OUTPUT_ROOT/chatgpt_calibration"
python -m src.rq3_manual import --bundle "$RQ3_BUNDLE" \
  --responses "$RQ3_BUNDLE/responses" --session-log "$RQ3_BUNDLE/session_log.json" \
  --pass-number 1 --output "$RQ3_BUNDLE/judge_first.jsonl"
python -m src.rq3_manual import --bundle "$RQ3_BUNDLE" \
  --responses "$RQ3_BUNDLE/responses" --session-log "$RQ3_BUNDLE/session_log.json" \
  --pass-number 2 --output "$RQ3_BUNDLE/judge_second.jsonl"
python -m src.rq3_manual repeatability \
  --first "$RQ3_BUNDLE/judge_first.jsonl" --second "$RQ3_BUNDLE/judge_second.jsonl" \
  --output "$RQ3_BUNDLE/repeatability.json"
exit
```

No key or network request is used by these commands. Scores are integers 1–5;
Q is calculated by code, with knowledge required only for the applicable
categories. Imported records retain the exact raw answer and its hash, model
label, date/session metadata, generation identity and manual evaluation identity.
For knowledge items, an instruction score of 1 or 2 requires knowledge score 1
under the official rubric.

Review `repeatability.json` and freeze MMD and δ in `configs/rq3.yaml` before the
final pilots below. Do not edit configs while jobs are running. Keep the manual
judge label, UI settings, batch size and judging code unchanged after calibration;
main export checks its policy against this report. If the UI model or rubric
changes, recalibrate. Keep results from this UI protocol separately labelled from
API-judged results in the report. Dataset preparation is unaffected by this choice.

### Final pilots, determinism and walltime estimation

```bash
PILOT1_JOB=$(qsub -o "$RQ3_LOG_DIR/" -l walltime=08:00:00 \
  -v RQ3_MODE=pilot_first,RQ3_TIMING_REPEATS=3 jobs/nscc/gpu.pbs)
PILOT2_JOB=$(qsub -o "$RQ3_LOG_DIR/" -W "depend=afterok:$PILOT1_JOB" -l walltime=04:00:00 \
  -v RQ3_MODE=pilot_second jobs/nscc/gpu.pbs)
VERIFY_JOB=$(qsub -o "$RQ3_LOG_DIR/" -W "depend=afterok:$PILOT2_JOB" \
  -v RQ3_ACTION=validate jobs/nscc/finish.pbs)
BUDGET_JOB=$(qsub -o "$RQ3_LOG_DIR/" -W "depend=afterok:$PILOT1_JOB" \
  -v RQ3_ACTION=budget,RQ3_TIMING_REPEATS=3 \
  jobs/nscc/finish.pbs)
qstat -ans
```

`validate` checks independent rewrites, actual noise hashes, pixels and component
behavior, and writes the reports required by production. For the documented
MAE < 1 fallback, explicitly resubmit validation with
`-v RQ3_ACTION=validate,RQ3_MAE_TOLERANCE=1` and report that tolerance.

Inspect `reports/rq3_nscc_budget.json`. The estimator uses the slowest observed
charged cost per condition, adds one warmup per needed arm, allows 30 minutes
for loading/JIT/startup and applies 30% workload headroom. Its `array_walltime`
is a rounded scheduling recommendation, not a confidence interval. It is
conservative because it charges thinking separately across the replay arms.
Large test edit regions can still cost more than the pilot.

Quality-only budgeting uses `RQ3_TIMING_REPEATS=0`; a full quality-plus-timing
plan uses 3. The latter counts 800 generations total (200 quality + 600 timing),
before warmups. Budgeting that full workload also safely overestimates a timing
resume that already has its 200 quality outputs.

### Run quality shards, then controlled timing, then merge

Use the measured budget to adjust `walltime` below. After the validation report
passes, submit the whole array:

```bash
QUALITY_ARRAY=$(qsub -o "$RQ3_LOG_DIR/" -J '0-3%2' -l walltime=12:00:00 \
  -v RQ3_MODE=main,RQ3_NUM_SHARDS=4,RQ3_TIMING_REPEATS=0 jobs/nscc/gpu.pbs)
printf 'Quality array: %s\n' "$QUALITY_ARRAY"
```

If `VERIFY_JOB` is still queued/running, add
`-W "depend=afterok:$VERIFY_JOB"` to the quality submission. If it has completed,
the runner checks its reports directly. Do not submit a dependency against an
old job ID that PBS has already removed from history.

Once a quiet timing window is arranged, queue timing repetitions and the CPU
merge. The first 200 outputs and thought caches are reused:

```bash
TIMING_ARRAY=$(qsub -o "$RQ3_LOG_DIR/" -J '0-3%1' -W "depend=afterok:$QUALITY_ARRAY" \
  -l walltime=12:00:00 \
  -v RQ3_MODE=main,RQ3_NUM_SHARDS=4,RQ3_TIMING_REPEATS=3 jobs/nscc/gpu.pbs)
MERGE_JOB=$(qsub -o "$RQ3_LOG_DIR/" -W "depend=afterok:$TIMING_ARRAY" \
  -v RQ3_ACTION=merge,RQ3_NUM_SHARDS=4,RQ3_TIMING_REPEATS=3 jobs/nscc/finish.pbs)
printf 'Timing: %s\nMerge: %s\n' "$TIMING_ARRAY" "$MERGE_JOB"
```

Keep the entire array ID quoted, including its `[]` suffix. Dependencies use
`afterok` on the whole array. Merge independently verifies every shard's
completion marker, counts, example membership, output hashes, thoughts and
runtime settings; it cannot publish an incomplete experiment even if a deleted
subjob is ignored by PBS's array exit status.

The merge produces `$RQ3_OUTPUT_ROOT/main/merged/generations.jsonl` with references
to the original shard files. **Keep all shard directories.** Moving results to
another machine requires preserving or deliberately remapping the recorded
absolute paths; downloading only the merged JSONL is insufficient.

Keep four shards even when running serially (`%1`). Changing shard count or
membership mid-run changes cache ownership and is rejected. Additional seeds
can be submitted as a separate, fully planned run with `RQ3_SEEDS=42:1234:7`
and a fresh `RQ3_RUN_DIR`; this generates 600 quality outputs, not 400, because
the new run does not import seed 42 from the existing shards. Use matching
`RQ3_SEEDS` and `RQ3_RUN_ROOT` on its finish job.

### Monitor, interrupt and resume

```bash
qstat -u "$USER"
qstat -ans
qstat -t "$QUALITY_ARRAY"
qstat -f "$TIMING_ARRAY"
tail -f "$RQ3_LOG_DIR/main_JOB_ID.log"
```

The log filename uses the actual PBS job ID, including the array index. Completed
jobs have per-attempt metadata/exit files under each output directory, while
`shard_complete.json` certifies all requested work. A hard scheduler kill may
prevent the shell exit file from being written; use `qstat -xf JOB_ID` while PBS
retains history and inspect the completion marker.

For example, retry just shard 2 after its previous job has ended:

```bash
RETRY_JOB=$(qsub -o "$RQ3_LOG_DIR/" -l walltime=12:00:00 \
  -v RQ3_MODE=main,RQ3_NUM_SHARDS=4,RQ3_SHARD_INDEX=2,RQ3_TIMING_REPEATS=3 \
  jobs/nscc/gpu.pbs)
```

Retrying the entire array is also safe after all previous subjobs stop: completed
images are skipped, and only missing repetitions are generated. Every GPU job
passes `--resume`. Do not change configs, code, dependencies, CPU thread settings,
GPU model or offload mode between attempts. The runtime guard intentionally
rejects incompatible allocations. Every new allocation warms the required arms.

Image/cache writes are atomic. An interrupted, unterminated final JSONL row can
be recovered on resume after preserving a backup; earlier log corruption is
rejected. Resume saves completed outputs, not a half-finished diffusion state.

To cancel your whole array deliberately:

```bash
qdel "$TIMING_ARRAY"
```

If an array fails, its `afterok` dependents will not become a successful pipeline
automatically after you create a separate retry job. Remove obsolete dependent
jobs when needed and submit a fresh finish job after all shards complete.

### Final manual scoring and CPU analysis

After main quality/timing shards have merged, export all 200 repeat-zero outputs
per seed. Use a CPU allocation, activate the environment as above, and reuse the
**exact** calibration model label (and `--ui-settings` if you customized it):

```bash
read -r -p 'Same ChatGPT model label used for calibration: ' RQ3_JUDGE_LABEL
python -m src.rq3_manual export \
  --generations "$RQ3_OUTPUT_ROOT/main/merged/generations.jsonl" \
  --kind main --batch-size 2 --judge-label "$RQ3_JUDGE_LABEL" \
  --calibration-report "$RQ3_OUTPUT_ROOT/chatgpt_calibration/repeatability.json" \
  --output-dir "$RQ3_OUTPUT_ROOT/chatgpt_main"
```

Download/extract `chatgpt_main.zip` and follow the same batch procedure. The main
bundle has one pass: 100 chats at two outputs per batch for seed 42. Timing
repetitions are never judged. Return `responses/` and `session_log.json` to the
`chatgpt_main` directory on NSCC, then run on an allocated CPU node:

```bash
RQ3_BUNDLE="$RQ3_OUTPUT_ROOT/chatgpt_main"
python -m src.rq3_manual import --bundle "$RQ3_BUNDLE" \
  --responses "$RQ3_BUNDLE/responses" --session-log "$RQ3_BUNDLE/session_log.json" \
  --pass-number 1 --output "$RQ3_BUNDLE/scores.jsonl"
python -m src.rq3_manual analyze --bundle "$RQ3_BUNDLE" \
  --scores "$RQ3_BUNDLE/scores.jsonl" \
  --generations "$RQ3_OUTPUT_ROOT/main/merged/generations.jsonl" \
  --output-dir "$RQ3_OUTPUT_ROOT/main/merged/analysis_manual"
```

Use 4 CPUs, 16 GB and up to 2 hours for analysis. Add `--similarity` for optional
LPIPS/PSNR diagnostics once their dependencies/weights are available. The manual
analyzer requires the complete paired score matrix and at least three measured
timing repetitions, validates image/protocol hashes and computes the existing
RQ3 contrasts, component ablations, bootstrap intervals and quality–latency plot.
Use this manual command instead of the API-oriented `RQ3_ACTION=analyze` job.
The report explicitly records the manual evaluation policy. Keep original shard
images available; merged logs reference them. No GPU is needed for judging or
standard statistical analysis.

### Back up results without copying the environment and model cache

After jobs finish, run these transfers **from your Mac**, using your assigned
login or NSCC-approved transfer endpoint. The examples use the default paths;
adjust the remote source if you overrode the storage root.

```bash
export NSCC_DEST='YOUR_NSCC_USER@YOUR_ASPIRE2A_LOGIN'
export RQ3_BACKUP="$HOME/ai6132-rq3-backup/$(date +%Y%m%d)"
mkdir -p "$RQ3_BACKUP/results" "$RQ3_BACKUP/logs" "$RQ3_BACKUP/reports" \
  "$RQ3_BACKUP/environment" "$RQ3_BACKUP/code"
rsync -avH "${NSCC_DEST}:scratch/ai6132/outputs/" "$RQ3_BACKUP/results/"
rsync -av "${NSCC_DEST}:scratch/ai6132/logs/" "$RQ3_BACKUP/logs/"
rsync -av "${NSCC_DEST}:ai6132_group/reports/" "$RQ3_BACKUP/reports/"
rsync -av "${NSCC_DEST}:scratch/ai6132/state/logs/" "$RQ3_BACKUP/environment/"
rsync -av "${NSCC_DEST}:scratch/ai6132/state/envs/ai6132/.rq3-ready.json" \
  "$RQ3_BACKUP/environment/"
rsync -av --exclude='.nscc/' --exclude='.conda/' --exclude='external' \
  --include='.env.example' --exclude='.env*' \
  --exclude='outputs/' --exclude='logs/' --exclude='__pycache__/' \
  "${NSCC_DEST}:ai6132_group/" "$RQ3_BACKUP/code/"
# Verify result contents; an empty itemized list means no differences.
rsync -nrcH --itemize-changes "${NSCC_DEST}:scratch/ai6132/outputs/" "$RQ3_BACKUP/results/"
```

`-H` preserves hardlinks between identical repeat images; copy the complete
output tree in one invocation. No `--delete` is used. The checksum verification
reads all result files; run it after writers stop. For a large backup, arrange
the scan through a compute allocation/approved transfer service. Preserve all
shards, pilot outputs, thought caches, scores/raw judge responses and manifests,
not just the merged directory. Logs still record the original absolute NSCC
paths: the backup is an archive, and analysis on a different machine requires
deliberate path remapping. Keep an unchanged copy for provenance.

The Linux environment, caches and prepared dataset can be recreated from the
recorded pins; do not copy them into your 50GB HOME allocation. No existing
NSCC files are deleted by these commands. The scripts are syntax- and CPU-tested
locally; they have **not** been submitted to NSCC from this development session.

#!/bin/bash

#PBS -N rq2_step1x
#PBS -P personal-ngch0136
#PBS -q normal
# ASPIRE2A enforces 16 CPUs + 110 GB RAM per GPU (A100 40GB).
#PBS -l select=1:ngpus=1:ncpus=16:mem=110G
#PBS -l walltime=24:00:00
#PBS -j oe

# =============================================================
# Step1X RQ2 jobs on ASPIRE2A (PBS Pro). Submit from the repo root.
# Pick the stage with STAGE, in this order:
#
#   1. Smoke test (2 K-dev examples, every RQ2.1 configuration type):
#        qsub -v STAGE=smoke jobs/rq2_step1x_aspire2a.sh
#
#   2. S7 determinism check (10 K-dev examples, two regenerations):
#        qsub -v STAGE=s7 jobs/rq2_step1x_aspire2a.sh
#
#   3. K-dev thought cache (built once, never regenerated):
#        qsub -v STAGE=cache jobs/rq2_step1x_aspire2a.sh
#
#   4. RQ2.1 grid. Resubmitting the same command resumes where it stopped.
#      Whole grid in one job (builds the thought cache first if missing,
#      so step 3 can be skipped):
#        qsub -v STAGE=rq2_1 jobs/rq2_step1x_aspire2a.sh
#      Or one job per grid point (run step 3 first, then):
#        for t in off cached; do for c in 2 6 10; do
#          qsub -v STAGE=rq2_1,THINKING=$t,CFGS=$c jobs/rq2_step1x_aspire2a.sh
#        done; done
#      Optional guidance points if time allows: c in 4 8.
#      Never run two jobs on the same (THINKING, CFG) at once; they share
#      a log file. Never start several THINKING=cached jobs before the
#      cache exists; they would each try to build it.
#
#   5. Validate the RQ2.1 logs before handing images to the judge
#      (no GPU needed; login nodes are not allowed for computation,
#      so request a CPU-only job instead):
#        qsub -v STAGE=check_2_1 -l select=1:ncpus=4:mem=16G jobs/rq2_step1x_aspire2a.sh
#      With the optional points:
#        qsub -v "STAGE=check_2_1,CFGS=2 4 6 8 10" -l select=1:ncpus=4:mem=16G jobs/rq2_step1x_aspire2a.sh
#
#   6. RQ2.2 quality runs: 20 and 28 steps at guidance 6, thinking off/on
#      (360 images; the 50-step points are the RQ2.1 runs):
#        qsub -v STAGE=rq2_2 jobs/rq2_step1x_aspire2a.sh
#
#   7. RQ2.2 latency, all 6 configurations in one job on one node
#      (10 K-dev examples, warm-up, median of 3 repeats):
#        qsub -v STAGE=rq2_2_timing jobs/rq2_step1x_aspire2a.sh
#
#   8. Validate the RQ2.2 runs (and the timing run, if present):
#        qsub -v STAGE=check_2_2 -l select=1:ncpus=4:mem=16G jobs/rq2_step1x_aspire2a.sh
# =============================================================

set -eo pipefail

STAGE=${STAGE:?set STAGE to smoke, s7, cache, rq2_1, check_2_1, rq2_2, rq2_2_timing or check_2_2}
SEED=${SEED:-42}
THINKING=${THINKING:-"off cached"}
CFGS=${CFGS:-"2 6 10"}
TIMING_REPEATS=${TIMING_REPEATS:-1}

# RQ2.2: the guidance selected in RQ2.1 (6 for both thinking modes) and the
# new step counts. The 50-step points are the RQ2.1 runs.
CFG_2_2=${CFG_2_2:-6}
STEPS_2_2=${STEPS_2_2:-"20 28"}
N_TIMING=${N_TIMING:-10}
TIMING_REPEATS_2_2=${TIMING_REPEATS_2_2:-3}

# RQ2.1 is fixed at 50 steps, reflection off (configs/rq2/step1x.yaml).
STEPS=50

CONFIG=configs/rq2/step1x.yaml
SPLITS=data/splits.json
KDEV_CACHE=outputs/rq2/thought_cache/k_dev_seed${SEED}.jsonl

# ---------------------------------------------------------
# Working directory and live log
# ---------------------------------------------------------

# PBS starts jobs in $HOME; go back to where qsub was run (the repo root).
cd "${PBS_O_WORKDIR:-$HOME/reasoning-image-edit}"

mkdir -p logs

# PBS only copies its own stdout file back when the job ends, so write a
# separate log that can be followed live with tail -f.
JOB_TAG=${PBS_JOBID%%.*}
JOB_TAG=${JOB_TAG:-local}
LOG_FILE=logs/rq2_step1x_${STAGE}_${JOB_TAG}.log
exec > "$LOG_FILE" 2>&1

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job ID: ${PBS_JOBID:-local}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Stage: $STAGE | seed $SEED | thinking [$THINKING] | cfg [$CFGS]"
echo "Start: $(date)"
echo "=========================================="

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

module load miniforge3

# Optional: the cu128 PyTorch wheel bundles its own CUDA runtime, so a
# CUDA module is not required. If you need one, check `module avail cuda`
# and set its exact name here.
# module load cuda/<version>

eval "$(conda shell.bash hook)"
conda activate reasoning-image-edit

# /home is only 50 GB on ASPIRE2A; keep model weights out of it.
# Set HF_HOME to your scratch or project directory (check `myquota`).
export HF_HOME=${HF_HOME:-/scratch/users/ntu/$USER/hf_cache}
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HUGGINGFACE_HUB_CACHE"

# S7: deterministic cuBLAS; must be set before Python starts.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null)"
echo "HF_HOME: $HF_HOME"
which python
nvidia-smi || echo "nvidia-smi not available (CPU-only job?)"

python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )
PY

# ---------------------------------------------------------
# Stages
# ---------------------------------------------------------

case "$STAGE" in

smoke)
    OUT=outputs/rq2_smoke
    LOGS=logs/rq2_smoke
    CACHE=$OUT/thought_cache/k_dev_n2_seed${SEED}.jsonl

    # Scratch stage: start clean on every submission.
    rm -rf "$OUT" "$LOGS"

    COMMON=(
        --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED"
        --n_examples 2 --output_dir "$OUT" --log_dir "$LOGS"
        --subquestion 2.1 --num_inference_steps "$STEPS"
    )

    # Thought cache for the 2 examples.
    python -m src.rq2.build_thought_cache \
        --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
        --n_examples 2 --out "$CACHE"

    # Full RQ2.1 grid, with timing repeats to exercise that path.
    python -m src.rq2.run_rq2 "${COMMON[@]}" \
        --thinking off cached --true_cfg_scale 2 6 10 \
        --thought_cache "$CACHE" --timing_repeats 3

    # Repeat of one grid point in a new process (determinism).
    python -m src.rq2.run_rq2 "${COMMON[@]}" \
        --thinking cached --true_cfg_scale 6 \
        --thought_cache "$CACHE" --run_label rep

    # Native live thinking (replay must reproduce it).
    python -m src.rq2.run_rq2 "${COMMON[@]}" \
        --thinking live --true_cfg_scale 6

    python -m src.rq2.check_rq2 \
        --log_dir "$LOGS" --split k_dev --seed "$SEED" --expected_n 2 \
        --thought_cache "$CACHE" --rep_label rep
    ;;

s7)
    OUT=outputs/rq2_s7
    LOGS=logs/rq2_s7
    CACHE_A=$OUT/thought_cache/k_dev_n10_seed${SEED}_a.jsonl
    CACHE_B=$OUT/thought_cache/k_dev_n10_seed${SEED}_b.jsonl
    RUN_DIR=rq2_step1x_think-cached_cfg6_steps${STEPS}

    rm -rf "$OUT" "$LOGS"

    # Two independent thinking passes (separate processes).
    for CACHE in "$CACHE_A" "$CACHE_B"; do
        python -m src.rq2.build_thought_cache \
            --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
            --n_examples 10 --out "$CACHE"
    done

    # Two independent generations from the same cache.
    for LABEL in a b; do
        python -m src.rq2.run_rq2 \
            --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
            --n_examples 10 --output_dir "$OUT" --log_dir "$LOGS" \
            --subquestion 2.1 --num_inference_steps "$STEPS" \
            --thinking cached --true_cfg_scale 6 \
            --thought_cache "$CACHE_A" --run_label "$LABEL"
    done

    S7_STATUS=0

    python -m src.rq2.check_determinism \
        --caches "$CACHE_A" "$CACHE_B" || S7_STATUS=1

    python -m src.rq2.check_determinism \
        --images "$OUT/${RUN_DIR}_a/k_dev/seed_${SEED}" \
                 "$OUT/${RUN_DIR}_b/k_dev/seed_${SEED}" || S7_STATUS=1

    echo "S7 overall: $([ $S7_STATUS -eq 0 ] && echo PASS || echo FAIL)"
    exit $S7_STATUS
    ;;

cache)
    python -m src.rq2.build_thought_cache \
        --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
        --out "$KDEV_CACHE"
    ;;

rq2_1)
    CACHE_ARGS=()

    if [[ " $THINKING " == *" cached "* ]]; then
        # Build the K-dev thought cache on first use; later jobs and
        # resubmissions reuse it (it is never regenerated).
        if [ ! -f "$KDEV_CACHE" ]; then
            python -m src.rq2.build_thought_cache \
                --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
                --out "$KDEV_CACHE"
        fi

        CACHE_ARGS=(--thought_cache "$KDEV_CACHE")
    fi

    # THINKING and CFGS are intentionally unquoted (space-separated lists).
    python -m src.rq2.run_rq2 \
        --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
        --subquestion 2.1 --num_inference_steps "$STEPS" \
        --thinking $THINKING --true_cfg_scale $CFGS \
        --timing_repeats "$TIMING_REPEATS" \
        "${CACHE_ARGS[@]}"
    ;;

check_2_1)
    python -m src.rq2.check_rq2 \
        --log_dir logs/rq2 --split k_dev --seed "$SEED" --expected_n 90 \
        --thinking $THINKING --cfgs $CFGS \
        --thought_cache "$KDEV_CACHE"
    ;;

rq2_2)
    # RQ2.2 quality runs: new step counts at the RQ2.1-selected guidance
    # (CFG_2_2, 6 for both thinking modes). The 50-step points are the RQ2.1
    # runs and are reused. Separate output/log roots keep the RQ2.2 judging
    # bundle to these runs only. Thinking replays the exact RQ2.1 thoughts.
    if [ ! -f "$KDEV_CACHE" ]; then
        echo "Missing $KDEV_CACHE (built in RQ2.1); pull rq2_john first."
        exit 1
    fi

    for STEP in $STEPS_2_2; do
        python -m src.rq2.run_rq2 \
            --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
            --subquestion 2.2 --num_inference_steps "$STEP" \
            --thinking off cached --true_cfg_scale "$CFG_2_2" \
            --thought_cache "$KDEV_CACHE" \
            --output_dir outputs/rq2_2 --log_dir logs/rq2_2 \
            --timing_repeats "$TIMING_REPEATS"
    done
    ;;

rq2_2_timing)
    # RQ2.2 latency: every configuration (thinking off/on x STEPS_2_2 and 50
    # steps, guidance CFG_2_2) timed in this one job on one node, after
    # warm-up, as the median of TIMING_REPEATS_2_2 repeats, on a fixed
    # stratified K-dev subset. Thinking is timed on the same node too.
    OUT=outputs/rq2_2_timing
    LOGS=logs/rq2_2_timing
    CACHE=$OUT/thought_cache/k_dev_n${N_TIMING}_seed${SEED}.jsonl

    # All timings must come from one job: never resume a partial run.
    if [ -e "$LOGS" ]; then
        echo "$LOGS exists. Timings must come from a single job; remove"
        echo "$LOGS and $OUT to measure again."
        exit 1
    fi

    python -m src.rq2.build_thought_cache \
        --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
        --n_examples "$N_TIMING" --timing_repeats "$TIMING_REPEATS_2_2" \
        --out "$CACHE"

    for STEP in $STEPS_2_2 50; do
        python -m src.rq2.run_rq2 \
            --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
            --n_examples "$N_TIMING" \
            --subquestion 2.2 --num_inference_steps "$STEP" \
            --thinking off cached --true_cfg_scale "$CFG_2_2" \
            --thought_cache "$CACHE" --timing_repeats "$TIMING_REPEATS_2_2" \
            --output_dir "$OUT" --log_dir "$LOGS"
    done
    ;;

check_2_2)
    CHECK_STATUS=0

    # STEPS_2_2 is intentionally unquoted (space-separated list).
    python -m src.rq2.check_rq2 \
        --log_dir logs/rq2_2 --split k_dev --seed "$SEED" --expected_n 90 \
        --thinking off cached --cfgs "$CFG_2_2" --steps $STEPS_2_2 \
        --thought_cache "$KDEV_CACHE" || CHECK_STATUS=1

    # Timing run: same thoughts as RQ2.1, and images identical to the
    # RQ2.1 (50 steps) and RQ2.2 (other steps) images.
    if [ -d logs/rq2_2_timing ]; then
        python -m src.rq2.check_rq2 \
            --log_dir logs/rq2_2_timing --split k_dev --seed "$SEED" \
            --expected_n "$N_TIMING" \
            --thinking off cached --cfgs "$CFG_2_2" --steps $STEPS_2_2 50 \
            --thought_cache outputs/rq2_2_timing/thought_cache/k_dev_n${N_TIMING}_seed${SEED}.jsonl \
            --reference_cache "$KDEV_CACHE" \
            --reference_log_dir logs/rq2 logs/rq2_2 || CHECK_STATUS=1
    fi

    exit $CHECK_STATUS
    ;;

*)
    echo "Unknown STAGE: $STAGE"
    exit 1
    ;;

esac

echo
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="

#!/bin/bash

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# UniReason (BAGEL-7B) loads fully onto a single A100-SXM4-40GB.
# Estimated peak VRAM: ~20-25 GB (14 GB weights + KV caches + activations).
# The original eval uses 8 GPUs for data parallelism only, not model parallel,
# so the full model fits on one GPU.  Verified GPU: A100-SXM4-40GB (40960 MiB).
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=rq2_unireason
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# =============================================================
# One-time cluster setup (run manually before first sbatch):
#   module load anaconda
#   conda create -n UniReason python=3.10 -y
#   conda activate UniReason
#   pip install -r ~/reasoning-image-edit/requirements_unireason.txt
#   pip install flash_attn==2.5.8 --no-build-isolation
#   git clone https://github.com/AlenjandroWang/UniReason ~/UniReason
#   huggingface-cli download Alex11556666/UniReason \
#       --local-dir ~/hf_cache/hub/models--Alex11556666--UniReason
#   cd ~/hf_cache/hub/models--Alex11556666--UniReason && cat ema_part_* > ema.safetensors
#
# UniReason RQ2 jobs.  Same stage structure as rq2_step1x.sh.
# Pick the stage with STAGE, in this order:
#
#   1. Smoke test (2 K-dev examples, every RQ2.1 configuration type):
#        sbatch --export=ALL,STAGE=smoke jobs/rq2_unireason.sh
#
#   2. K-dev thought cache (built once, never regenerated):
#        sbatch --export=ALL,STAGE=cache jobs/rq2_unireason.sh
#
#   3. RQ2.1 grid (cfg_text_scale grid, thinking off/cached/live):
#        sbatch --export=ALL,STAGE=rq2_1 jobs/rq2_unireason.sh
#      Or one job per grid point:
#        for t in off cached; do for c in 2 4 6; do
#          sbatch --export=ALL,STAGE=rq2_1,THINKING=$t,CFGS=$c jobs/rq2_unireason.sh
#        done; done
#
#   4. Validate RQ2.1 logs:
#        sbatch --export=ALL,STAGE=check_2_1 jobs/rq2_unireason.sh
#
#   5. RQ2.2 quality runs (20 and 28 steps at cfg 4, thinking off/cached):
#        sbatch --export=ALL,STAGE=rq2_2 jobs/rq2_unireason.sh
#
#   6. RQ2.2 K-test frozen run (run ONCE):
#        sbatch --export=ALL,STAGE=rq2_2_ktest jobs/rq2_unireason.sh
#
#   7. Validate RQ2.2 runs:
#        sbatch --export=ALL,STAGE=check_2_2 jobs/rq2_unireason.sh
# =============================================================

set -eo pipefail

STAGE=${STAGE:?set STAGE to smoke, cache, rq2_1, check_2_1, rq2_2, rq2_2_ktest or check_2_2}
SEED=${SEED:-42}
THINKING=${THINKING:-"off cached"}
CFGS=${CFGS:-"2 4 6"}
TIMING_REPEATS=${TIMING_REPEATS:-1}

CFG_2_2=${CFG_2_2:-4}
STEPS_2_2=${STEPS_2_2:-"20 28"}

FROZEN_THINKING_2_2=${FROZEN_THINKING_2_2:-cached}
FROZEN_STEPS_2_2=${FROZEN_STEPS_2_2:-28}

STEPS=50

CONFIG=configs/rq2/unireason.yaml
SPLITS=data/splits.json
KDEV_CACHE=outputs/rq2_unireason/thought_cache/k_dev_seed${SEED}.jsonl

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Stage: $STAGE | seed $SEED | thinking [$THINKING] | cfg [$CFGS]"
echo "Start: $(date)"
echo "=========================================="

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

module purge
module load anaconda
module load cuda/12.8.0

eval "$(conda shell.bash hook)"
conda activate UniReason

cd ~/reasoning-image-edit

export HF_HOME=$HOME/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

mkdir -p logs

echo "Branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
which python
nvidia-smi

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
    OUT=outputs/rq2_unireason_smoke
    LOGS=logs/rq2_unireason_smoke
    CACHE=$OUT/thought_cache/k_dev_n2_seed${SEED}.jsonl

    rm -rf "$OUT" "$LOGS"

    COMMON=(
        --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED"
        --n_examples 2 --output_dir "$OUT" --log_dir "$LOGS"
        --subquestion 2.1 --num_inference_steps "$STEPS"
    )

    python -m src.rq2.build_thought_cache \
        --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
        --n_examples 2 --out "$CACHE" --backend unireason

    python -m src.rq2.run_rq2 "${COMMON[@]}" \
        --thinking off cached --true_cfg_scale 2 4 6 \
        --thought_cache "$CACHE" --timing_repeats 2

    python -m src.rq2.check_rq2 \
        --log_dir "$LOGS" --split k_dev --seed "$SEED" --expected_n 2 \
        --thought_cache "$CACHE"
    ;;

cache)
    python -m src.rq2.build_thought_cache \
        --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
        --out "$KDEV_CACHE" --backend unireason
    ;;

rq2_1)
    CACHE_ARGS=()

    if [[ " $THINKING " == *" cached "* ]]; then
        if [ ! -f "$KDEV_CACHE" ]; then
            python -m src.rq2.build_thought_cache \
                --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
                --out "$KDEV_CACHE" --backend unireason
        fi
        CACHE_ARGS=(--thought_cache "$KDEV_CACHE")
    fi

    python -m src.rq2.run_rq2 \
        --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
        --subquestion 2.1 --num_inference_steps "$STEPS" \
        --thinking $THINKING --true_cfg_scale $CFGS \
        --timing_repeats "$TIMING_REPEATS" \
        "${CACHE_ARGS[@]}"
    ;;

check_2_1)
    python -m src.rq2.check_rq2 \
        --log_dir logs/rq2_unireason --split k_dev --seed "$SEED" --expected_n 90 \
        --thinking $THINKING --cfgs $CFGS \
        --thought_cache "$KDEV_CACHE"
    ;;

rq2_2)
    if [ ! -f "$KDEV_CACHE" ]; then
        echo "Missing $KDEV_CACHE; run cache stage first."
        exit 1
    fi

    for STEP in $STEPS_2_2; do
        python -m src.rq2.run_rq2 \
            --config "$CONFIG" --splits "$SPLITS" --split k_dev --seed "$SEED" \
            --subquestion 2.2 --num_inference_steps "$STEP" \
            --thinking off cached --true_cfg_scale "$CFG_2_2" \
            --thought_cache "$KDEV_CACHE" \
            --output_dir outputs/rq2_unireason_2 --log_dir logs/rq2_unireason_2 \
            --timing_repeats "$TIMING_REPEATS"
    done
    ;;

rq2_2_ktest)
    OUT=outputs/rq2_unireason_2_ktest
    LOGS=logs/rq2_unireason_2_ktest
    KTEST_CACHE=$OUT/thought_cache/k_test_seed${SEED}.jsonl
    FROZEN_LOG=rq2_unireason_think-${FROZEN_THINKING_2_2}_cfg${CFG_2_2}_steps${FROZEN_STEPS_2_2}_k_test_seed${SEED}.jsonl

    for f in "$LOGS"/*.jsonl; do
        [ -e "$f" ] || continue
        if [ "$(basename "$f")" != "$FROZEN_LOG" ]; then
            echo "$LOGS already holds another configuration ($(basename "$f"))."
            echo "K-test is used exactly once; refusing to run a different configuration."
            exit 1
        fi
    done

    CACHE_ARGS=()

    if [ "$FROZEN_THINKING_2_2" = "cached" ]; then
        if [ ! -f "$KTEST_CACHE" ]; then
            python -m src.rq2.build_thought_cache \
                --config "$CONFIG" --splits "$SPLITS" --split k_test --seed "$SEED" \
                --out "$KTEST_CACHE" --backend unireason
        fi
        CACHE_ARGS=(--thought_cache "$KTEST_CACHE")
    fi

    python -m src.rq2.run_rq2 \
        --config "$CONFIG" --splits "$SPLITS" --split k_test --seed "$SEED" \
        --allow_k_test --subquestion 2.2 \
        --num_inference_steps "$FROZEN_STEPS_2_2" \
        --thinking "$FROZEN_THINKING_2_2" --true_cfg_scale "$CFG_2_2" \
        --output_dir "$OUT" --log_dir "$LOGS" \
        --timing_repeats "$TIMING_REPEATS" \
        "${CACHE_ARGS[@]}"
    ;;

check_2_2)
    CHECK_STATUS=0

    python -m src.rq2.check_rq2 \
        --log_dir logs/rq2_unireason_2 --split k_dev --seed "$SEED" --expected_n 90 \
        --thinking off cached --cfgs "$CFG_2_2" --steps $STEPS_2_2 \
        --thought_cache "$KDEV_CACHE" || CHECK_STATUS=1

    if [ -d logs/rq2_unireason_2_ktest ]; then
        KTEST_CHECK_ARGS=()
        if [ "$FROZEN_THINKING_2_2" = "cached" ]; then
            KTEST_CHECK_ARGS=(--thought_cache outputs/rq2_unireason_2_ktest/thought_cache/k_test_seed${SEED}.jsonl)
        fi
        python -m src.rq2.check_rq2 \
            --log_dir logs/rq2_unireason_2_ktest --split k_test --seed "$SEED" --expected_n 300 \
            --thinking "$FROZEN_THINKING_2_2" --cfgs "$CFG_2_2" --steps "$FROZEN_STEPS_2_2" \
            "${KTEST_CHECK_ARGS[@]}" || CHECK_STATUS=1
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

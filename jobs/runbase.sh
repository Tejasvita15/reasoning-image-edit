#!/bin/bash

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=rq1_base_kdev
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

module purge
module load anaconda
module load cuda/12.8.0

eval "$(conda shell.bash hook)"
conda activate step1x

cd ~/reasoning-image-edit

# ---------------------------------------------------------
# Hugging Face cache
# ---------------------------------------------------------

export HF_HOME=$HOME/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------

mkdir -p logs
mkdir -p outputs/rq1/base

# ---------------------------------------------------------
# Environment check
# ---------------------------------------------------------

echo "Python:"
which python

echo
echo "GPU:"
nvidia-smi

echo
echo "PyTorch:"
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )
PY

# ---------------------------------------------------------
# RQ1 Base / K-dev
# ---------------------------------------------------------

echo
echo "=========================================="
echo "Starting RQ1 Base K-dev"
echo "=========================================="

python -m src.run_experiment \
    --config configs/base.yaml \
    --splits data/splits.json \
    --split k_dev \
    --seed 42

echo
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
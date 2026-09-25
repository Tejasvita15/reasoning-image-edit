#!/bin/bash

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=step1x_base
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Load cluster modules
module purge
module load anaconda
module load cuda/12.8.0

# Enable conda inside batch shell
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate step1x

# Move to project directory
cd ~/reasoning-image-edit

# Hugging Face cache
export HF_HOME=$HOME/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

mkdir -p logs
mkdir -p outputs/base

echo "Python:"
which python

echo "GPU:"
nvidia-smi

echo "PyTorch:"
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "Starting experiment..."

python -m src.run_experiment \
    --config configs/base.yaml \
    --evaluation-set data/evaluation_set.json

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
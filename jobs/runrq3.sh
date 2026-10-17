#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=rq3
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the ai6132_group repository}"
source jobs/notifications.sh
rq3_dry_run=0
for rq3_arg in "$@"; do
    case $rq3_arg in
        --notify) export TELEGRAM_NOTIFICATIONS=on ;;
        --no-notify) export TELEGRAM_NOTIFICATIONS=off ;;
        --dry-run) rq3_dry_run=1 ;;
    esac
done
[[ $rq3_dry_run == 0 ]] || export TELEGRAM_NOTIFICATIONS=off
ai6132_notifications_begin 'RQ3 Slurm generation' 43200
trap 'rq3_code=$?; trap - EXIT; ai6132_notifications_finish "$rq3_code"; exit "$rq3_code"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
module purge
module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate ai6132
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
mkdir -p outputs/rq3
nvidia-smi
python -m src.rq3.run "$@"

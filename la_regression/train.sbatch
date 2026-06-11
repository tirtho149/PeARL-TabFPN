#!/bin/bash
#SBATCH --job-name=la_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s"
#SBATCH --time=00:40:00
#SBATCH --partition=scavenger
#SBATCH --output=la_regression/la_train-%j.out
#SBATCH --error=la_regression/la_train-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
COHORT="${COHORT:-breast}"
cd /work/mech-ai-scratch/tirtho/PeARL-TabFPN
source venv/bin/activate
echo "[la_train] cohort=$COHORT node=$(hostname)"
python la_regression/train_head.py --cohort "$COHORT"
echo "[la_train] done"

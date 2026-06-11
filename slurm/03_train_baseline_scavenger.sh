#!/bin/bash
#SBATCH --job-name=pearl_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s"
#SBATCH --time=24:00:00
#SBATCH --partition=scavenger
#SBATCH --output=logs/pearl_baseline-%j.out
#SBATCH --error=logs/pearl_baseline-%j.err
# PeARL+MLP baseline (the paper reproduction). Cohort via COHORT env (breast|skin|lymph).
#   COHORT=breast PEARL_REPO=$PWD sbatch slurm/03_train_baseline_scavenger.sh
set -euo pipefail
export PYTHONUNBUFFERED=1
COHORT="${COHORT:-breast}"
PEARL_REPO="${PEARL_REPO:-$PWD}"; cd "$PEARL_REPO"; mkdir -p logs
source "${PEARL_VENV:-$PEARL_REPO/venv}/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "[baseline] cohort=$COHORT node=$(hostname)"
python scripts/train_baseline.py --apple-to-apple --cohort "$COHORT" --folds 5 \
    --output-dir "reproduction_results_${COHORT}"
echo "[baseline] done"

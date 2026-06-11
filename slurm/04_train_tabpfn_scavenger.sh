#!/bin/bash
#SBATCH --job-name=pearl_tabpfn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s"
#SBATCH --time=48:00:00
#SBATCH --partition=scavenger
#SBATCH --output=logs/pearl_tabpfn-%j.out
#SBATCH --error=logs/pearl_tabpfn-%j.err
# PeARL+TabPFN head. Needs TABPFN_TOKEN in .env. Cohort via COHORT env.
#   COHORT=breast PEARL_REPO=$PWD sbatch slurm/04_train_tabpfn_scavenger.sh
set -euo pipefail
export PYTHONUNBUFFERED=1
COHORT="${COHORT:-breast}"
PEARL_REPO="${PEARL_REPO:-$PWD}"; cd "$PEARL_REPO"; mkdir -p logs
source "${PEARL_VENV:-$PEARL_REPO/venv}/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "[tabpfn] cohort=$COHORT node=$(hostname)"
python scripts/train_tabpfn.py --apple-to-apple --cohort "$COHORT" --folds 5 \
    --output-dir "reproduction_results_${COHORT}_tabpfn"
echo "[tabpfn] done"

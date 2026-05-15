#!/bin/bash
#SBATCH --job-name=pearl_train_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/pearl_train_baseline-%j.out
#SBATCH --error=logs/pearl_train_baseline-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Phase 03 — PEaRL+MLP baseline (5-fold CV, Breast, ~7 hr on A100).
# Submit:
#   PEARL_REPO=$PWD sbatch slurm/03_train_baseline.sh
#
# Re-runs are safe: per-fold checkpoints/results are written incrementally.

set -euo pipefail
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs

source "$PEARL_VENV/bin/activate"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "[train_baseline] node=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -n1 || echo 'no gpu')"
nvidia-smi || true

python scripts/train_baseline.py --apple-to-apple --n-sections 36 --folds 5
echo "[train_baseline] done"

#!/bin/bash
#SBATCH --job-name=pearl_train_tabpfn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/pearl_train_tabpfn-%j.out
#SBATCH --error=logs/pearl_train_tabpfn-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Phase 04 — PEaRL+TabPFN (5-fold CV, Breast, ~45 hr on A100).
# Submit:
#   PEARL_REPO=$PWD sbatch slurm/04_train_tabpfn.sh
#
# Heaviest job in the pipeline. Runs after Phase 03 so the cheaper
# baseline finishes first and frees the GPU.

set -euo pipefail
# Force unbuffered Python output so progress lands in the log in real time
# instead of only flushing when the job ends.
export PYTHONUNBUFFERED=1
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

if [ -z "${TABPFN_TOKEN:-}" ] && [ -z "${TABPFN_API_KEY:-}" ]; then
  echo "[train_tabpfn] FATAL: TABPFN_TOKEN/TABPFN_API_KEY not set after sourcing .env" >&2
  echo "  Add 'TABPFN_TOKEN=<key>' to $PEARL_REPO/.env (see README.md)." >&2
  exit 1
fi

echo "[train_tabpfn] node=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -n1 || echo 'no gpu')"
nvidia-smi || true

python scripts/train_tabpfn.py --apple-to-apple --n-sections 36 --folds 5
echo "[train_tabpfn] done"

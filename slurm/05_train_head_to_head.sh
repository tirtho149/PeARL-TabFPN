#!/bin/bash
#SBATCH --job-name=pearl_head_to_head
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/pearl_head_to_head-%j.out
#SBATCH --error=logs/pearl_head_to_head-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Phase 05 — bundled head-to-head (5-fold CV, both heads, Breast, ~50 hr).
# Submit:
#   PEARL_REPO=$PWD sbatch slurm/05_train_head_to_head.sh
#
# Alternative to running 03 + 04 as two separate jobs. Useful when GPU
# allocation supports a single 50-hour reservation.

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
  echo "[head_to_head] FATAL: TABPFN_TOKEN/TABPFN_API_KEY not set after sourcing .env" >&2
  echo "  Add 'TABPFN_TOKEN=<key>' to $PEARL_REPO/.env (see README.md)." >&2
  exit 1
fi

echo "[head_to_head] node=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -n1 || echo 'no gpu')"
nvidia-smi || true

python scripts/run_reproduction.py --apple-to-apple --n-sections 36 --folds 5
echo "[head_to_head] done"

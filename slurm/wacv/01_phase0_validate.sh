#!/bin/bash
#SBATCH --job-name=wacv_phase0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/wacv_phase0-%j.out
#SBATCH --error=logs/wacv_phase0-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# WACV Phase 0 — gating validation. THIS IS THE GATE: do not submit
# subsequent phases until wacv_results/phase0/config.json is written.

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs wacv_results/phase0

source "$PEARL_VENV/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

echo "[wacv_phase0] node=$(hostname)"
nvidia-smi || true

python scripts/wacv/phase0_validate.py \
  --output-dir wacv_results/phase0 --only all
echo "[wacv_phase0] done"

#!/bin/bash
#SBATCH --job-name=wacv_phase2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/wacv_phase2-%j.out
#SBATCH --error=logs/wacv_phase2-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# WACV Phase 2 — configuration sweep (Breast + Skin).

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs wacv_results/phase2

source "$PEARL_VENV/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

echo "[wacv_phase2] node=$(hostname)"
nvidia-smi || true

python scripts/wacv/phase2_config_sweep.py \
  --cohorts Breast Skin --folds 5 \
  --axes estimators context precision \
  --output-dir wacv_results/phase2
echo "[wacv_phase2] done"

#!/bin/bash
#SBATCH --job-name=wacv_phase3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/wacv_phase3-%j.out
#SBATCH --error=logs/wacv_phase3-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# WACV Phase 3 — calibration analysis (post-hoc, CPU-only).

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs wacv_results/phase3

source "$PEARL_VENV/bin/activate"

python scripts/wacv/phase3_calibration.py \
  --cohorts Breast Skin Lymph \
  --phase1-dir wacv_results/phase1 \
  --output-dir wacv_results/phase3
echo "[wacv_phase3] done"

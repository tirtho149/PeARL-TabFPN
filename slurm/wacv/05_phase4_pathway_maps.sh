#!/bin/bash
#SBATCH --job-name=wacv_phase4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/wacv_phase4-%j.out
#SBATCH --error=logs/wacv_phase4-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# WACV Phase 4 — pathway activation maps (post-hoc, CPU-only).

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs wacv_results/phase4

source "$PEARL_VENV/bin/activate"

python scripts/wacv/phase4_pathway_maps.py \
  --cohorts Breast Skin Lymph \
  --phase1-dir wacv_results/phase1 \
  --output-dir wacv_results/phase4 \
  --top-k 20
echo "[wacv_phase4] done"

#!/bin/bash
#SBATCH --job-name=wacv_phase5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/wacv_phase5-%j.out
#SBATCH --error=logs/wacv_phase5-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# WACV Phase 5 — compute characterization. Single Breast fold, three
# combinations. Numbers cited in the WACV paper come from this job.

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs wacv_results/phase5

source "$PEARL_VENV/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

echo "[wacv_phase5] node=$(hostname)"
nvidia-smi || true

python scripts/wacv/phase5_compute.py \
  --cohort Breast --n-sections 36 \
  --output-dir wacv_results/phase5
echo "[wacv_phase5] done"

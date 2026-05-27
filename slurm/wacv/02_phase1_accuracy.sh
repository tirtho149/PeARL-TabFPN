#!/bin/bash
#SBATCH --job-name=wacv_phase1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=144:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/wacv_phase1-%j.out
#SBATCH --error=logs/wacv_phase1-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# WACV Phase 1 — core accuracy (3 cohorts × {MLP, TabPFN-v2, TabPFN-3}
# × 5 folds). Wall-clock dominated by 3 cohorts × ~50 hr each — split
# into per-cohort sbatch submissions if your Nova allocation is shorter
# than 144 hr.

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs wacv_results/phase1

source "$PEARL_VENV/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

# Sanity gate — refuse to run Phase 1 without a Phase 0 config.
if [ ! -f wacv_results/phase0/config.json ]; then
  echo "[wacv_phase1] FATAL: wacv_results/phase0/config.json missing."
  echo "  Submit slurm/wacv/01_phase0_validate.sh first."
  exit 2
fi

COHORTS="${COHORTS:-Breast Skin Lymph}"
echo "[wacv_phase1] cohorts=$COHORTS node=$(hostname)"
nvidia-smi || true

python scripts/wacv/phase1_accuracy.py \
  --cohorts $COHORTS --folds 5 \
  --output-dir wacv_results/phase1
echo "[wacv_phase1] done"

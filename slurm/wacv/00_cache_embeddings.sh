#!/bin/bash
#SBATCH --job-name=wacv_cache_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/wacv_cache_embeddings-%j.out
#SBATCH --error=logs/wacv_cache_embeddings-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# WACV Section 1 — per-(cohort, fold) embedding cache.
# Submit:
#   PEARL_REPO=$PWD COHORT=Breast sbatch slurm/wacv/00_cache_embeddings.sh
#   PEARL_REPO=$PWD COHORT=Skin   sbatch slurm/wacv/00_cache_embeddings.sh
#   PEARL_REPO=$PWD COHORT=Lymph  sbatch slurm/wacv/00_cache_embeddings.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
COHORT="${COHORT:-Breast}"
cd "$PEARL_REPO"
mkdir -p logs wacv_results/embeddings_cache

source "$PEARL_VENV/bin/activate"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "[wacv_cache_embeddings] cohort=$COHORT node=$(hostname)"
nvidia-smi || true

python scripts/wacv/cache_embeddings.py --cohort "$COHORT" \
  --output-dir wacv_results/embeddings_cache --folds 5
echo "[wacv_cache_embeddings] done"

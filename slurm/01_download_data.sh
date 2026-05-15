#!/bin/bash
#SBATCH --job-name=pearl_download_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/pearl_download_data-%j.out
#SBATCH --error=logs/pearl_download_data-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Phase 01 — download HEST-1k Breast cohort from HuggingFace (~45 GB).
# Submit:
#   PEARL_REPO=$PWD sbatch slurm/01_download_data.sh
#
# Idempotent — `snapshot_download` resumes partial downloads. Requires
# HF_TOKEN / HUGGINGFACE_HUB_TOKEN in .env (gated dataset).

set -euo pipefail
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs

source "$PEARL_VENV/bin/activate"

# Pick up HF_TOKEN from .env if present so SETUP_DATA.sh's auth check passes.
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "[download] starting SETUP_DATA.sh"
bash SETUP_DATA.sh
echo "[download] done"

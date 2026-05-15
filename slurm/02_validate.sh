#!/bin/bash
#SBATCH --job-name=pearl_validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=nova
#SBATCH --output=logs/pearl_validate-%j.out
#SBATCH --error=logs/pearl_validate-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Phase 02 — structural validation on stub data (~1 min).
# Submit:
#   PEARL_REPO=$PWD sbatch slurm/02_validate.sh
#
# Must pass before launching the expensive training jobs.

set -euo pipefail
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs

source "$PEARL_VENV/bin/activate"

echo "[validate] running scripts/validate.py"
python scripts/validate.py
echo "[validate] done"

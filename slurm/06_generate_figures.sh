#!/bin/bash
#SBATCH --job-name=pearl_generate_figures
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=nova
#SBATCH --output=logs/pearl_generate_figures-%j.out
#SBATCH --error=logs/pearl_generate_figures-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Phase 06 — render BIBM head-to-head figures (~2 min).
# Submit:
#   PEARL_REPO=$PWD sbatch slurm/06_generate_figures.sh
#
# Reads reproduction_results/reproduction_results.json + the per-fold
# predictions and writes seven PNGs to reproduction_results/figures/.

set -euo pipefail
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs

source "$PEARL_VENV/bin/activate"

echo "[figures] starting"
python scripts/generate_figures.py \
  --results-dir reproduction_results \
  --output-dir reproduction_results/figures
echo "[figures] done"

#!/bin/bash
#SBATCH --job-name=pearl_install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=nova
#SBATCH --output=logs/pearl_install-%j.out
#SBATCH --error=logs/pearl_install-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Phase 00 — one-time venv install + editable package install.
# Submit from the repo root:
#   PEARL_REPO=$PWD sbatch slurm/00_install.sh
#
# Creates $PEARL_VENV (default $PEARL_REPO/venv) and installs the package
# in editable mode so `pearl_tabpfn` is importable. Idempotent —
# re-running just refreshes pip + reinstalls deps.

set -euo pipefail
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs

echo "[install] repo=$PEARL_REPO venv=$PEARL_VENV"

if [ ! -d "$PEARL_VENV" ]; then
  echo "[install] creating venv at $PEARL_VENV"
  python -m venv "$PEARL_VENV"
fi

source "$PEARL_VENV/bin/activate"
python -m pip install --upgrade pip wheel
pip install -e .

echo "[install] verifying pearl_tabpfn import"
python -c "import pearl_tabpfn; from pearl_tabpfn import config, data, encoders, baseline, tabpfn_head, reproduction; print('pearl_tabpfn ok')"

echo "[install] done"

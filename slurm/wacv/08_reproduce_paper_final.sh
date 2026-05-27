#!/usr/bin/env bash
#SBATCH --job-name=wacv_paper_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/wacv_paper_final-%j.out
#SBATCH --error=logs/wacv_paper_final-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Final stage of the WACV paper chain — aggregates per-fold predictions
# across cohorts, generates Figures 3-10 + Tables 1-3, renders paper.tex.
#
# Standalone use:
#   PEARL_REPO=$PWD sbatch slurm/wacv/08_reproduce_paper_final.sh
#
# Or as the tail of slurm/wacv/full_paper.sh, which submits this with the
# chain's --dependency=afterok so it only runs after phase{1..5} + survival
# all finish.
#
# Env vars:
#   RUN_SURVIVAL=no   skip --include-survival (used by full_paper.sh --no-survival)

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
RUN_SURVIVAL="${RUN_SURVIVAL:-yes}"

cd "$PEARL_REPO"
mkdir -p logs wacv_results/paper_run

source "$PEARL_VENV/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

echo "[paper_final] node=$(hostname)"
echo "[paper_final] RUN_SURVIVAL=$RUN_SURVIVAL"

EXTRA_FLAGS=""
if [[ "$RUN_SURVIVAL" == "no" ]]; then
    EXTRA_FLAGS="--skip-survival"
fi

python scripts/reproduce_paper.py \
    --skip-smokes --skip-training $EXTRA_FLAGS \
    --output-dir wacv_results/paper_run \
    --cohorts Breast,Skin,Lymph \
    --apple-to-apple

echo "[paper_final] done"
echo "  figures : wacv_results/paper_run/paper_figures/"
echo "  tables  : wacv_results/paper_run/paper_figures/tables_1_2.json"
echo "  paper   : wacv_results/paper_run/paper.tex"

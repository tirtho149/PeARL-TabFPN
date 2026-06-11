#!/bin/bash
#SBATCH --job-name=h2h_figures
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --partition=nova
#SBATCH --output=la_regression/h2h_figures-%j.out
#SBATCH --error=la_regression/h2h_figures-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
COHORT="${COHORT:-breast}"
cd /work/mech-ai-scratch/tirtho/PeARL-TabFPN
source venv/bin/activate
echo "[figures] cohort=$COHORT"
python la_regression/make_figures.py --cohort "$COHORT"
echo "[figures] done"

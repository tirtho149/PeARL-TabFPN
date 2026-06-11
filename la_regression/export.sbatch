#!/bin/bash
#SBATCH --job-name=la_export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --partition=nova
#SBATCH --output=la_regression/la_export-%j.out
#SBATCH --error=la_regression/la_export-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
COHORT="${COHORT:-breast}"
cd /work/mech-ai-scratch/tirtho/PeARL-TabFPN
source venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "[la_export] cohort=$COHORT node=$(hostname)"
python la_regression/export_dataset.py --cohort "$COHORT"
echo "[la_export] done"

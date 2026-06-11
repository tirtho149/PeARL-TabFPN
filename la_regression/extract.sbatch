#!/bin/bash
#SBATCH --job-name=la_extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|h200"
#SBATCH --time=01:00:00
#SBATCH --partition=scavenger
#SBATCH --output=la_regression/la_extract-%j.out
#SBATCH --error=la_regression/la_extract-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
COHORT="${COHORT:-breast}"
LA_ROOT=/work/mech-ai-scratch/tirtho/LocateAnythingBench
cd $LA_ROOT
source .venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi
export PYTHONPATH="$LA_ROOT/eagle/Embodied:${PYTHONPATH:-}"
echo "[la_extract] cohort=$COHORT node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
python /work/mech-ai-scratch/tirtho/PeARL-TabFPN/la_regression/extract_embeddings.py --cohort "$COHORT"
echo "[la_extract] done"

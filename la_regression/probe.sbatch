#!/bin/bash
#SBATCH --job-name=la_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|l40s"
#SBATCH --time=00:30:00
#SBATCH --partition=scavenger
#SBATCH --output=la_regression/la_probe-%j.out
#SBATCH --error=la_regression/la_probe-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
LA_ROOT=/work/mech-ai-scratch/tirtho/LocateAnythingBench
cd $LA_ROOT
source .venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi
export PYTHONPATH="$LA_ROOT/eagle/Embodied:${PYTHONPATH:-}"
echo "[la_probe] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
python /work/mech-ai-scratch/tirtho/PeARL-TabFPN/la_regression/probe_la_embed.py
echo "[la_probe] done"

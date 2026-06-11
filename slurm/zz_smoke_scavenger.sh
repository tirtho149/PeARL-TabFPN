#!/bin/bash
#SBATCH --job-name=pearl_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s"
#SBATCH --time=00:40:00
#SBATCH --partition=scavenger
#SBATCH --output=logs/pearl_smoke-%j.out
#SBATCH --error=logs/pearl_smoke-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${PEARL_REPO:-$PWD}"
mkdir -p logs
source venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "[smoke] node=$(hostname)"
python scripts/run_reproduction.py --apple-to-apple --smoke-test --output-dir ./smoke_results
echo "[smoke] done"

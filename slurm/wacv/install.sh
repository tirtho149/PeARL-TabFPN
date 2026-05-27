#!/bin/bash
#SBATCH --job-name=pearl_install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=nova
#SBATCH --output=logs/pearl_install-%j.out
#SBATCH --error=logs/pearl_install-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Install gate — runs SETUP_ENV.sh on a GPU node so the CUDA wheel is picked
# correctly (login nodes often lack nvidia-smi). Requests 1 GPU just so the
# post-install probe (`torch.cuda.is_available`) sees a device.

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs

echo "[pearl_install] node=$(hostname)"
nvidia-smi || echo "(no GPU visible — install will fall back to CPU torch)"

# Run the cross-platform installer. --venv points at the canonical $PEARL_VENV
# so subsequent SLURM scripts find it via the same path.
bash SETUP_ENV.sh --venv "$PEARL_VENV"

echo "[pearl_install] activating venv to verify"
# shellcheck disable=SC1091
source "$PEARL_VENV/bin/activate"
python - <<'PYEOF'
import sys, torch
print(f"python   : {sys.version.split()[0]}")
print(f"torch    : {torch.__version__}")
print(f"CUDA built : {torch.version.cuda or 'CPU-only'}")
print(f"CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU      : {p.name}, {p.total_memory/(1024**3):.1f} GB")
PYEOF

echo "[pearl_install] done — next: sbatch slurm/wacv/smoke_gates.sh"

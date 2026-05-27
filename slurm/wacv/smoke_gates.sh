#!/bin/bash
#SBATCH --job-name=pearl_smokes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=nova
#SBATCH --output=logs/pearl_smokes-%j.out
#SBATCH --error=logs/pearl_smokes-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Smoke gate bundle — runs every smoke test in order on a GPU node.
# Must pass before submitting any phase{1..5} or training SLURM job.
#
# Gate ordering (matches README "Smoke gates" section):
#   1. smoke_no_data   — PCC/smoothing/ssGSEA math (numpy only)
#   2. smoke_tabpfn3   — API + Tier B runtime fit/predict on CPU
#   3. smoke_gpu       — CUDA + TabPFN v3 on GPU (canonical pre-train gate)
#   4. smoke_survival  — GDC API + clinical + C-index + WSI (downloads 24 MB)
#   5. validate        — apple-to-apple training loop on stub tensors
#   6. verify_data     — real HEST loading on all 3 cohorts (needs HEST on disk)

set -uo pipefail   # NOT -e — we want every smoke to run even if one fails,
                    # then exit non-zero at the end if any failed.
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
cd "$PEARL_REPO"
mkdir -p logs

echo "[smoke_gates] node=$(hostname)"
nvidia-smi || true

source "$PEARL_VENV/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

# Track which smokes failed so we exit non-zero at the end without short-circuiting.
declare -a FAILED=()
run_gate() {
    local name=$1; shift
    echo ""
    echo "=================================================="
    echo "GATE: $name"
    echo "=================================================="
    if ! "$@"; then
        echo "FAILED: $name"
        FAILED+=("$name")
    fi
}

run_gate "smoke_no_data"   python scripts/smoke_no_data.py
run_gate "smoke_tabpfn3"   python scripts/smoke_tabpfn3.py
run_gate "smoke_gpu"       python scripts/smoke_gpu.py
run_gate "smoke_survival"  python scripts/smoke_survival.py
run_gate "validate"        python scripts/validate.py
run_gate "verify_data"     python scripts/verify_data.py

echo ""
echo "=================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL SMOKE GATES PASSED — safe to launch training jobs."
    exit 0
else
    echo "SMOKE GATES FAILED: ${FAILED[*]}"
    echo "DO NOT submit phase{1..5} or training SLURM jobs until these pass."
    exit 1
fi

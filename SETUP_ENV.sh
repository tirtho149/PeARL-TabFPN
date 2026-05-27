#!/usr/bin/env bash
# SETUP_ENV.sh — reproducible environment install for PEaRL + TabPFN v3.
#
# Handles:
#   - Python 3.11 venv (chosen for torch+tabpfn+scanpy+lifelines compat;
#     3.10 also works, 3.13+ does not yet have wheels for everything)
#   - Correct PyTorch CUDA wheel based on detected CUDA version
#   - Project package install + optional survival extra
#   - Post-install sanity smokes
#
# Usage:
#   bash SETUP_ENV.sh                  # auto-detect CUDA, install all
#   bash SETUP_ENV.sh --cpu            # force CPU torch (Mac dev or no GPU)
#   bash SETUP_ENV.sh --cuda 12.1      # explicit CUDA version
#   bash SETUP_ENV.sh --no-survival    # skip TCGA-BRCA extras
#   bash SETUP_ENV.sh --venv .venv     # custom venv dir (default: venv)
#   bash SETUP_ENV.sh --python python3.11   # custom Python interpreter
#
# After this completes, run the smoke gates (each is fast, exits non-zero on
# failure):
#   source venv/bin/activate
#   python scripts/smoke_no_data.py        # PCC + smoothing + ssGSEA math
#   python scripts/smoke_tabpfn3.py        # TabPFN v3 API + runtime
#   python scripts/smoke_gpu.py            # CUDA + TabPFN v3 on GPU
#   python scripts/smoke_survival.py       # GDC API + clinical + C-index + WSI
#   python scripts/validate.py             # apple-to-apple training loop on stubs
#
# Only after every smoke passes should you launch a full training run.

set -euo pipefail

# ---- defaults ----
VENV_DIR="venv"
PYTHON_BIN=""
CUDA_VERSION=""    # auto-detect if blank, --cpu forces CPU
INSTALL_SURVIVAL="yes"
SKIP_SMOKES="no"

# ---- arg parse ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)            CUDA_VERSION="cpu"; shift ;;
        --cuda)           CUDA_VERSION="$2"; shift 2 ;;
        --no-survival)    INSTALL_SURVIVAL="no"; shift ;;
        --venv)           VENV_DIR="$2"; shift 2 ;;
        --python)         PYTHON_BIN="$2"; shift 2 ;;
        --skip-smokes)    SKIP_SMOKES="yes"; shift ;;
        -h|--help)        sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1"; exit 1 ;;
    esac
done

# ---- pick a Python interpreter ----
if [[ -z "$PYTHON_BIN" ]]; then
    for cand in python3.11 python3.12 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            v=$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            case "$v" in
                3.10|3.11|3.12) PYTHON_BIN="$cand"; break ;;
            esac
        fi
    done
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: no usable Python (need 3.10, 3.11, or 3.12)" >&2
    echo "       on macOS: brew install python@3.11" >&2
    echo "       on Ubuntu: apt install python3.11 python3.11-venv" >&2
    exit 1
fi

PYV=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "==> Using Python $PYV ($(command -v $PYTHON_BIN))"

# ---- detect CUDA if not specified ----
if [[ -z "$CUDA_VERSION" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # nvidia-smi reports CUDA version (e.g. "CUDA Version: 12.4")
        detected=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
        if [[ -n "$detected" ]]; then
            # Get CUDA version from nvidia-smi header line
            cuda_full=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1 | awk '{print $3}' || true)
            if [[ -n "$cuda_full" ]]; then
                # Round to nearest PyTorch-supported wheel: 11.8, 12.1, 12.4, 12.6
                case "$cuda_full" in
                    11.*) CUDA_VERSION="11.8" ;;
                    12.0|12.1|12.2|12.3) CUDA_VERSION="12.1" ;;
                    12.4|12.5) CUDA_VERSION="12.4" ;;
                    12.6|12.7|12.8|12.9) CUDA_VERSION="12.6" ;;
                    13.*) CUDA_VERSION="12.6" ;;  # PyTorch doesn't ship CUDA 13 wheels yet
                    *) CUDA_VERSION="12.4" ;;     # safe default for unknown
                esac
                echo "==> Detected GPU driver, CUDA $cuda_full → installing torch CUDA $CUDA_VERSION wheel"
            else
                CUDA_VERSION="cpu"
                echo "==> nvidia-smi present but no CUDA version reported — falling back to CPU torch"
            fi
        fi
    fi
    if [[ -z "$CUDA_VERSION" ]]; then
        CUDA_VERSION="cpu"
        echo "==> No NVIDIA GPU detected — installing CPU torch (Mac dev / no-GPU box)"
    fi
fi

# ---- create venv ----
if [[ -d "$VENV_DIR" ]]; then
    echo "==> Reusing existing venv at $VENV_DIR"
else
    echo "==> Creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --quiet --upgrade pip wheel setuptools

# ---- install torch FIRST with the right CUDA wheel ----
case "$CUDA_VERSION" in
    cpu)
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        echo "==> Installing CPU torch from $TORCH_INDEX"
        ;;
    11.8)  TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    12.1)  TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    12.4)  TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    12.6)  TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
    *) echo "ERROR: unsupported CUDA version $CUDA_VERSION" >&2; exit 1 ;;
esac
if [[ "$CUDA_VERSION" != "cpu" ]]; then
    echo "==> Installing CUDA $CUDA_VERSION torch from $TORCH_INDEX"
fi
pip install --quiet --index-url "$TORCH_INDEX" 'torch>=2.5' 'torchvision>=0.16' \
    || { echo "ERROR: torch install failed. On Intel Mac the wheel index caps at torch 2.2.2;" >&2
         echo "       TabPFN v3 won't run here — use a Linux GPU box or Apple Silicon Mac." >&2
         exit 1; }

# ---- install everything else ----
echo "==> Installing project package (editable)"
EXTRAS=""
if [[ "$INSTALL_SURVIVAL" == "yes" ]]; then
    EXTRAS="[survival]"
    echo "    (+ survival extras: openslide, lifelines, tifffile)"
fi
pip install --quiet -e ".$EXTRAS"

# ---- verify torch sees CUDA (if we asked for it) ----
echo ""
echo "==> Post-install verification"
python - <<PYEOF
import sys
import torch
print(f"  python   : {sys.version.split()[0]}")
print(f"  torch    : {torch.__version__}")
print(f"  CUDA built : {torch.version.cuda or 'CPU-only build'}")
print(f"  CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"  GPU count: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / (1024**3)
        print(f"    [{i}] {p.name} — {gb:.1f} GB, capability {p.major}.{p.minor}")
elif "$CUDA_VERSION" != "cpu":
    print()
    print("  WARNING: requested CUDA $CUDA_VERSION but torch.cuda.is_available() is False.")
    print("           Check NVIDIA driver, that you're on a GPU node (not a login node),")
    print("           and that CUDA_VISIBLE_DEVICES isn't set to ''.")
PYEOF

# Probe TabPFN v3 API
echo ""
python - <<'PYEOF'
import sys
try:
    import tabpfn
    from tabpfn.constants import ModelVersion
    from tabpfn import TabPFNRegressor
    print(f"  tabpfn   : {tabpfn.__version__}")
    print(f"  versions : {[v.name for v in ModelVersion]}")
    assert "V3" in [v.name for v in ModelVersion]
    assert hasattr(TabPFNRegressor, "create_default_for_version")
    print("  ModelVersion.V3 + create_default_for_version: OK")
except Exception as e:
    print(f"  tabpfn import failed: {type(e).__name__}: {e}")
    print("  → Expected on Intel Mac (torch < 2.5). Otherwise check torch install.")
    sys.exit(0)  # not fatal — surfaces in smoke_tabpfn3.py
PYEOF

if [[ "$INSTALL_SURVIVAL" == "yes" ]]; then
    echo ""
    python - <<'PYEOF'
try:
    import openslide, tifffile, lifelines
    print(f"  openslide: {openslide.__version__} (dylib {openslide.__library_version__})")
    print(f"  tifffile : {tifffile.__version__}")
    print(f"  lifelines: {lifelines.__version__}")
except Exception as e:
    print(f"  survival extras check failed: {type(e).__name__}: {e}")
PYEOF
fi

# ---- run smoke gates ----
echo ""
if [[ "$SKIP_SMOKES" == "yes" ]]; then
    echo "==> Skipping smokes (--skip-smokes)"
else
    echo "==> Running smoke gates"
    for smoke in scripts/smoke_no_data.py scripts/smoke_tabpfn3.py scripts/smoke_survival.py; do
        if [[ ! -f "$smoke" ]]; then continue; fi
        echo ""
        echo "----- $smoke -----"
        # smoke_survival defaults to downloading 24 MB; pass --no-wsi by default
        # so the env-install doesn't surprise the user with a network pull.
        case "$smoke" in
            *smoke_survival*)  python "$smoke" --no-wsi    || echo "(non-fatal)" ;;
            *smoke_tabpfn3*)
                if python -c "import torch; assert torch.cuda.is_available() or '$CUDA_VERSION' == 'cpu'" 2>/dev/null; then
                    python "$smoke" --skip-runtime || echo "(non-fatal)"
                else
                    python "$smoke" --skip-runtime || echo "(non-fatal)"
                fi ;;
            *)                 python "$smoke" || echo "(non-fatal)" ;;
        esac
    done
fi

echo ""
echo "============================================================"
echo "ENV READY"
echo "============================================================"
echo "  venv     : $VENV_DIR"
echo "  python   : $PYV"
echo "  torch    : $CUDA_VERSION ($([[ "$CUDA_VERSION" == "cpu" ]] && echo "CPU only" || echo "CUDA wheel"))"
echo "  survival : $INSTALL_SURVIVAL"
echo ""
echo "Activate with:  source $VENV_DIR/bin/activate"
echo ""
echo "Next gates (run all of these before any expensive job):"
echo "  python scripts/smoke_no_data.py"
echo "  python scripts/smoke_tabpfn3.py            # full runtime check on GPU"
echo "  python scripts/smoke_gpu.py                # GPU + TabPFN v3 GPU fit"
echo "  python scripts/smoke_survival.py           # downloads 24 MB sample WSI"
echo "  python scripts/validate.py                 # apple-to-apple structural"
echo "  python scripts/verify_data.py              # needs real HEST data on disk"
echo ""
echo "Then run a real training job:"
echo "  python scripts/run_reproduction.py --apple-to-apple --cohort Breast --folds 5"
echo "============================================================"

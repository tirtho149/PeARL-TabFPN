#!/bin/bash
#SBATCH --job-name=wacv_survival
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/wacv_survival-%j.out
#SBATCH --error=logs/wacv_survival-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tirtho@iastate.edu

# Survival arm — Figure/Table 3 of arXiv:2510.03455.
# 5-fold AB-MIL + Cox on TCGA-BRCA WSIs.
#
# Prereqs:
#   • WSIs downloaded to $WSI_DIR (default: $PEARL_REPO/tcga_brca_wsi)
#   • Clinical TSV at $CLINICAL_TSV (default: $PEARL_REPO/tcga_smoke/brca_survival.tsv)
#   • TCGA-BRCA is ~1.08 TB total — make sure WSI_DIR points at scratch or
#     an external volume, not the home filesystem.

set -euo pipefail
export PYTHONUNBUFFERED=1
PEARL_REPO="${PEARL_REPO:-$PWD}"
PEARL_VENV="${PEARL_VENV:-$PEARL_REPO/venv}"
WSI_DIR="${WSI_DIR:-$PEARL_REPO/tcga_brca_wsi}"
CLINICAL_TSV="${CLINICAL_TSV:-$PEARL_REPO/tcga_smoke/brca_survival.tsv}"

cd "$PEARL_REPO"
mkdir -p logs wacv_results/survival/embeddings

source "$PEARL_VENV/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

echo "[wacv_survival] node=$(hostname)"
echo "[wacv_survival] WSI_DIR=$WSI_DIR"
echo "[wacv_survival] CLINICAL_TSV=$CLINICAL_TSV"
nvidia-smi || true

# Quick pre-flight: both data sources must exist
if [ ! -d "$WSI_DIR" ]; then
    echo "ERROR: WSI_DIR $WSI_DIR not found. Download via gdc-client first." >&2
    exit 1
fi
if [ ! -f "$CLINICAL_TSV" ]; then
    echo "ERROR: clinical TSV $CLINICAL_TSV not found." >&2
    echo "  Run: python scripts/smoke_survival.py --no-wsi  (downloads clinical to tcga_smoke/)" >&2
    exit 1
fi

python scripts/train_survival.py \
    --wsi-dir "$WSI_DIR" \
    --clinical-tsv "$CLINICAL_TSV" \
    --output-dir wacv_results/survival \
    --cache-dir wacv_results/survival/embeddings \
    --encoder uni \
    --epochs 30 \
    --n-folds 5 \
    --max-tiles-per-slide 1024

echo "[wacv_survival] done"

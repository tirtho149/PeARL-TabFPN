#!/usr/bin/env bash
# full_paper.sh — submit the entire WACV paper pipeline as a chained
# dependency graph. One bash invocation; SLURM runs the rest unattended.
#
# This is the SBATCH equivalent of `python scripts/reproduce_paper.py`.
# Prefer this on HPC so each phase gets its own resource allocation and
# the long jobs don't tie up your shell.
#
# Usage:
#   PEARL_REPO=$PWD bash slurm/wacv/full_paper.sh                 # full run
#   PEARL_REPO=$PWD bash slurm/wacv/full_paper.sh --no-survival   # skip TCGA-BRCA
#   PEARL_REPO=$PWD bash slurm/wacv/full_paper.sh --dry-run       # print sbatch chain
#
# Dependency chain (each job runs after the previous one succeeds):
#   install → smoke_gates → cache_embeddings → phase0_validate →
#   phase1_accuracy → phase2_config_sweep → phase3_calibration →
#   phase4_pathway_maps → phase5_compute → [07_survival] → figures+tables
#
# Outputs:
#   wacv_results/{phase0..5}/, wacv_results/survival/,
#   wacv_results/paper_run/paper_figures/, wacv_results/paper_run/paper.tex

set -euo pipefail

PEARL_REPO="${PEARL_REPO:-$PWD}"
RUN_SURVIVAL="yes"
DRY_RUN="no"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-survival) RUN_SURVIVAL="no"; shift ;;
        --dry-run)     DRY_RUN="yes"; shift ;;
        -h|--help)     sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1"; exit 1 ;;
    esac
done

cd "$PEARL_REPO"
mkdir -p logs

# Helper — submit a job that depends on the previous one (or unconditionally).
# Captures the SLURM jobid and prints the chain as we go.
prev=""
submit() {
    local script=$1
    local desc=$2
    local dep_flag=""
    if [[ -n "$prev" ]]; then dep_flag="--dependency=afterok:$prev"; fi
    if [[ "$DRY_RUN" == "yes" ]]; then
        echo "  [dry-run] sbatch $dep_flag $script   # $desc"
        prev="DRY$RANDOM"
    else
        local jid
        jid=$(sbatch --parsable $dep_flag --export=ALL,PEARL_REPO="$PEARL_REPO" "$script")
        echo "  [$jid] $desc  (after $prev)"
        prev="$jid"
    fi
}

echo "============================================================"
echo " WACV FULL-PAPER SBATCH CHAIN"
echo "============================================================"
echo " PEARL_REPO    : $PEARL_REPO"
echo " run_survival  : $RUN_SURVIVAL"
echo " dry_run       : $DRY_RUN"
echo "============================================================"

submit slurm/wacv/install.sh             "Stage 0 — env install"
submit slurm/wacv/smoke_gates.sh         "Stage 1 — smoke gates (must pass)"
submit slurm/wacv/00_cache_embeddings.sh "Stage 2 — cache UNI embeddings"
submit slurm/wacv/01_phase0_validate.sh  "Stage 3 — Phase 0 validate"
submit slurm/wacv/02_phase1_accuracy.sh  "Stage 4 — Phase 1 accuracy (Tables 1+2)"
submit slurm/wacv/03_phase2_config_sweep.sh "Stage 5 — Phase 2 config sweep"
submit slurm/wacv/04_phase3_calibration.sh  "Stage 6 — Phase 3 calibration"
submit slurm/wacv/05_phase4_pathway_maps.sh "Stage 7 — Phase 4 pathway maps"
submit slurm/wacv/06_phase5_compute.sh   "Stage 8 — Phase 5 compute chars"
if [[ "$RUN_SURVIVAL" == "yes" ]]; then
    submit slurm/wacv/07_survival.sh     "Stage 9 — Survival (Table 3)"
fi

# Final stage: aggregate + figures + paper.tex render.
# Uses the dedicated final SLURM script we ship in-repo so dry-runs don't
# create scratch files. The script reads RUN_SURVIVAL from env to decide
# whether to pass --skip-survival.
export RUN_SURVIVAL
submit slurm/wacv/08_reproduce_paper_final.sh "Stage 10 — aggregate + figures + paper.tex"

echo ""
echo "============================================================"
echo " CHAIN SUBMITTED"
echo "============================================================"
echo " Monitor with:  squeue -u \$USER"
echo " Logs:          $PEARL_REPO/logs/"
echo " Final output:  $PEARL_REPO/wacv_results/paper_run/paper.tex"
echo "============================================================"

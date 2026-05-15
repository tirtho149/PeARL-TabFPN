#!/bin/bash
# Submit the full PeARL-TabFPN pipeline as a SLURM dependency chain.
#
# One command, hands-off afterwards: each phase queues with
# `--dependency=afterok:<prev_jobid>` so SLURM runs them sequentially as
# each predecessor succeeds. Each phase still gets its own allocation —
# the CPU-only phases (install / download / validate) do not hold a GPU.
#
# Usage (from the repo root):
#   PEARL_REPO=$PWD bash slurm/submit_all.sh                    # Breast only: 00→01→02→03→04→06 (~53 hr)
#   PEARL_REPO=$PWD bash slurm/submit_all.sh --bundled          # Breast only: 00→01→02→05→06   (~50 hr)
#   PEARL_REPO=$PWD bash slurm/submit_all.sh --cohorts Breast,Skin,Lymph   # All three PEaRL cohorts, chained
#   PEARL_REPO=$PWD bash slurm/submit_all.sh --cohorts Breast,Skin,Lymph --bundled --skip-install --skip-download
#
# Flags:
#   --bundled              Use slurm/05_train_head_to_head.sh in place of 03+04.
#   --cohorts <list>       Comma-separated cohorts to run sequentially (default: Breast).
#                          Allowed: Breast, Skin, Lymph (the three PEaRL paper cohorts).
#   --skip-install         Skip phase 00 (venv already exists, package already installed).
#   --skip-download        Skip phase 01 (hest_data/ already populated).
#   --dry-run              Print the sbatch commands without actually submitting.
#
# On success this prints the full chain of job IDs. Cancel the whole
# pipeline with `scancel <jobid> ...` for any of them — dependent jobs
# auto-cancel.

set -euo pipefail

PEARL_REPO="${PEARL_REPO:-$PWD}"
cd "$PEARL_REPO"

BUNDLED=0
SKIP_INSTALL=0
SKIP_DOWNLOAD=0
DRY_RUN=0
COHORTS="Breast"

# Parse args. `--cohorts` takes a value, so handle the two-token form.
ARGS=("$@")
i=0
while [ $i -lt ${#ARGS[@]} ]; do
  arg="${ARGS[$i]}"
  case "$arg" in
    --bundled)        BUNDLED=1 ;;
    --skip-install)   SKIP_INSTALL=1 ;;
    --skip-download)  SKIP_DOWNLOAD=1 ;;
    --dry-run)        DRY_RUN=1 ;;
    --cohorts)
      i=$((i+1))
      COHORTS="${ARGS[$i]:-}"
      if [ -z "$COHORTS" ]; then
        echo "[submit_all] --cohorts requires a comma-separated list" >&2
        exit 2
      fi
      ;;
    --cohorts=*)
      COHORTS="${arg#--cohorts=}"
      ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[submit_all] unknown flag: $arg" >&2
      echo "Run with --help for usage." >&2
      exit 2
      ;;
  esac
  i=$((i+1))
done

# Validate cohorts.
IFS=',' read -ra COHORT_ARR <<< "$COHORTS"
for c in "${COHORT_ARR[@]}"; do
  case "$c" in
    Breast|Skin|Lymph) ;;
    *)
      echo "[submit_all] unknown cohort '$c'; allowed: Breast, Skin, Lymph" >&2
      exit 2
      ;;
  esac
done

# submit_phase <script-path> [<dep-jobid>] [<cohort>]
# Echoes the new job ID on stdout; logs the submission to stderr.
submit_phase() {
  local script="$1"
  local dep="${2:-}"
  local cohort="${3:-}"
  local cmd=(sbatch --parsable)
  if [ -n "$dep" ]; then
    cmd+=("--dependency=afterok:${dep}")
  fi
  if [ -n "$cohort" ]; then
    cmd+=("--export=ALL,PEARL_COHORT=${cohort},PEARL_REPO=${PEARL_REPO}")
  fi
  cmd+=("$script")
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] ${cmd[*]}" >&2
    local tag="DRYRUN_$(basename "$script" .sh)"
    [ -n "$cohort" ] && tag="${tag}_${cohort}"
    echo "$tag"
    return
  fi
  local jobid
  jobid="$("${cmd[@]}")"
  if [[ ! "$jobid" =~ ^[0-9]+$ ]]; then
    echo "[submit_all] FAILED: unexpected sbatch output for $script: $jobid" >&2
    exit 1
  fi
  printf '[submit_all] queued %-40s  job=%s%s%s\n' \
    "$script" "$jobid" \
    "${cohort:+  [cohort=$cohort]}" \
    "${dep:+  (after $dep)}" >&2
  echo "$jobid"
}

mkdir -p logs

echo "[submit_all] repo=$PEARL_REPO  cohorts=$COHORTS  bundled=$BUNDLED  skip_install=$SKIP_INSTALL  skip_download=$SKIP_DOWNLOAD  dry_run=$DRY_RUN" >&2

PREV=""
declare -a CHAIN_LABELS CHAIN_IDS

queue() {
  local label="$1" script="$2" cohort="${3:-}"
  local id
  id="$(submit_phase "$script" "$PREV" "$cohort")"
  PREV="$id"
  CHAIN_LABELS+=("$label")
  CHAIN_IDS+=("$id")
}

# One-time CPU-only phases (no cohort).
if [ "$SKIP_INSTALL" -eq 0 ]; then
  queue "00 install"  "slurm/00_install.sh"
fi
if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
  queue "01 download" "slurm/01_download_data.sh"
fi

# Per-cohort: validate (cheap, cohort-aware logging) → train → figures.
for cohort in "${COHORT_ARR[@]}"; do
  queue "02 validate ($cohort)" "slurm/02_validate.sh" "$cohort"
  if [ "$BUNDLED" -eq 1 ]; then
    queue "05 head_to_head ($cohort)" "slurm/05_train_head_to_head.sh" "$cohort"
  else
    queue "03 baseline ($cohort)" "slurm/03_train_baseline.sh" "$cohort"
    queue "04 tabpfn ($cohort)"   "slurm/04_train_tabpfn.sh"   "$cohort"
  fi
  queue "06 figures ($cohort)"    "slurm/06_generate_figures.sh" "$cohort"
done

echo ""
echo "[submit_all] dependency chain queued (${#CHAIN_IDS[@]} jobs):"
for i in "${!CHAIN_LABELS[@]}"; do
  printf '  %-32s  job=%s\n' "${CHAIN_LABELS[$i]}" "${CHAIN_IDS[$i]}"
done
echo ""
LAST_IDX=$(( ${#CHAIN_IDS[@]} - 1 ))
echo "[submit_all] monitor:  squeue -u \$USER"
echo "[submit_all] cancel:   scancel ${CHAIN_IDS[*]}"

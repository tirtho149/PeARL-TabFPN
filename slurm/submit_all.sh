#!/bin/bash
# Submit the full PeARL-TabFPN pipeline as a SLURM dependency chain.
#
# One command, hands-off afterwards: each phase queues with
# `--dependency=afterok:<prev_jobid>` so SLURM runs them sequentially as
# each predecessor succeeds. Each phase still gets its own allocation —
# the CPU-only phases (install / download / validate) do not hold a GPU.
#
# Usage (from the repo root):
#   PEARL_REPO=$PWD bash slurm/submit_all.sh                 # split: 00→01→02→03→04→06 (~53 hr)
#   PEARL_REPO=$PWD bash slurm/submit_all.sh --bundled       # bundled: 00→01→02→05→06   (~50 hr, one long GPU alloc)
#   PEARL_REPO=$PWD bash slurm/submit_all.sh --skip-install
#   PEARL_REPO=$PWD bash slurm/submit_all.sh --skip-download
#   PEARL_REPO=$PWD bash slurm/submit_all.sh --skip-install --skip-download --bundled
#
# Flags:
#   --bundled         Use slurm/05_train_head_to_head.sh in place of 03+04.
#   --skip-install    Skip phase 00 (venv already exists, package already installed).
#   --skip-download   Skip phase 01 (hest_data/ already populated).
#   --dry-run         Print the sbatch commands without actually submitting.
#
# On success this prints the full chain of job IDs. Cancel the whole
# pipeline with `scancel <jobid>` for any one of them — dependent jobs
# auto-cancel.

set -euo pipefail

PEARL_REPO="${PEARL_REPO:-$PWD}"
cd "$PEARL_REPO"

BUNDLED=0
SKIP_INSTALL=0
SKIP_DOWNLOAD=0
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --bundled)        BUNDLED=1 ;;
    --skip-install)   SKIP_INSTALL=1 ;;
    --skip-download)  SKIP_DOWNLOAD=1 ;;
    --dry-run)        DRY_RUN=1 ;;
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
done

# submit_phase <script-path> [<dep-jobid>]
# Echoes the new job ID on stdout; logs the submission to stderr.
submit_phase() {
  local script="$1"
  local dep="${2:-}"
  local cmd=(sbatch --parsable)
  if [ -n "$dep" ]; then
    cmd+=("--dependency=afterok:${dep}")
  fi
  cmd+=("$script")
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] ${cmd[*]}" >&2
    echo "DRYRUN_$(basename "$script" .sh)"
    return
  fi
  local jobid
  jobid="$("${cmd[@]}")"
  if [[ ! "$jobid" =~ ^[0-9]+$ ]]; then
    echo "[submit_all] FAILED: unexpected sbatch output for $script: $jobid" >&2
    exit 1
  fi
  printf '[submit_all] queued %-40s  job=%s%s\n' \
    "$script" "$jobid" "${dep:+  (after $dep)}" >&2
  echo "$jobid"
}

# Make sure the chain has somewhere to write logs.
mkdir -p logs

echo "[submit_all] repo=$PEARL_REPO  bundled=$BUNDLED  skip_install=$SKIP_INSTALL  skip_download=$SKIP_DOWNLOAD  dry_run=$DRY_RUN" >&2

PREV=""
declare -a CHAIN_LABELS CHAIN_IDS

queue() {
  local label="$1" script="$2"
  local id
  id="$(submit_phase "$script" "$PREV")"
  PREV="$id"
  CHAIN_LABELS+=("$label")
  CHAIN_IDS+=("$id")
}

if [ "$SKIP_INSTALL" -eq 0 ]; then
  queue "00 install"   "slurm/00_install.sh"
fi
if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
  queue "01 download"  "slurm/01_download_data.sh"
fi
queue "02 validate"    "slurm/02_validate.sh"

if [ "$BUNDLED" -eq 1 ]; then
  queue "05 head_to_head" "slurm/05_train_head_to_head.sh"
else
  queue "03 baseline"  "slurm/03_train_baseline.sh"
  queue "04 tabpfn"    "slurm/04_train_tabpfn.sh"
fi
queue "06 figures"     "slurm/06_generate_figures.sh"

echo ""
echo "[submit_all] dependency chain queued:"
for i in "${!CHAIN_LABELS[@]}"; do
  printf '  %-20s  job=%s\n' "${CHAIN_LABELS[$i]}" "${CHAIN_IDS[$i]}"
done
echo ""
LAST_IDX=$(( ${#CHAIN_IDS[@]} - 1 ))
echo "[submit_all] monitor with:  squeue -u \$USER  |  tail -F logs/pearl_*-${CHAIN_IDS[$LAST_IDX]}.out"
echo "[submit_all] cancel the whole pipeline:  scancel ${CHAIN_IDS[*]}"

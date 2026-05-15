# Reproducibility checklist

End-to-end recipe for reproducing the BIBM 2026 head-to-head numbers on
Iowa State's Nova HPC cluster. Every phase has a `slurm/*.sh` script.

## 0. One-time setup on the login node

```bash
git clone https://github.com/tirtho149/PeARL-TabFPN.git
cd PeARL-TabFPN
# Put your HuggingFace token in .env (gitignored).
# Required for HEST-1k (gated) and UNI (gated):
echo "HF_TOKEN=hf_xxx..." > .env
echo "HUGGINGFACE_HUB_TOKEN=hf_xxx..." >> .env
```

Then accept gating terms at:
- https://huggingface.co/datasets/MahmoodLab/hest
- https://huggingface.co/MahmoodLab/UNI

## 1. Install the venv (Phase 00, ~5 min)

```bash
sbatch slurm/00_install.sh
```

Creates `./venv/`, installs deps, `pip install -e .` so `pearl_tabpfn` is
importable as a package. Writes to `logs/pearl_install-<jobid>.out`.

## 2. Download HEST-1k Breast cohort (Phase 01, ~30 min)

```bash
sbatch slurm/01_download_data.sh
```

Downloads only what `pearl_tabpfn.data` reads:
`hest_data/st/{id}.h5ad` + `hest_data/patches/{id}.h5` for all Breast
sections, plus the metadata CSV. ~45 GB. Skip if data is already there.

## 3. Structural validation (Phase 02, ~1 min)

```bash
sbatch slurm/02_validate.sh
```

Runs `scripts/validate.py`. Exercises every code path with stub data — NO
PCC numbers produced, only a pass/fail on whether the pipeline is wired
correctly. Must pass before launching the expensive training jobs.

## 4. Run the head-to-head (Phase 03/04 or 05)

**Option A — split into two SLURM jobs** (recommended; lets the cheaper
baseline finish first and frees the GPU between phases):

```bash
sbatch slurm/03_train_baseline.sh    # ~7 hours
sbatch slurm/04_train_tabpfn.sh      # ~45 hours
```

**Option B — single bundled job:**

```bash
sbatch slurm/05_train_head_to_head.sh   # ~50 hours
```

Each fold writes `results/<run-name>/predictions/fold_{i}.npz` and the
incremental `fold_results.json` as it completes. The final
`reproduction_results.json` (summary mean ± std + per-fold metrics + the
paper's reference numbers in `paper_breast_baseline`) is written when all
5 folds finish.

## 5. Generate figures (Phase 06, ~2 min)

```bash
sbatch slurm/06_generate_figures.sh
```

Loads fold-0's saved predictions and produces seven head-to-head PNGs in
`results/<run-name>/figures/`.

## 6. Fill the BIBM paper

See `paper/README.md` for the placeholder-filling recipe. The 16 `\TBD`
cells in `paper/BIBM2026_PEaRL_TabPFN.tex` come from the `summary` block
of `reproduction_results.json`.

## Reproducibility guarantees

- **Same code paths for both heads.** PEaRL+MLP and PEaRL+TabPFN share
  `pearl_tabpfn.encoders` (PathwayEncoder, VisionEncoder, ContrastiveLoss)
  bit-for-bit; only the head module differs.
- **Apple-to-apple settings bundled.** `--apple-to-apple` sets every
  paper-faithful knob simultaneously (see `docs/APPLE_TO_APPLE.md` for the
  full table — 11 settings).
- **Deterministic section selection** (`np.random.default_rng(42)`).
- **HEST pre-flight check** (`pearl_tabpfn.reproduction.verify_hest_data`)
  fails fast if any selected section's h5ad or patches is missing on
  disk, so a 50-hour run can't silently report PCC on fewer sections than
  requested.
- **No synthetic-data fallbacks.** All silent `np.random.randn` data
  paths that previously fed the PCC pipeline have been removed. Any PCC
  number in the output is from real HEST-1k spots through real UNI
  features.

## What can still differ from the paper

Documented at the bottom of `docs/APPLE_TO_APPLE.md`. Summary: exact
section identities (paper doesn't enumerate which 36 Breast sections),
random seed details, and TCGA-BRCA survival (we don't have those WSIs).

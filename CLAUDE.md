# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research codebase reproducing **PEaRL** (Pathway-Enhanced Representation Learning, arXiv:2510.03455) on the HEST-1k spatial transcriptomics dataset, plus a follow-up variant that replaces the MLP prediction head with **TabPFN** (a pretrained in-context tabular regressor). The two heads share the encoder stack bit-for-bit and are compared apple-to-apple. Results feed an IEEE BIBM 2026 paper draft in `paper/`.

## Layout

```
PeARL-TabFPN/
├── src/pearl_tabpfn/          # importable package (`pip install -e .`)
│   ├── config.py              # global Config dataclass (`cfg`)
│   ├── data.py                # HEST-1k loading, ssGSEA, HESTDataset
│   ├── encoders.py            # PathwayEncoder, VisionEncoder, ContrastiveLoss, VitFeatureExtractor
│   ├── baseline.py            # PEaRL + MLP head (paper-faithful)
│   ├── tabpfn_head.py         # PEaRLWithTabPFN, TabPFNHead (3 modes)
│   ├── trainer.py             # Trainer (single-section, legacy — not used by current scripts)
│   ├── eval.py                # compute_metrics, plots
│   ├── figures.py             # BIBM head-to-head figures
│   ├── reproduction.py        # 5-fold CV runner; canonical entry-point body
│   └── __init__.py
├── scripts/                   # thin CLI wrappers
│   ├── run_reproduction.py    # canonical entry — both heads, --apple-to-apple
│   ├── train_baseline.py      # injects --head-mode mlp
│   ├── train_tabpfn.py        # injects --head-mode tabpfn
│   ├── validate.py            # structural smoke test on stub data
│   ├── verify_data.py         # checks HEST loads + real PCC > 0
│   └── generate_figures.py    # post-run figure renderer
├── slurm/                     # Nova SBATCH scripts (00..06)
├── docs/                      # APPLE_TO_APPLE.md, REPRODUCIBILITY.md
├── paper/                     # IEEE BIBM 2026 LaTeX source
├── reference/                 # PEaRL + HEST-1k PDFs
├── pyproject.toml             # package metadata + deps
├── requirements.txt           # mirror of pyproject deps
├── SETUP_DATA.sh              # HEST-1k download from HF
└── CLAUDE.md / README.md
```

## Common commands

### Local install + smoke

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
bash SETUP_DATA.sh                    # downloads HEST-1k (~3.9GB), gated
python scripts/validate.py            # ~1 min, structural pass/fail
python scripts/verify_data.py         # CPU-only real-PCC sanity check
```

### Apple-to-apple reproduction (canonical for BIBM 2026)

```bash
# Bundled job — both heads, 5-fold CV on Breast (~50 hr on a 24 GB GPU)
python scripts/run_reproduction.py --apple-to-apple --n-sections 36 --folds 5

# Split: baseline first (~7 hr), then TabPFN (~45 hr)
python scripts/train_baseline.py --apple-to-apple --n-sections 36 --folds 5
python scripts/train_tabpfn.py   --apple-to-apple --n-sections 36 --folds 5

# Iteration knobs (not apple-to-apple — do not use for BIBM numbers)
python scripts/run_reproduction.py --smoke-test           # 5 sections, 2 folds, 5 epochs (~10 min)
python scripts/run_reproduction.py --split spot           # KFold by spot (paper convention; easier)
python scripts/run_reproduction.py --split section        # GroupKFold by section (no leakage; harder)
python scripts/run_reproduction.py --encoder vit          # ImageNet ViT-L/16 if UNI access is blocked
```

### Nova HPC

Submit phases in order; see `docs/REPRODUCIBILITY.md` for the full recipe.

```bash
PEARL_REPO=$PWD sbatch slurm/00_install.sh         # ~5 min
PEARL_REPO=$PWD sbatch slurm/01_download_data.sh   # ~30 min
PEARL_REPO=$PWD sbatch slurm/02_validate.sh        # ~1 min
PEARL_REPO=$PWD sbatch slurm/03_train_baseline.sh  # ~7 hr
PEARL_REPO=$PWD sbatch slurm/04_train_tabpfn.sh    # ~45 hr
# OR a single bundled job:
PEARL_REPO=$PWD sbatch slurm/05_train_head_to_head.sh   # ~50 hr
PEARL_REPO=$PWD sbatch slurm/06_generate_figures.sh     # ~2 min
```

There is no test suite, linter, or CI. `scripts/validate.py` is the canonical structural smoke; `--smoke-test` on `run_reproduction.py` is the canonical end-to-end smoke.

## Architecture

### One canonical entry point, two thin wrappers

| Script | Role | Notes |
|---|---|---|
| `scripts/run_reproduction.py` | Both heads in one job | Calls `pearl_tabpfn.reproduction.main()` |
| `scripts/train_baseline.py` | MLP head only | Injects `--head-mode mlp` then calls `main()` |
| `scripts/train_tabpfn.py` | TabPFN head only | Injects `--head-mode tabpfn` then calls `main()` |

All three drive the same 5-fold cross-validation runner in `pearl_tabpfn.reproduction`. The split exists so SLURM can run the cheap baseline first and free the GPU before the long TabPFN job.

`pearl_tabpfn.trainer.Trainer` is still present but no current script uses it — it's a single-section two-stage trainer kept for reference.

### Two-stage training

1. **Stage 1 — Contrastive pretraining**: aligns image and pathway embeddings via symmetric NT-Xent loss (`encoders.ContrastiveLoss`). All trainable parameters update.
2. **Stage 2 — Supervised fine-tuning**: freezes the pathway encoder (and the vision encoder, in cached mode) and trains only the prediction head(s) on MSE for gene + pathway regression.

### Feature caching vs full-backbone path (`reproduction.py`)

Two code paths, selected automatically based on whether the backbone is partially unfrozen:

**Path A — cached features (`run_one_fold_cached`).** UNI is fully frozen. The script extracts `(N, 1024)` features once per fold and reuses them. `PEaRLCached` projects cached features through `feat_proj: Linear(1024, embed_dim)`. ~10 min/fold for MLP, ~9 hr/fold for TabPFN-pure.

**Path B — full backbone (`run_one_fold_full_backbone`).** Enabled by `--unfreeze-last-4-blocks` or `--apple-to-apple`. The last 4 transformer blocks of UNI are trainable during stage 1, so cached features go stale; we instead run the full UNI forward at every step. Uses `PEaRL` / `PEaRLWithTabPFN` directly with patch tensors. ~1 hr/fold MLP, ~10 hr/fold TabPFN-pure.

Both paths share `compute_metrics`, `aggregate_folds`, and `print_summary`. Path B is more faithful to arXiv:2510.03455 but ~4× slower; pick based on whether parity with the paper or iteration speed matters more.

If you change anything that affects the vision branch (ImageNet vs UNI normalization, freeze policy, unfreeze count), update `encoders.VisionEncoder._unfreeze_last_n_blocks`, `encoders.VisionEncoder.extract_features`, and `reproduction.PEaRLCached.forward_vision` together — they must agree on what frozen prefix the cached features represent.

### Encoders (`pearl_tabpfn.encoders`)

- **`PathwayEncoder`** — projects pathway scores `(B, n_pathways)` + 2D spatial coords through a Transformer encoder to a shared `embed_dim` space.
- **`VisionEncoder`** — supports two backbones via the `backbone=` arg:
  - `"uni"`: `MahmoodLab/UNI` via timm `hf-hub:` (DINOv2 ViT-L/16 trained on 100k WSIs). Gated on HuggingFace — requires login. **Paper-faithful path.**
  - `"vit_l_16"` (default): timm `vit_large_patch16_224`, falls back to `torchvision.models.vit_l_16` wrapped in `VitFeatureExtractor`.
  - Both emit 1024-d CLS embeddings, projected through `proj_head: Linear(1024, embed_dim)`. Backbone is frozen by default; with `freeze_backbone=False`, the last 4 transformer blocks are unfrozen. `backbone_features()` / `head_from_features()` are split out for the feature-caching path.

### Baseline head (`pearl_tabpfn.baseline`)

- **`PEaRL`** — combines both encoders; `forward_contrastive` returns L2-normalized embeddings (stage 1), `forward_supervised` runs vision encoder → MLP heads (stage 2). The supervised path **only uses the vision encoder** — pathway encoder is unused at inference.
- **`SupervisedLoss`** — MSE on gene + pathway outputs.

### TabPFN head (`pearl_tabpfn.tabpfn_head`)

`TabPFNHead` is a hybrid head with three operating modes:

| `mode` | Behavior | Apple-to-apple? |
|---|---|---|
| `refinement` | MLP trained; TabPFN replaces MLP prediction on the top-k highest-MLP-residual-variance dims. | No — partial replacement only. |
| `residual` | MLP trained; TabPFN fits MLP residuals on top-k highest-residual-variance dims; α-shrinkage blend with closed-form LS on a 10% holdout, clipped to [0,1]. | No — partial. Guarantees never-worse-than-MLP on the holdout. |
| **`pure`** | TabPFN fits **every** output dim (no top-k); MLP is not used at inference. One `TabPFNRegressor` per output dim, all 1,775 dims for Breast. | **Yes** — this is the canonical apple-to-apple head-to-head. |

Important quirks:
- TabPFN is a 1-D in-context regressor, so multi-output means a list of regressors, one per dim. `mode="pure"` materializes the full bank.
- `forward()` always returns the MLP output even in `pure` mode; `apply_tabpfn(...)` overwrites it at eval. Keeping the MLP in the module makes loading checkpoints/state-dicts uniform.
- TabPFN is fitted via `fit_tabpfn_heads(X_train, y_pathway_train, y_gene_train)` after stage 2 ends.
- If `tabpfn` is not installed, `use_tabpfn` flips to False at construction and the head is MLP-only — check logs for `"TabPFN not available, MLP-only path will be used"` if a TabPFN run looks identical to baseline.

`PEaRLWithTabPFN` reuses `PathwayEncoder` and `VisionEncoder` verbatim (imported from `pearl_tabpfn.encoders`).

### Data flow (`pearl_tabpfn.data`)

- `load_hest_sample(hest_dir, sample_id, ...)` — single section. Reads HEST-1k (h5ad expression + image patches), aligns barcodes, computes ssGSEA pathway scores via `ssgsea()`. Returns `(patches, genes, pathways, coords)` numpy arrays.
- `load_hest_multi_sample(hest_dir, sample_ids, ...)` — used by `reproduction.py`. Concatenates sections and **aligns pathway columns across sections by pooled variance** (top-N by pooled std after concatenation). Without this, top-N-by-variance picks different pathways per section, breaking cross-section training. Returns `section_ids` for `GroupKFold`. Set `normalization="paper"` for paper-style per-gene min-max + z-normed pathways.
- `HESTDataset` yields a dict with keys `patch`, `gene`, `pathway`, `coord` — every model expects this exact shape.

### Evaluation (`pearl_tabpfn.eval`)

`compute_metrics(predictions, targets, drop_constant_cols=True)` is the canonical metric function. Returns `{PCC, MSE, MAE, n_cols_used, n_cols_dropped}`. **`drop_constant_cols=True` is critical**: without it, all-zero target columns pair with small predictions to trivially "correlate" via the global flatten, inflating reported PCC. Always check `n_cols_dropped` when comparing runs.

### Configuration (`pearl_tabpfn.config`)

Single global `cfg = Config()` consumed everywhere. Many fields are overridable via env vars: `HEST_DATA_ROOT`, `HEST_MAX_SPOTS`, `PEARL_BATCH_SIZE`, `PEARL_NO_AMP`, `PEARL_SCRATCH_ENCODERS`, `HEST_ID_BREAST/SKIN/LYMPH`. `cfg.HEST_IDS` maps dataset names → HEST IDs (`Breast → TENX99`, `Skin → TENX158`, `Lymph → TENX143`). `cfg.DATASET_PATHWAYS` records the per-dataset pathway-count target the paper uses (Breast: 775, Skin: 609, Lymph: 1100).

### Figures (`pearl_tabpfn.figures`)

`generate_head_to_head_figures(...)` produces seven PNGs for the BIBM paper. `figure5_survival_cindex` is included for completeness but is an orphan helper — survival is out-of-scope in this repo (TCGA-BRCA WSIs aren't shipped); pass an empty dict to skip it.

### Outputs

`reproduction_results/`:
- `fold_results.json` (incremental, written after each fold)
- `reproduction_results.json` (final: per-fold + aggregated mean ± std + the paper's reference numbers in `paper_breast_baseline`)
- `predictions/fold_{i}.npz` per fold
- `figures/` after Phase 06

## Conventions and gotchas

- **Two parallel implementations of the training loop**: `reproduction.run_one_fold_cached` (cached features) and `reproduction.run_one_fold_full_backbone` (apple-to-apple). Encoder + loss changes propagate (both import from `pearl_tabpfn.encoders`/`baseline`/`tabpfn_head`), but freeze policies, optimizers, and early-stopping live independently in each. **When changing training behavior, sweep both.** A third `Trainer` class in `pearl_tabpfn.trainer` is legacy and not invoked by any current script.
- **Apple-to-apple is a single flag**: `--apple-to-apple` bundles 8-neighbor smoothing, <1000-spots gene filter, Scanpy-flavor HVG, Reactome+MSigDB pathways, raw pathway scale, last-4-blocks UNI unfreeze, learnable NT-Xent τ, paper gene normalization, section-stratified split, keep-constant-cols, and TabPFN-pure head replacement. Use it for any number that goes into the BIBM 2026 paper. This is the only setting where PEaRL+MLP can be directly compared to the paper's 0.5868 gene PCC / 0.5055 pathway PCC.
- **Stage 2 freezes encoders** — if you add a new trainable component, decide whether it belongs to the encoder (frozen in stage 2) or the head (trainable). In `reproduction.py` only `pathway_encoder` is frozen in stage 2 of the cached path because the vision branch is already represented by precomputed features.
- **UNI requires HuggingFace authentication** — `huggingface-cli login` and accept the gated repo terms. If unavailable, pass `--encoder vit` to use ImageNet ViT-L/16 (will not match paper numbers).
- **Authoritative dataset** is HEST-1k from HuggingFace (`MahmoodLab/hest` per `cfg.HF_HEST_REPO`). Never commit `hest_data/` (gitignored, ~3.9GB).
- **Constant-column filtering** — always pass `drop_constant_cols=True` to `compute_metrics` (the default). PCC numbers without this filter are not comparable to paper numbers.
- AMP is on by default; disable with `PEARL_NO_AMP=1` if you hit numerical issues on a particular GPU.
- **Survival is out-of-scope.** The PEaRL paper reports a TCGA-BRCA C-index of 0.659, but that requires whole-slide images this repo doesn't ship. The BIBM paper draft excludes survival.
- **Nova workflow** lives in `docs/REPRODUCIBILITY.md`; the SBATCH scripts are in `slurm/`. They read `PEARL_REPO` / `PEARL_VENV` so submission is portable.

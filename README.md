# PEaRL with TabPFN: Gene Expression Prediction from Histology

**Submission to IEEE BIBM 2026** — head-to-head comparison of MLP vs TabPFN heads on top of the PEaRL framework (arXiv:2510.03455), under an apple-to-apple reproduction protocol on HEST-1k Breast.

## Authors

**Ushashi Bhattacharjee**¹, **Alloy Das**¹, **Saria Hannan**¹, **Tirtho Roy**¹, and **Soumik Sarkar**¹

¹Iowa State University, Ames, IA

**Special Thanks**: Koushik Howlader for valuable discussions and feedback.

## What this repo is

A controlled head-to-head benchmark of **PEaRL+MLP** vs **PEaRL+TabPFN** on the HEST-1k Breast cancer cohort. The only thing that changes between the two conditions is the stage-2 prediction head; every other knob — UNI v1 backbone with last-4-blocks fine-tuned, 8-neighbor smoothing, Reactome+MSigDB Hallmark pathway pool, raw pathway-target scaling, 5-fold CV — is fixed at the values reported in arXiv:2510.03455. One CLI flag (`--apple-to-apple`) bundles every paper-faithful setting.

Outputs feed the IEEE BIBM 2026 paper draft in `paper/BIBM2026_PEaRL_TabPFN.tex`.

## Dataset

PEaRL evaluates on three HEST-1k cohorts. The pipeline supports each via the `--cohort` flag; per-cohort pathway counts auto-default to the paper's value.

| Cohort | Organ filter | Available sections (HEST v1.1.0) | Pathways ($P$) | Representative section |
|---|---|---|---|---|
| **Breast** | `organ == "Breast"` | 117 (cap 36) | 775 | `TENX99` (IDC) |
| **Skin** | `organ == "Skin"` | 80 (cap 36) | 609 | `TENX158` |
| **Lymph** | `organ == "Lymphoid"` | 5 (cap 5) | 1,100 | `TENX143` |

Other invariants across cohorts:

| Item | Value |
|---|---|
| Dataset | **HEST-1k** (`MahmoodLab/hest` on HuggingFace, gated) |
| Per-section input | `hest_data/st/{id}.h5ad` (expression) + `hest_data/patches/{id}.h5` (224×224 H&E patches) |
| Spots per section (cap) | 400 default, `--max-spots-per-section` |
| Genes | top-1000 HVGs by pooled variance, Scanpy `flavor="seurat"` |
| Pathway sources | Reactome + MSigDB Hallmark, ssGSEA-scored per spot |
| Total payload to download | **~45 GB** for the Breast cohort; Skin and Lymph add to this total when pulled |
| Foundation backbone | **UNI v1** (`MahmoodLab/UNI`, gated DINOv2 ViT-L/16 on 100k WSIs) |
| Spatial coords | 2-D, included with every spot |

The metadata CSV ships with the dataset; the actual `.h5ad` and `.h5` files are pulled by `slurm/01_download_data.sh`. The repo currently has 1 sample (`TENX99.h5ad`, 30 MB) checked in for sanity-test purposes only; the full per-cohort payload arrives during Phase 01.

## End-to-end on Nova (Iowa State HPC)

The seven SBATCH scripts under `slurm/` automate every phase. Submit them in order; each is idempotent (re-running just refreshes its phase). All scripts read `PEARL_REPO` (default `$PWD`) and `PEARL_VENV` (default `$PEARL_REPO/venv`) so the same scripts work from any allocation.

### 0. One-time login-node setup

```bash
git clone https://github.com/tirtho149/PeARL-TabFPN.git
cd PeARL-TabFPN

# HuggingFace token (gated dataset + gated UNI backbone — both required)
echo "HF_TOKEN=hf_xxx..." >  .env
echo "HUGGINGFACE_HUB_TOKEN=hf_xxx..." >> .env
```

Then in a browser, accept the gating terms at:
- https://huggingface.co/datasets/MahmoodLab/hest
- https://huggingface.co/MahmoodLab/UNI

### 1. Submit everything as a dependency chain (one command)

```bash
# Default: Breast only — all phases chained with --dependency=afterok.
PEARL_REPO=$PWD bash slurm/submit_all.sh

# All three PEaRL cohorts, end-to-end:
PEARL_REPO=$PWD bash slurm/submit_all.sh --cohorts Breast,Skin,Lymph
```

Each phase gets its own allocation — CPU-only phases (00/01/02) do not hold a GPU. With `--cohorts`, the 00/01 phases run once at the top and the per-cohort 02→03→04→06 chain runs serially for each cohort. Each cohort's outputs land under `reproduction_results/<cohort>/`.

Useful flags:

```bash
bash slurm/submit_all.sh --bundled            # use slurm/05_train_head_to_head.sh in place of 03+04
bash slurm/submit_all.sh --skip-install       # skip 00 if venv already exists
bash slurm/submit_all.sh --skip-download      # skip 01 if hest_data/ is already populated
bash slurm/submit_all.sh --dry-run            # print sbatch commands without submitting
```

The script prints the full chain of job IDs at the end; cancel the whole pipeline with `scancel <jobid> <jobid> ...` (dependents auto-cancel).

### 2. Or submit phases individually

```bash
PEARL_REPO=$PWD sbatch slurm/00_install.sh         # CPU, ~5 min
PEARL_REPO=$PWD sbatch slurm/01_download_data.sh   # CPU, ~30 min, ~45 GB
PEARL_REPO=$PWD sbatch slurm/02_validate.sh        # CPU, ~1 min
PEARL_REPO=$PWD sbatch slurm/03_train_baseline.sh  # GPU, ~7 hr
PEARL_REPO=$PWD sbatch slurm/04_train_tabpfn.sh    # GPU, ~45 hr
PEARL_REPO=$PWD sbatch slurm/06_generate_figures.sh
```

`slurm/05_train_head_to_head.sh` is an alternative to 03+04 — runs both heads in a single ~50 hr bundled job. Use it when your allocation supports one long reservation; use 03+04 when you want the cheap baseline to finish first and free the GPU between phases.

Every job emails `tirtho@iastate.edu` on BEGIN / END / FAIL (edit the `--mail-user` line in `slurm/*.sh` for a different recipient). Logs land in `logs/pearl_<job>-<jobid>.{out,err}`.

### 3. Total wall-clock budget

Recommended GPU: **24 GB NVIDIA A100 / RTX 3090 / TITAN RTX**. 16 GB cards work with `--batch-size 64`.

**Per cohort (one-shot training, 5-fold CV):**

| Phase | Wall time on 24 GB GPU | Bottleneck |
|---|---|---|
| 00 install | ~5 min | pip wheels (once) |
| 01 download | ~30 min | HF mirror network (once) |
| 02 validate | ~1 min | CPU stub training |
| 03 baseline (MLP) | ~7 hr (Breast/Skin), ~1 hr (Lymph) | 5 folds × ~1 hr (last-4-blocks unfrozen) |
| 04 TabPFN-pure | ~45 hr (Breast/Skin), ~7 hr (Lymph) | TabPFNRegressor bank per fold (775 / 609 / 1,100 dims) |
| 06 figures | ~2 min | matplotlib |

**All three PEaRL cohorts chained via `--cohorts Breast,Skin,Lymph`:**

| Cohort | 02–04–06 wall time |
|---|---|
| Breast | ~53 hr |
| Skin | ~50 hr |
| Lymph | ~8 hr |
| **All three** | **~112 hr** (~4.7 days back-to-back) |

If you start Phase 00 on a Monday morning and queue all jobs as a dependency chain, Breast figures land Thursday morning and the full three-cohort run finishes the following Saturday. On smaller GPUs (16 GB) add ~30% to Phases 03–05.

### 4. Outputs

Each cohort writes its own namespaced subdirectory:

```
reproduction_results/
├── breast/
│   ├── fold_results.json              # incremental — written after each fold
│   ├── reproduction_results.json      # final: per-fold metrics + 5-fold mean ± std + paper reference
│   ├── predictions/
│   │   ├── fold_0.npz                 # coords, pathway/gene preds + truth (both heads)
│   │   └── ...
│   └── figures/                       # written by Phase 06
│       ├── fig_h2h_1_metric_bars.png
│       └── ...
├── skin/   (same structure)
└── lymph/  (same structure)
```

The `summary` block in each `reproduction_results.json` fills the matching cohort block in `paper/BIBM2026_PEaRL_TabPFN.tex` Table 1.

## Local install (laptop / workstation, optional)

For development, smoke tests, or running on a single workstation with a 24 GB GPU:

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
bash SETUP_DATA.sh                  # ~45 GB HEST-1k pull
python scripts/validate.py          # ~1 min, structural pass/fail
python scripts/run_reproduction.py --apple-to-apple --n-sections 36 --folds 5
python scripts/generate_figures.py --results-dir reproduction_results
```

`pip install -e .` resolves every dependency from `pyproject.toml` (torch, torchvision, timm, scanpy, tabpfn, …). Tested on Python 3.11, PyTorch 2.5.1+cu121, timm 1.0.26, tabpfn 7.1.1.

For a fast sniff test (5 sections, 2 folds, 5 epochs, ~10 minutes — **does NOT match paper**):

```bash
python scripts/run_reproduction.py --smoke-test
```

## Headline result

**5-fold cross-validated, HEST-1k Breast cancer (36 sections), apple-to-apple protocol.** Numbers populated after the reproduction run completes; see `reproduction_results.json`.

| Target | **PEaRL+MLP (ours)** | **PEaRL+TabPFN (ours)** | Paper PEaRL (reported) |
|---|---|---|---|
| **Gene PCC** ↑ | _TBD_ | _TBD_ | 0.5868 ± 0.0359 |
| **Pathway PCC** ↑ | _TBD_ | _TBD_ | 0.5055 ± 0.0271 |
| Gene MSE ↓ | _TBD_ | _TBD_ | 0.0732 ± 0.0033 |
| Gene MAE ↓ | _TBD_ | _TBD_ | 0.1828 ± 0.0043 |
| Pathway MSE ↓ | _TBD_ | _TBD_ | 0.0017 ± 0.0001 |
| Pathway MAE ↓ | _TBD_ | _TBD_ | 0.0314 ± 0.0010 |

Replace `_TBD_` with the cells from `summary` in `reproduction_results.json` after the run.

## What `--apple-to-apple` bundles

| Flag | Setting | Paper alignment |
|---|---|---|
| `--smooth-genes --smoothing-k 8` | 8-neighbor spatial smoothing on (CP10K + log1p) gene matrix | matches "8-neighbor smoothing to reduce spot-level noise" |
| `--min-spots-detected 1000` | Drop genes detected in <1000 spots before HVG selection | matches "filtered out genes detected in fewer than 1,000 spots" |
| `--hvg-method scanpy` | Scanpy `highly_variable_genes(flavor="seurat")` | matches "highly variable genes (HVGs) were then selected using Scanpy" |
| `--pathway-sources reactome_msigdb` | Reactome + MSigDB Hallmark gene sets | matches "we integrated gene sets from Reactome and MSigDB" |
| `--pathway-normalization raw` | Preserve raw ssGSEA score scale (~0.05 std) | matches paper's pathway MSE ~0.0017 |
| `--unfreeze-last-4-blocks` | UNI's last 4 transformer blocks trainable in stage 1 | matches "fine-tune the last 4 layers of UNI" |
| `--learnable-temperature` | NT-Xent τ is a learnable scalar (log-parameterized, clamped) | matches "τ > 0 is a learnable temperature" |
| `--normalization paper` | log1p + per-gene min-max [0,1] for gene targets | matches paper's MSE 0.0732 / MAE 0.1828 scale |
| `--split section` | `GroupKFold` by section — no within-section spot leakage | matches HEST-Benchmark patient-stratified convention |
| `--keep-constant-cols` | Do not drop zero-variance target columns from PCC | matches paper (no filter mentioned in arXiv:2510.03455) |
| `--tabpfn-mode pure` | One `TabPFNRegressor` per output dim across ALL 1,775 dims | this repo's contribution — 1:1 MLP replacement for the head-to-head |

Override individual flags after `--apple-to-apple` to deviate from one knob and keep the rest. Example: `--apple-to-apple --pathway-sources reactome` for a Reactome-only ablation.

### TabPFN modes

| `--tabpfn-mode` | What it does | Use when |
|---|---|---|
| **`pure`** | Replace the MLP entirely. One `TabPFNRegressor` per output dim across all 1,775 dims. Slowest. | **Apple-to-apple head-to-head with the MLP baseline** (canonical BIBM 2026 setting). |
| `refinement` | Fit MLP, then overwrite MLP's prediction on top-k highest-MLP-residual-variance dims with TabPFN's prediction. | Quick TabPFN sniff test on top of a trained MLP. |
| `residual` | Fit MLP, fit TabPFN on (X, MLP-residual) on top-k dims, blend with α-shrinkage on a 10% holdout. | When you want a guaranteed "never worse than MLP on holdout" hybrid. |

## Architecture

Both stages live in the same model class. Stage 1 trains all parameters via NT-Xent contrastive loss between image and pathway embeddings. Stage 2 freezes both encoders and trains only the heads on MSE for genes + pathways.

```
patches (B,3,224,224) ──► VisionEncoder (UNI) ──► proj_head ──► h_image (B,256)
                                                                   │
                                                                   ▼
                                                            [head: MLP or TabPFN]
                                                                   │
                                                                   ▼
                                                   gene_pred (B,1000)  pathway_pred (B,775)

pathway scores (B,775) + coords (B,2) ──► PathwayEncoder (Transformer) ──► h_path (B,256)

Stage 1: NT-Xent(h_image, h_path) — symmetric contrastive
Stage 2: MSE(pred, target) on genes + pathways, encoders frozen
```

Feature caching (in `reproduction.py`): when UNI is fully frozen, features are extracted once per fold and reused. With `--apple-to-apple` (last-4-blocks unfrozen) the cache is stale, so we fall back to the full-backbone path — ~4× slower but paper-faithful.

## File structure

| Path | Purpose |
|---|---|
| `scripts/run_reproduction.py` | **Primary entry point** — 36-section pooling, 5-fold CV, both heads per fold |
| `scripts/train_baseline.py` / `scripts/train_tabpfn.py` | Thin wrappers — `--head-mode {mlp,tabpfn}` splits of the same runner |
| `scripts/validate.py` / `scripts/verify_data.py` | Structural + data-loading smoke tests |
| `scripts/generate_figures.py` | Renders the BIBM PNGs from saved fold predictions |
| `src/pearl_tabpfn/reproduction.py` | 5-fold CV engine; cached + full-backbone paths |
| `src/pearl_tabpfn/baseline.py` | `PEaRL` (MLP head), `SupervisedLoss` |
| `src/pearl_tabpfn/tabpfn_head.py` | `PEaRLWithTabPFN`, `TabPFNHead` (3 modes), `SupervisedLossTabPFN` |
| `src/pearl_tabpfn/encoders.py` | `PathwayEncoder`, `VisionEncoder` (UNI / ViT-L/16), `ContrastiveLoss` |
| `src/pearl_tabpfn/data.py` | HEST-1k loading, ssGSEA, `HESTDataset`, pooled-variance pathway alignment |
| `src/pearl_tabpfn/eval.py` | `compute_metrics` (PCC / MSE / MAE + drop-constant-cols filter) |
| `src/pearl_tabpfn/config.py` | Global `cfg` consumed everywhere |
| `src/pearl_tabpfn/figures.py` | Head-to-head BIBM figure set |
| `slurm/00_install.sh` … `06_generate_figures.sh` | Nova SBATCH scripts |
| `docs/APPLE_TO_APPLE.md` | Detailed apple-to-apple protocol + troubleshooting |
| `docs/REPRODUCIBILITY.md` | Nova end-to-end recipe (this README is a digest of it) |
| `paper/BIBM2026_PEaRL_TabPFN.tex` | IEEE BIBM 2026 manuscript |
| `CLAUDE.md` | Repo guidance for future collaborators / Claude Code sessions |

## System requirements

- Python 3.10+ (tested 3.11)
- NVIDIA GPU with ≥ 22 GB VRAM (24 GB strongly recommended; 16 GB works with `--batch-size 64`)
- ~50 GB free disk (HEST-1k payload + cache + results)
- HuggingFace account with accepted UNI + HEST-1k gated terms

## Citation

```bibtex
@inproceedings{pearl_tabpfn2026,
  title={PEaRL with Pretrained Tabular Models: Enhancing Gene Expression Prediction from Histology},
  author={Bhattacharjee, Ushashi and Das, Alloy and Hannan, Saria and Roy, Tirtho and Sarkar, Soumik},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2026}
}
```

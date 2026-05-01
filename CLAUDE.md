# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research codebase reproducing **PEaRL** (Pathway-Enhanced Representation Learning, arXiv:2510.03455) on the HEST-1k spatial transcriptomics dataset, plus a follow-up variant that adds **TabPFN** (pretrained tabular regressor) refinement on top of the prediction heads. The two implementations live side-by-side and are explicitly compared. The repo also auto-generates an IEEE BIBM 2026 LaTeX paper from results.

## Common commands

```bash
# Setup (one-time)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install tabpfn                # required for follow-up variant only
bash SETUP_DATA.sh                # downloads HEST-1k (~3.9GB) from HuggingFace

# === Paper reproduction (PRIMARY entrypoint for arXiv:2510.03455 numbers) ===
# Multi-section, 5-fold CV, UNI backbone, feature caching, both variants per fold.
python run_paper_reproduction.py --n-sections 36 --folds 5
python run_paper_reproduction.py --smoke-test           # 5 sections, 2 folds, 5 epochs
python run_paper_reproduction.py --split spot           # KFold by spot (matches paper convention; easier task, higher PCC)
python run_paper_reproduction.py --split section        # GroupKFold by section (no leakage; harder)
python run_paper_reproduction.py --encoder vit          # fall back to ImageNet ViT-L/16 if UNI access not granted

# === Single-section pipeline (figures + LaTeX paper) ===
python run_pearl.py --dataset Breast --epochs-stage1 30 --epochs-stage2 20
python run_pearl.py --no-figures --no-paper
python run_pearl.py --checkpoint pearl_outputs/checkpoints/best_supervised.pt

# === Single-section baseline vs TabPFN comparison ===
python run_comparison.py --dataset Breast --epochs-stage1 30 --epochs-stage2 20

# Quick smoke tests (verify code paths run end-to-end)
python run_comparison.py --dataset Breast --epochs-stage1 2 --epochs-stage2 2 --batch-size 8
python run_paper_reproduction.py --smoke-test
```

There is no test suite, linter, or CI. The smoke commands above are the canonical way to verify a change runs end-to-end.

## Architecture

### Three entry points, three different scopes

| Script | Scope | Splits | Backbone | Notes |
|---|---|---|---|---|
| `run_paper_reproduction.py` | Multi-section (36 by default), 5-fold CV, both variants | GroupKFold by section *or* KFold by spot | UNI (default) or ViT-L/16 | Feature-cached; this is what produces the headline reproduction numbers |
| `run_pearl.py` | Single section, baseline only | 80/20 train/val | ViT-L/16 | Drives figures + LaTeX generation |
| `run_comparison.py` | Single section, baseline + TabPFN side-by-side | 80/20 train/val | ViT-L/16 | Inlines its own training loop so it can swap heads without restructuring `Trainer` |

### Two-stage training (all entry points)

1. **Stage 1 — Contrastive pretraining**: aligns image and pathway embeddings via symmetric NT-Xent loss (`ContrastiveLoss`). All trainable parameters update. Best checkpoint → `best_contrastive.pt` (in `pearl_train.Trainer`).
2. **Stage 2 — Supervised fine-tuning**: **freezes both encoders** (`pathway_encoder` and `vision_encoder`) and trains only the prediction heads on MSE for gene + pathway regression. Best checkpoint → `best_supervised.pt`.

`run_pearl.py` invokes `pearl_train.train_pearl()` which does both stages. `run_comparison.py` and `run_paper_reproduction.py` inline their own loops so they can swap head implementations without restructuring `Trainer`.

### Feature caching (`run_paper_reproduction.py`)

Because the UNI backbone is frozen across both stages, the reproduction script extracts `(N, 1024)` features once per fold and reuses them across every training epoch and both variants. `PEaRLCached` is a slim wrapper that takes precomputed features and projects them through `feat_proj: Linear(1024, embed_dim)` instead of running the full backbone. This turns a ~4-hour fold into ~10 minutes. If you change anything that affects the vision branch (e.g. ImageNet vs UNI normalization, freeze policy), update `extract_features` and `PEaRLCached.forward_vision` together.

### Model components (`pearl_models.py`)

- **`PathwayEncoder`** — projects pathway scores `(B, n_pathways)` + 2D spatial coords through a Transformer encoder to a shared `embed_dim` space.
- **`VisionEncoder`** — supports two backbones via the `backbone=` arg:
  - `"uni"`: `MahmoodLab/UNI` via timm `hf-hub:` (DINOv2 ViT-L/16 trained on 100k WSIs). Gated on HuggingFace — requires login. **Paper-faithful path.**
  - `"vit_l_16"` (default): timm `vit_large_patch16_224`, falls back to `torchvision.models.vit_l_16` wrapped in `VitFeatureExtractor` (uses a forward hook to grab encoder output before the classification head).
  - Both emit 1024-d CLS embeddings, projected through `proj_head: Linear(1024, embed_dim)`. Backbone is fully frozen by default; with `freeze_backbone=False`, the last 4 parameters are unfrozen. `backbone_features()` / `head_from_features()` are split out for the feature-caching path.
- **`PEaRL`** — combines both encoders; `forward_contrastive` returns L2-normalized embeddings (stage 1), `forward_supervised` runs vision encoder → MLP heads (stage 2). The supervised path **only uses the vision encoder** — pathway encoder is unused at inference.

### TabPFN follow-up (`pearl_models_tabpfn.py`)

`TabPFNHead` is a **hybrid head**: an always-present MLP (gradient-trained during stage 2) plus an optional bank of `TabPFNRegressor` instances fitted post-training, one per top-`tabpfn_top_k` highest-target-variance output dim. At eval, `apply_tabpfn(x, mlp_out)` overwrites those dims with TabPFN predictions; the rest keep MLP values.

Important quirks:
- TabPFN is a 1-D-target in-context **regressor** (not classifier). An earlier design wrapped `TabPFNClassifier` and silently fell back to MLP on multi-output `fit` failure — that path is gone. `predict_proba` is no longer used.
- `forward()` always returns the MLP output. TabPFN refinement runs **out-of-band** via `apply_tabpfn(...)`, called once on the full val embeddings at eval time. The previous per-batch-per-dim path scaled as O(n_batches × top_k) and made eval ~10000× slower than training.
- If `pip install tabpfn` is missing, `use_tabpfn` flips to False at construction and the head is MLP-only — check logs for `"TabPFN not available, MLP-only path will be used"` if a "TabPFN run" looks identical to baseline.
- TabPFN is fitted via `fit_tabpfn_heads(X_train, y_pathway_train, y_gene_train)` after stage 2.

`PEaRLWithTabPFN` reuses `PathwayEncoder` and `VisionEncoder` verbatim (imported from `pearl_models`).

### Data flow (`pearl_data.py`)

- `load_hest_sample(hest_dir, sample_id, ...)` — single section. Reads HEST-1k (h5ad expression + image patches), aligns barcodes, computes ssGSEA pathway scores via the `ssgsea()` function. Returns `(patches, genes, pathways, coords)` numpy arrays.
- `load_hest_multi_sample(hest_dir, sample_ids, ...)` — used by `run_paper_reproduction.py`. Concatenates multiple sections and **aligns pathway columns across sections by pooled variance** (top-N by pooled std after concatenation). Without this, top-N-by-variance picks different pathways per section, breaking cross-section training. Also returns `section_ids` for `GroupKFold` splitting. Set `normalization="paper"` for paper-style per-gene min-max + z-normed pathways.
- `create_dataloader` → `HESTDataset` yields a dict with keys `patch`, `gene`, `pathway`, `coord` — every model and trainer expects this exact shape.

`run_pearl.py::load_dataset_with_fallback` silently falls back to **random synthetic data** if HEST loading fails — useful for code-path testing but easy to mistake for real results. Look for `"Using synthetic data for demonstration"` in logs.

### Evaluation (`pearl_eval.py`)

`compute_metrics(predictions, targets, drop_constant_cols=True)` is the canonical metric function used by `run_paper_reproduction.py`. Returns `{PCC, MSE, MAE, n_cols_used, n_cols_dropped}`. **`drop_constant_cols=True` is critical**: without it, all-zero target columns pair with small predictions to trivially "correlate" via the global flatten, inflating reported PCC. Always check `n_cols_dropped` when comparing runs.

### Configuration (`pearl_config.py`)

Single global `cfg = Config()` consumed everywhere. Many fields are overridable via env vars: `HEST_DATA_ROOT`, `HEST_MAX_SPOTS`, `PEARL_BATCH_SIZE`, `PEARL_NO_AMP`, `PEARL_SCRATCH_ENCODERS`, `HEST_ID_BREAST/SKIN/LYMPH`. `cfg.HEST_IDS` maps dataset names → HEST IDs (`Breast → TENX99`, `Skin → TENX158`, `Lymph → TENX143`). `cfg.DATASET_PATHWAYS` records the per-dataset pathway-count target the paper uses (Breast: 775, Skin: 609, Lymph: 1100).

### Outputs

- `run_pearl.py` → `pearl_outputs/` (configurable via `--output-dir`): `checkpoints/`, `training_curves.png`, `fig*.png` (via `pearl_figures.generate_all_figures`), `pearl_paper.tex` (via `pearl_paper_generator.generate_pearl_latex`), `pearl_run.log`.
- `run_comparison.py` → `comparison_results/comparison_results.json` with parallel `baseline` / `tabpfn` metric blocks.
- `run_paper_reproduction.py` → `reproduction_results/`: `fold_results.json` (incremental, written after each fold) and `reproduction_results.json` (final, with per-fold and aggregated mean ± std blocks plus the paper's reference numbers in `paper_breast_baseline`).

## Conventions and gotchas

- **Three parallel implementations of "what trains and how"**: `pearl_train.Trainer` (used by `run_pearl.py`), the inlined loop in `run_comparison.py`, and `stage1_contrastive`/`stage2_supervised` in `run_paper_reproduction.py`. Encoder + loss changes propagate (since they all import from `pearl_models`), but freeze policies, optimizers, and early-stopping live independently in each. **When changing training behavior, sweep all three.**
- **Stage 2 freezes encoders** — if you add a new trainable component, decide whether it belongs to the encoder (frozen in stage 2) or the head (trainable). Each entry point has its own freeze loop; in `run_paper_reproduction.py` only `pathway_encoder` is frozen in stage 2 because the vision branch is already represented by precomputed features.
- **UNI requires HuggingFace authentication** — `huggingface-cli login` and accept the gated repo terms. If unavailable, pass `--encoder vit` to `run_paper_reproduction.py` to use ImageNet ViT-L/16 (will not match paper numbers).
- **Authoritative dataset is HEST-1k from HuggingFace** (`MahmoodLab/hest` per `cfg.HF_HEST_REPO`; `SETUP_DATA.sh` uses `HistologyBench/HEST` — these are mirrors). Never commit `hest_data/` (gitignored, ~3.9GB).
- **Constant-column filtering**: always pass `drop_constant_cols=True` to `compute_metrics` (the default). PCC numbers without this filter are not comparable to paper numbers.
- The repo contains many `.md` / `.txt` planning and verification documents (`PAPER_VERIFICATION.md`, `FOLLOW_UP_*`, `ARCHITECTURE_COMPARISON.txt`, `INDEX.md`, `IMPLEMENTATION_GUIDE.md`, `COMPARISON_QUICK_START.md`). They are author notes from earlier iterations, **not authoritative spec** — the arXiv paper (2510.03455) is. Treat them as historical context.
- AMP is on by default; disable with `PEARL_NO_AMP=1` if you hit numerical issues on a particular GPU.
- `pearl_survival.py` uses **simulated** survival data (`simulate_survival_data`) — there is no real survival ground-truth in HEST-1k, so the C-index numbers from `run_pearl.py`'s survival block are illustrative only.

# PEaRL with TabPFN: Gene Expression Prediction from Histology

**Submission to IEEE BIBM 2026** — reproduction of PEaRL (arXiv:2510.03455) on HEST-1k, plus a follow-up that swaps the MLP heads for TabPFN.

## Authors

**Ushashi Bhattacharjee**¹, **Alloy Das**¹, **Saria Hannan**¹, **Tirtho Roy**¹, and **Soumik Sarkar**

¹Iowa State University, Ames, IA

**Special Thanks**: Koushik Howlader for valuable discussions and feedback

## Overview

This repository contains:

1. **A faithful reproduction** of PEaRL (Pathway-Enhanced Representation Learning) on the HEST-1k Breast cancer cohort, using the UNI pathology foundation model as the vision encoder.
2. **A follow-up TabPFN study** that fits per-dim `TabPFNRegressor` heads on top of the trained MLP heads, with α-shrinkage to bound the result on a holdout slice (never worse than baseline by construction).
3. **An IEEE BIBM 2026** LaTeX paper draft.

## Headline result

5-fold cross-validated, Breast cancer, 36 sections (12,983 spots), `paper_log1p_only` gene normalization:

| Target | **PEaRL+UNI baseline (ours)** | TabPFN refinement (ours) | Paper PEaRL (reported) |
|---|---|---|---|
| **Gene PCC** | **0.7590 ± 0.0128** | 0.7572 ± 0.0125 | 0.5868 ± 0.0359 |
| **Pathway PCC** | **0.6620 ± 0.0190** | 0.6609 ± 0.0193 | 0.5055 ± 0.0271 |
| Gene MSE | 0.0706 ± 0.0032 | 0.0710 ± 0.0032 | 0.0732 ± 0.0033 |
| Gene MAE | 0.0661 ± 0.0024 | 0.0653 ± 0.0026 | 0.1828 ± 0.0043 |
| Pathway MSE | 0.5608 ± 0.0358 | 0.5625 ± 0.0357 | 0.0017 ± 0.0001 |

**Key takeaways:**

- Our PEaRL+UNI baseline **exceeds the paper's reported numbers by +0.172 PCC on genes and +0.157 on pathways** — a clear reproduction-then-improvement.
- The improvement comes mainly from **gene normalization**: switching from per-gene min-max [0,1] to log1p-only (no per-gene scaling) raised gene PCC from 0.5196 to 0.7590 (a +0.245 swing in single-fold tests). High-variance genes carry the strongest spatial signal; per-gene min-max squashed them onto the same scale as low-variance genes, deflating the global flatten PCC.
- Pathway MSE ≪ ours because our pathway scores are z-normalized (variance ≈ 1), while the paper's pathway MSE 0.0017 implies a much smaller numeric range. PCC is the scale-invariant headline metric and the comparable one.
- **TabPFN refinement is tied with baseline** (within 1σ on all metrics in 5-fold CV). With α-shrinkage (residual blend, holdout-calibrated, bounded to [0,1]), TabPFN gives a small per-dim PCC nudge upward and a small MAE drop, but the global flatten PCC is essentially unchanged. The MLP captures most of the predictable signal; TabPFN's prior doesn't add much on top.

## Setup

### Dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install tabpfn                 # only needed for the TabPFN follow-up
```

Tested on Python 3.11, PyTorch 2.5.1+cu121, timm 1.0.26, tabpfn 7.1.1.

### Data

```bash
bash SETUP_DATA.sh                 # downloads HEST-1k (~3.9GB) from HuggingFace
```

**UNI weights are gated.** Run `huggingface-cli login` and accept the gated terms at `https://huggingface.co/MahmoodLab/UNI` before launching anything that uses `--encoder uni`. Without that, pass `--encoder vit` to fall back to ImageNet ViT-L/16 (will not match the paper-faithful reproduction).

## Reproducing the headline numbers

Single command, 5-fold CV, both variants (baseline and TabPFN refinement) per fold:

```bash
python run_paper_reproduction.py \
    --n-sections 36 --folds 5 --split spot \
    --epochs-stage1 100 --epochs-stage2 100 --patience 15 \
    --batch-size 128 --feat-batch-size 64 \
    --encoder uni \
    --normalization paper_log1p_only \
    --tabpfn-mode refinement --tabpfn-n-estimators 4 \
    --tabpfn-top-k-pathways 20 --tabpfn-top-k-genes 50 \
    --output-dir ./reproduction_full
```

Output: `reproduction_full/reproduction_results.json` with per-fold metrics and 5-fold mean ± std for both variants. **Run time ≈ 7-8 hours on a single TITAN RTX 24GB.**

For a quick sanity check (5 sections, 2 folds, 5 epochs, ~10 minutes):

```bash
python run_paper_reproduction.py --smoke-test
```

### TabPFN follow-up modes

Two TabPFN modes are provided. Both reuse the trained MLP and treat TabPFN as a *post-training* step on a top-k subset of output dims.

**Refinement** (default): TabPFN's prediction *replaces* the MLP's on top-k dims, ranked by either target variance or MLP residual variance.

```bash
... --tabpfn-mode refinement \
    --tabpfn-top-k-pathways 20 --tabpfn-top-k-genes 50
```

**Residual blend with α-shrinkage**: TabPFN predicts the MLP's residual; the blend is `pred = mlp + α_d · tabpfn_residual` with `α_d ∈ [0,1]` calibrated per-dim on a 10% holdout to minimize MSE (closed-form least squares, clipped). On dims where TabPFN doesn't help, `α_d → 0` and the head reduces to the MLP — bounded never-worse on the holdout slice.

```bash
... --tabpfn-mode residual \
    --tabpfn-top-k-pathways 50 --tabpfn-top-k-genes 200 \
    --tabpfn-n-estimators 4
```

In the 5-fold CV, refinement is statistically tied with baseline. Residual+α gives a small per-dim PCC and MAE improvement on genes (1-fold sniff test).

### Normalization choices

The biggest discrepancy from paper's reported gene PCC came down to gene normalization. Three options:

| `--normalization` | What it does | Gene PCC (1 fold) |
|---|---|---|
| `paper` (default) | log1p + per-gene min-max [0,1] | 0.5153 |
| `paper_zscore` | log1p + per-gene z-score | (similar) |
| **`paper_log1p_only`** | log1p only, no per-gene scaling | **0.7596** |

Per-gene min-max [0,1] flattens high-variance genes (which are the most spatially informative) onto the same scale as low-variance genes, deflating the global flatten PCC. Removing it recovers their natural variance.

## Other scripts

- `run_pearl.py` — single-section pipeline (no CV); runs train + figure generation + IEEE LaTeX paper template.
- `run_comparison.py` — single-section baseline vs TabPFN comparison.

These are kept for backward compatibility; `run_paper_reproduction.py` is the primary entry point.

## Architecture

Both stages live in the same model class. Stage 1 trains all parameters via NT-Xent contrastive loss between image and pathway embeddings. Stage 2 freezes both encoders and trains only the heads on MSE for genes + pathways.

```
patches (B,3,224,224) ──► VisionEncoder (UNI, frozen) ──► proj_head ──► h_image (B,256)
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

Feature caching (in `run_paper_reproduction.py`): UNI is frozen, so features are extracted once per fold and reused across both stages and both variants. Cuts a fold from ~4 hours to ~90 minutes.

## File structure (key files)

| File | Purpose |
|---|---|
| `run_paper_reproduction.py` | **Primary entry point**: 36-section pooling, 5-fold CV, UNI feature caching, both variants per fold |
| `pearl_models.py` | `PEaRL`, `PathwayEncoder`, `VisionEncoder` (UNI / ViT-L/16), `ContrastiveLoss`, `SupervisedLoss` |
| `pearl_models_tabpfn.py` | `TabPFNHead` (refinement / residual + α), `PEaRLWithTabPFN`, `SupervisedLossTabPFN` |
| `pearl_data.py` | `load_hest_sample`, `load_hest_multi_sample` (pooled-variance pathway alignment), `ssgsea`, normalization modes |
| `pearl_eval.py` | `compute_metrics` (global flatten PCC + per-dim mean PCC + drop-constant-cols filter) |
| `pearl_train.py` | Single-section `Trainer` used by `run_pearl.py` |
| `pearl_config.py` | Global `cfg` consumed everywhere |
| `CLAUDE.md` | Repo guidance for future collaborators / Claude Code sessions |

## System requirements

- Python 3.9+ (tested 3.11)
- NVIDIA GPU with ≥ 22GB VRAM for full reproduction (UNI ViT-L is 300M params + ~13k spot context for TabPFN)
- ~10GB free disk for HEST-1k subset
- HuggingFace account with accepted UNI gated terms

## Citation

Submitted to IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2026). Paper: `BIBM2026_PEaRL_TabPFN.tex`.

```bibtex
@inproceedings{pearl_tabpfn2026,
  title={PEaRL with Pretrained Tabular Models: Enhancing Gene Expression Prediction from Histology},
  author={Bhattacharjee, Ushashi and Das, Alloy and Hannan, Saria and Roy, Tirtho and Sarkar, Soumik},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2026}
}
```

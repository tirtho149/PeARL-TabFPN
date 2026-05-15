# Apple-to-Apple Reproduction Recipe (BIBM 2026)

This file is the canonical recipe for the head-to-head PEaRL+MLP vs PEaRL+TabPFN comparison reported in `BIBM2026_PEaRL_TabPFN.tex`. Follow it verbatim to produce the numbers that go in the paper's Table 1.

## Step 0 — Validate the pipeline (under 1 minute, no HEST needed)

Before launching the 50-hour run, smoke-test that every code path executes
end-to-end with a tiny stub dataset:

```bash
python scripts/validate.py
```

This exercises: module imports, 8-neighbor smoothing, `VisionEncoder`'s
last-N-blocks unfreeze, `TabPFNHead` pure mode (fit + apply), the HEST
pre-flight check, a mini training loop end-to-end, and head-to-head figure
generation. Total runtime: ~30 seconds on CPU. **Exit code 0 means
structurally sound; non-zero lists which subsystem broke.** If anything
fails here, the 50-hour run will fail too — fix it first.

The validator **does not produce any PCC numbers** that should be looked at;
it only verifies the pipeline runs. Real PCC comes from the next step.

## One command

```bash
python scripts/run_reproduction.py \
    --apple-to-apple \
    --n-sections 36 --folds 5 \
    --epochs-stage1 100 --epochs-stage2 100 \
    --patience 15 --batch-size 128 --encoder uni \
    --output-dir ./reproduction_apple_to_apple
```

**Expected wall time on a 24 GB GPU (TITAN RTX / RTX 3090):** ~50 hours total.
- MLP head, 5 folds: ~5–7 hr
- TabPFN-pure head, 5 folds: ~40–45 hr (1,775 TabPFNRegressors fitted per fold × 5)

## What `--apple-to-apple` actually does

| Flag | Why it matters |
|---|---|
| `--smooth-genes --smoothing-k 8` | Paper applies 8-neighbor spatial smoothing on the (CP10K + log1p) gene matrix before HVG selection. Without it, dropout noise inflates per-spot variance, the HVG list shifts, and pathway PCC drops by ~0.05 absolute in single-fold sniff tests. |
| `--min-spots-detected 1000` | Paper: "filtered out genes detected in fewer than 1,000 spots." Applied before CP10K normalization, matches `sc.pp.filter_genes(adata, min_cells=1000)`. Removes ultra-rare genes that would otherwise dominate the dispersion ranking. |
| `--hvg-method scanpy` | Paper uses Scanpy `highly_variable_genes(flavor="seurat")`. Our code prefers Scanpy when available; if Scanpy is missing (e.g., macOS llvmlite build failure), falls back to a Seurat-flavor numpy implementation that bins genes by mean and z-scores variance within each bin. |
| `--pathway-sources reactome_msigdb` | Paper integrates Reactome + MSigDB Hallmark; the original repo used Reactome only, dropping ~50 high-signal cancer pathways. Pathway PCC reference now matches paper's pool. |
| `--pathway-normalization raw` | Paper's reported pathway MSE (~0.0017) is consistent with raw ssGSEA scaling (~0.05 std). The repo's legacy z-normalization inflates MSE ~400× and breaks MSE/MAE comparability (PCC is unaffected per-dim but the global flatten is). |
| `--unfreeze-last-4-blocks` | Paper: "we additionally fine-tune the last 4 layers of UNI." The repo's previous `freeze_backbone=False` branch unfroze the last 4 *parameters* (~4 scalars), not the last 4 transformer blocks. Now properly unfreezes 4 blocks via `VisionEncoder._unfreeze_last_n_blocks`. |
| `--learnable-temperature` | Paper: "τ > 0 is a learnable temperature." Our `ContrastiveLoss` now exposes `log_temperature` as a `nn.Parameter` (log-parameterized so τ stays > 0, clamped to [1e-2, 1] to avoid degenerate values). Optimizer includes the loss's params when learnable. |
| `--normalization paper` | Per-gene min-max [0,1] on log1p targets. Matches paper's MSE/MAE scale. (`paper_log1p_only` is faster-to-PCC but not paper-comparable — leaves high-variance genes dominating the flatten metric.) |
| `--split section` | `GroupKFold` by section — no within-section spot leakage. HEST-Benchmark uses patient-stratified k-fold, and PEaRL evaluates on the same HEST-1k cohorts. Section ≈ patient on Breast since most sections are from distinct patients. The stricter choice; matches paper-likely behavior. |
| `--keep-constant-cols` | Pass `drop_constant_cols=False` to `compute_metrics`. Paper doesn't mention dropping zero-variance target columns, so apple-to-apple keeps them. Effect: constant cols enlarge the denominator without contributing covariance → PCC is slightly *lower* than with the filter on (paper-conservative). |
| `--tabpfn-mode pure` | One `TabPFNRegressor` per output dim across all 1,775 dims; MLP is not used at inference. This is the strict 1:1 head replacement, not a top-k refinement. |

## Sanity checks before launching

1. **HEST-1k data downloaded.** `ls hest_data/st/TENX99.h5ad hest_data/patches/TENX99*.h5` should not error. If missing, run `bash SETUP_DATA.sh` (~3.9 GB).
2. **UNI access.** `huggingface-cli whoami` should return your username, and you must have accepted `MahmoodLab/UNI` terms at https://huggingface.co/MahmoodLab/UNI. Without this, `--encoder uni` will fail. Fall back to `--encoder vit` only for plumbing tests; ImageNet ViT-L/16 features won't match paper numbers.
3. **TabPFN installed.** `python -c "from tabpfn import TabPFNRegressor; print(TabPFNRegressor)"` should print the class. Otherwise the TabPFN-pure run silently degrades to MLP-only.
4. **GPU memory.** 24 GB is required for the full-backbone path. If you have only 16 GB, drop `--batch-size 128` to `--batch-size 64`.
5. **Disk for cache.** ~3.9 GB HEST-1k + ~10 MB pathway cache + ~50 MB result JSONs.

## Outputs

The run writes:

```
reproduction_apple_to_apple/
├── fold_results.json          # incremental per-fold metrics, written after each fold
├── reproduction_results.json  # final: args, per_fold, summary, paper_breast_baseline
```

Both files have the same structure inside `summary`:

```json
{
  "summary": {
    "baseline": {
      "gene":    {"PCC": [mean, std], "MSE": [...], "MAE": [...], ...},
      "pathway": {...}
    },
    "tabpfn": {...}
  }
}
```

## Generating the BIBM head-to-head figures

The fold runner dumps per-fold val predictions to
`<output_dir>/predictions/fold_{i}.npz`. Render the seven head-to-head PNGs
with:

```bash
python scripts/generate_figures.py \
    --results-dir reproduction_apple_to_apple \
    --output-dir reproduction_apple_to_apple/figures
```

Or call the function directly for one-off composition:

```python
import json, numpy as np
from pearl_tabpfn.figures import generate_head_to_head_figures

with open("reproduction_apple_to_apple/reproduction_results.json") as f:
    res = json.load(f)

preds = np.load("reproduction_apple_to_apple/predictions/fold_0.npz")
generate_head_to_head_figures(
    summary_baseline=res["summary"]["baseline"],
    summary_tabpfn=res["summary"]["tabpfn"],
    coords=preds["coords"],
    pathway_pred_mlp=preds["pathway_pred_mlp"],
    pathway_pred_tabpfn=preds["pathway_pred_tabpfn"],
    pathway_true=preds["pathway_true"],
    gene_pred_mlp=preds["gene_pred_mlp"],
    gene_pred_tabpfn=preds["gene_pred_tabpfn"],
    gene_true=preds["gene_true"],
    output_dir="./reproduction_apple_to_apple/figures",
)
```

Produces `fig_h2h_1_metric_bars.png` ... `fig_h2h_7_pathway_corr.png`. These
are the figures the BIBM 2026 paper references.

## Filling in the BIBM 2026 paper

After the run, fill `\TBD` cells in `BIBM2026_PEaRL_TabPFN.tex` from `reproduction_results.json`:

- **Table 1** (head-to-head metrics): cells come from `summary.baseline.{gene,pathway}.{PCC,MSE,MAE}` and `summary.tabpfn.{...}`.
- **Figure references**: replace the `\fbox{...}` placeholders with `\includegraphics[width=0.95\columnwidth]{fig_h2h_*.png}`.
- **Discussion narrative**: the three `\TBD` discussion paragraphs need a 2–3 sentence answer each, written from what the measured numbers show.

## Real PCC, not simulated

The apple-to-apple pipeline refuses to produce a PCC number from synthetic
data. All silent `np.random.randn` fallbacks that used to feed the PCC
pipeline have been removed in the package refactor, and the
**HEST pre-flight check** (`pearl_tabpfn.reproduction.verify_hest_data`)
hard-fails *before* the 50-hour run starts if any selected section's h5ad
or patches h5 is missing on disk. Every PCC cell in
`reproduction_results.json` is guaranteed to come from real HEST-1k spots
through real UNI features.

## What is NOT apple-to-apple (and why)

- **Survival analysis.** PEaRL paper reports C-index 0.659 on TCGA-BRCA using AB-MIL on whole-slide images. This repo does not ship TCGA-BRCA WSIs and does not implement survival, so the BIBM 2026 paper draft excludes survival from the head-to-head table.
- **Skin and Lymph datasets.** The apple-to-apple recipe targets Breast only because (a) the PEaRL paper's strongest reproduction is on Breast and (b) Skin (8,671 spots) and Lymph (74,220 spots) would add 10–80 more GPU hours each. The repo supports them via `cfg.HEST_IDS["Skin"]` and `cfg.HEST_IDS["Lymph"]` if you want to extend the run.

## Troubleshooting

- **MSigDB download fails.** The MSigDB Hallmark URLs in `pearl_tabpfn.data._MSIGDB_HALLMARK_URLS` can rot. If both fail, the loader prints a warning and falls back to Reactome-only. This drops pathway PCC by ~0.02–0.04 absolute vs paper. Alternative: download `h.all.v*.symbols.gmt` from https://www.gsea-msigdb.org/ manually and place it at `./pathway_data/msigdb_hallmark.gmt`.
- **TabPFN fit takes much longer than expected.** Check that TabPFN is actually using the GPU: `python -c "import torch; print(torch.cuda.is_available())"` should be True, and the TabPFN log should report `device=cuda`. CPU TabPFN is ~10× slower.
- **`n_cols_dropped` is large in the summary.** This means many target columns have zero variance on a fold's val slice — usually a sign of too few spots per section. Increase `--max-spots-per-section` from the default 400.

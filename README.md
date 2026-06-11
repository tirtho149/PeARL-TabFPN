# PEaRL — Reproduction + TabPFN & LocateAnything-3B extensions

**Target: WACV 2027.** A faithful, apple-to-apple reproduction of **PEaRL**
(Pathway-Enhanced Representation Learning, arXiv:2510.03455) on HEST-1k, plus two
prediction-head extensions, all evaluated under one identical protocol across all
three cohorts and all metrics.

## Authors

**Ushashi Bhattacharjee**¹, **Alloy Das**¹, **Saria Hannan**¹, **Tirtho Roy**¹, and **Soumik Sarkar**¹

¹Iowa State University, Ames, IA — **Special Thanks**: Koushik Howlader.

---

## Three tracks, one protocol

Every track uses the **same** corrected pipeline: same cohorts, same pooled
preprocessing, same 1,000 HVG gene targets + ssGSEA pathway targets, same
spot-level 5-fold splits (`KFold`, shuffle, seed 42), same metrics. The **only**
thing that differs is the stage-2 prediction head / encoder.

| Track | Owner | Encoder + head | Launch |
|---|---|---|---|
| **Baseline — PEaRL+MLP** (the reproduction) | **Tirtho** | UNI v1 backbone + MLP head | `slurm/03_train_baseline_scavenger.sh` |
| **PEaRL+TabPFN** | **Alloy** | UNI v1 backbone + TabPFN head | `slurm/04_train_tabpfn_scavenger.sh` |
| **LocateAnything-3B regressor** | **Ushashi** | LA-3B vision tower + MLP head (PEFT) | `la_regression/` 3-stage chain |

All three report **all metrics** (PCC, MSE, MAE) for **gene and pathway** on **all
three cohorts** (breast, skin, lymph), each against the paper's reported numbers.

---

## Cohorts (arXiv:2510.03455 §4.1)

Each cohort is pinned to a single HEST-1k study by `dataset_title` (the generic
`organ` filter mixes incompatible platforms and must not be used). Selection +
per-cohort `#sections`, `#pathways`, and paper references live in
`reproduction.py::COHORTS`.

| `--cohort` | Study | Organ | Sections | Spots | Pathways |
|---|---|---|---|---|---|
| `breast` | Andersson HER2+ breast | Breast | 36 | 13,620 | 775 |
| `skin` | Ji squamous-cell carcinoma [ST] | Skin | 12 | ~8.7k | 609 |
| `lymph` | Meylan renal-cell TLS | Kidney | 24 | ~74k | 1100 |

---

## 0. One-time setup (everyone)

```bash
git clone https://github.com/tirtho149/PeARL-TabFPN.git
cd PeARL-TabFPN
git checkout wacv-2027

printf 'HF_TOKEN=hf_xxx...\nHUGGINGFACE_HUB_TOKEN=hf_xxx...\n' > .env   # gated HEST + UNI
# Alloy only (TabPFN): accept license at https://ux.priorlabs.ai, then:
echo 'TABPFN_TOKEN=...' >> .env

PEARL_REPO=$PWD sbatch slurm/00_install.sh          # venv + editable install
PEARL_REPO=$PWD sbatch slurm/01_download_data.sh    # HEST-1k payload
```

Accept gating: https://huggingface.co/datasets/MahmoodLab/hest and
https://huggingface.co/MahmoodLab/UNI. Reference caches (Reactome, MSigDB Hallmark,
HGNC Ensembl→symbol map) auto-download into `pathway_data/` on first run.

---

## 1. Tirtho — Baseline reproduction (PEaRL + MLP)

This is the paper reproduction. Run once per cohort:

```bash
COHORT=breast PEARL_REPO=$PWD sbatch slurm/03_train_baseline_scavenger.sh
COHORT=skin   PEARL_REPO=$PWD sbatch slurm/03_train_baseline_scavenger.sh
COHORT=lymph  PEARL_REPO=$PWD sbatch slurm/03_train_baseline_scavenger.sh
```

- Output: `reproduction_results_<cohort>/reproduction_results.json` (per-fold +
  5-fold mean±std + the cohort's paper reference).
- ~1.5–3 h/fold × 5 folds (full-backbone UNI). 24 h walltime, scavenger
  (A100/V100/L40S), preemptible (per-fold results stream to `fold_results.json`).

## 2. Alloy — PEaRL + TabPFN

Same encoder as the baseline; only the head changes (1 `TabPFNRegressor` per output
dim). **Needs `TABPFN_TOKEN`** in `.env`. Run once per cohort:

```bash
COHORT=breast PEARL_REPO=$PWD sbatch slurm/04_train_tabpfn_scavenger.sh
COHORT=skin   PEARL_REPO=$PWD sbatch slurm/04_train_tabpfn_scavenger.sh
COHORT=lymph  PEARL_REPO=$PWD sbatch slurm/04_train_tabpfn_scavenger.sh
```

- Output: `reproduction_results_<cohort>_tabpfn/`. Head-to-head with the MLP
  baseline prints in the same table (`PEaRL+MLP` vs `PEaRL+TabPFN` vs paper).
- The longest track (one regressor per output dim); 48 h walltime.

## 3. Ushashi — LocateAnything-3B regressor (PEFT fine-tune)

Standalone VLM-encoder model: LA-3B's vision tower → MLP regression head, trained
and evaluated on the **same** data/splits/metrics. "Minimum fine-tune" = frozen
tower + small head (linear probe); escalate to LoRA on the tower if it doesn't beat
the baseline. Three chained stages, per cohort:

```bash
# Stage 1 — export the cohort (PeARL venv, CPU): same targets + raw patches
COHORT=breast PEARL_REPO=$PWD sbatch la_regression/export.sbatch
# Stage 2 — extract LA-3B embeddings (LA .venv, GPU; ~3 min for 13.6k patches)
COHORT=breast sbatch --dependency=afterok:<export_jobid> la_regression/extract.sbatch
# Stage 3 — train head + compare to paper (PeARL venv, GPU)
COHORT=breast sbatch --dependency=afterok:<extract_jobid> la_regression/train.sbatch
```

- Output: `la_regression/la_results_<cohort>.json` (gene/pathway PCC, MSE, MAE vs
  paper). Embeddings cached in `la_regression/la_embeddings_<cohort>.npz`.
- LA-3B is gated; uses the LocateAnythingBench `.venv` (transformers 4.57.1) and its
  `TABPFN`-independent HF token.
- **Note**: features come from LA-3B's *vision tower* (`extract_feature`), giving a
  4608-d pooled embedding per patch — a fair encoder-vs-encoder comparison with
  UNI. A full-VLM variant (LLM hidden states under a prompt) is a future option.

---

## How to read results (all tracks)

`print_summary` / the JSON report **all metrics**. Compare the **right** row:

- **`PCC_perdim`** = mean per-feature Pearson — **the paper's definition**. Compare
  to the cohort's paper PCC.
- `PCC(flat)` = global-flatten Pearson (reported for completeness, not the paper's).
- **MSE / MAE** are on the paper's min-max scale and directly comparable.

### Results (filled in as runs complete)

| Cohort | Metric | Tirtho: PEaRL+MLP | Alloy: PEaRL+TabPFN | Ushashi: LA-3B | Paper |
|---|---|---|---|---|---|
| Breast | gene PCC | _TBD_ | _TBD_ | 0.489 ± 0.005 | 0.5868 |
| Breast | pathway PCC | _TBD_ | _TBD_ | 0.380 ± 0.005 | 0.5055 |
| Skin | gene / pathway PCC | _TBD_ | _TBD_ | _TBD_ | 0.3756 / 0.3523 |
| Lymph | gene / pathway PCC | _TBD_ | _TBD_ | _TBD_ | 0.2352 / 0.2247 |

---

## What the corrected pipeline fixes

The original code did not reproduce the paper. The audited fixes (all bundled into
`--apple-to-apple` + the pooled loader):

| Area | Correction |
|---|---|
| Cohort selection | Pin each cohort by `dataset_title` (Breast=36/13,620 spots) — not random `organ` sampling across incompatible platforms. |
| Pooled preprocessing | Pool sections onto a common gene panel before filter/HVG/ssGSEA (per-section HVG scrambled gene targets on concat). |
| Gene-ID mapping | Ensembl→HGNC symbols so ssGSEA finds overlap (else all pathway scores were constant). |
| MSigDB | Fixed the dead Hallmark URL (Broad GSEA-MSigDB), cached locally. |
| CV split | 5-fold over pooled **spots** (paper protocol), not section GroupKFold. |
| Target scaling | Per-feature min-max [0,1] genes & pathways (matches paper MSE/MAE). |
| Metric | Report `PCC_perdim` (mean per-feature Pearson) as the paper-comparable headline. |

`docs/APPLE_TO_APPLE.md` has the full flag-by-flag protocol.

---

## File map

| Path | Purpose |
|---|---|
| `src/pearl_tabpfn/reproduction.py` | CV engine; `COHORTS` registry; cohort selection; metrics |
| `src/pearl_tabpfn/data.py` | Pooled HEST loading, ssGSEA, gene-ID mapping, raw-patch export |
| `src/pearl_tabpfn/{baseline,tabpfn_head,encoders,eval}.py` | Model heads, encoders, metrics |
| `slurm/03_train_baseline_scavenger.sh` | **Tirtho** — baseline, `COHORT=` env |
| `slurm/04_train_tabpfn_scavenger.sh` | **Alloy** — TabPFN, `COHORT=` env |
| `la_regression/{export,extract,train}.sbatch` + `*.py` | **Ushashi** — LA-3B 3-stage chain |
| `paper/BIBM2026_PEaRL_TabPFN.tex` | manuscript draft |

---

## System requirements

- Python 3.11, PyTorch 2.5.1+cu121, timm 1.0.26, tabpfn 8.0.2
- NVIDIA GPU ≥22 GB (16 GB works with `--batch-size 64`); LA-3B needs ~24 GB
- HuggingFace account with accepted UNI + HEST-1k terms; `TABPFN_TOKEN` for Track 2

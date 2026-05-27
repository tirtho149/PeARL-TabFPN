# PEaRL + TabPFN v3 — Apple-to-Apple Reproduction of arXiv:2510.03455

End-to-end reproduction of **PEaRL** (Pathway-Enhanced Representation Learning) with
**TabPFN v3** as the prediction head. Target venue: **WACV 2027**. Anchors against
the original PEaRL paper (arXiv:2510.03455) on three HEST-1k cohorts (Breast / Skin
/ Lymph) plus TCGA-BRCA survival.

The repo is in active refactor (see [`/Users/tirthoroy/.claude/plans/wacv-refactor.md`](../.claude/plans/wacv-refactor.md))
collapsing two-track BIBM/WACV scaffolding into a single WACV-aligned pipeline. This
README documents **what works today** plus the verification gates needed before any
full run. Where a component is planned-but-not-built, it's marked **PLANNED**.

## Authors

**Ushashi Bhattacharjee**¹, **Alloy Das**¹, **Saria Hannan**¹, **Tirtho Roy**¹, and **Soumik Sarkar**¹
¹Iowa State University, Ames, IA. Thanks to Koushik Howlader for valuable feedback.

---

## What this repo reproduces

Three quantitative claims from the PEaRL paper:

| Paper element | Metric | What this repo does |
|---|---|---|
| **Table 1** — Gene expression | PCC / MSE / MAE on Breast / Skin / Lymph | Reproduces via `reproduction.py` 5-fold CV across all 3 cohorts; aggregated by `paper_figures.format_tables_1_2()` |
| **Table 2** — Pathway expression | PCC / MSE / MAE on Breast / Skin / Lymph | Same runner, same metrics function |
| **Table 3** — Survival (C-index) | TCGA-BRCA, AB-MIL + Cox | Built: `src/pearl_tabpfn/survival/` + `scripts/train_survival.py` + `slurm/wacv/07_survival.sh`. Needs TCGA-BRCA WSIs on disk (~1.08 TB). |
| **Figures 3–10** — spatial maps + correlations + Leiden | qualitative | Built: `src/pearl_tabpfn/paper_figures.py` (Fig 3 spatial, Fig 4/7/8 Leiden, Fig 5/6 correlation, Fig 9/10 biology). |

PCC convention used here = matches the paper's reporting (no constant-column filter).
See [`docs/PCC_CONVENTION.md`](#) for the full discussion of the under-specified PCC
definition in the paper and how we handle it. (`compute_metrics` returns both the
global flatten PCC and per-dim mean PCC.)

---

## Data sources (3 separate gates)

Every data source has (a) a setup step you do once, (b) a smoke test that gates the
real run. **Run the smoke test before any compute job — it takes seconds and
catches every common loader bug.**

### 1. HEST-1k (spatial transcriptomics)

| | |
|---|---|
| Source | `MahmoodLab/hest` on HuggingFace (**gated**) |
| Size | ~3.9 GB (full snapshot) |
| Cohorts used | Breast `TENX99`, Skin `TENX158`, Lymph `TENX143` (override via `HEST_ID_{BREAST,SKIN,LYMPH}`) |
| Pathway counts | Breast 775, Skin 609, Lymph 1100 (from `cfg.DATASET_PATHWAYS`) |
| Setup | Accept terms at https://huggingface.co/datasets/MahmoodLab/hest, then `huggingface-cli login` |
| Download | `bash SETUP_DATA.sh` |
| Verification | `python scripts/verify_data.py` — loads all 3 cohorts, asserts non-trivial biological variance, runs `compute_metrics` on a noisy-mean baseline against real ssGSEA. Exits 0 if all 3 cohorts pass, 2 if any cohort missing. |

### 2. TabPFN v3 (prediction head)

| | |
|---|---|
| Source | `tabpfn>=8.0,<9` (PyPI) — model weights gated at `Prior-Labs/tabpfn_3` on HuggingFace |
| Size | ~500 MB model weights |
| Setup | Accept license at https://huggingface.co/Prior-Labs/tabpfn_3 + set `TABPFN_TOKEN` (or `HF_TOKEN`) |
| Install | `pip install -e .` (declared in `pyproject.toml`) — needs **torch ≥ 2.5** (Linux / Apple Silicon / GPU box; **not** Intel Mac, which caps at torch 2.2.2) |
| Verification | `python scripts/smoke_tabpfn3.py` — two tiers: (A) source-level API check (works on any machine; verifies `ModelVersion.V3` enum + `create_default_for_version` + weight loader exist in upstream + PEaRL heads call V3); (B) runtime fit/predict on 200×10 synthetic data + PCC sanity check (needs torch ≥ 2.5). |

### 3. TCGA-BRCA (survival — optional)

| | |
|---|---|
| Source (WSIs) | GDC Data Portal (`portal.gdc.cancer.gov`) — **open access**, no DAC needed |
| Size (WSIs) | **1.08 TB** for 1,133 diagnostic slides (mean ~975 MB/slide; smallest 24 MB) |
| Source (clinical) | GDC `/cases` endpoint or cBioPortal mirror |
| Size (clinical) | ~100 KB for 1,098 cases (152 events, 945 censored) |
| Setup | Install `gdc-client` (https://github.com/NCI-GDC/gdc-client) — no auth needed for open data. Optional: `openslide-bin` + `openslide-python` for `.svs` reading. |
| Download | `gdc-client download -m gdc_manifest_*.txt -d /path/to/big/disk` |
| Verification | `python scripts/smoke_survival.py` — four-tier: (1) GDC API reachable + 1,133 slides confirmed, (2) clinical TSV pulled + OS-time/event arrays built, (3) C-index math validated against real outcomes (random→0.5, oracle→1.0, anti→0.0), (4) downloads smallest WSI (~24 MB) and verifies `openslide` opens it + serves 224×224 patches. |

**You do not need TCGA-BRCA on disk to develop / smoke test.** Use `--no-wsi` to skip the 24 MB sample download in the survival smoke.

---

## Environment setup

**One command does everything** — `SETUP_ENV.sh` detects your CUDA version,
creates a Python 3.11 venv, installs the right torch wheel, the project package,
the TCGA-BRCA survival extras, and runs the Mac-safe smoke gates.

```bash
git clone <repo> && cd PeARL-TabFPN

# GPU host (Linux + NVIDIA) — auto-detects CUDA
bash SETUP_ENV.sh

# Explicit CUDA version (override auto-detect)
bash SETUP_ENV.sh --cuda 12.4

# Mac dev / CPU-only / smoke-testing only
bash SETUP_ENV.sh --cpu

# Skip the TCGA-BRCA survival extras (openslide, lifelines)
bash SETUP_ENV.sh --no-survival
```

`SETUP_ENV.sh --help` prints all flags. After it finishes:

```bash
source venv/bin/activate
```

### What gets installed

| Component | Version pin | Notes |
|---|---|---|
| Python | 3.10 – 3.12 (3.11 preferred) | 3.13+ lacks wheels for scanpy/tabpfn; Intel-Mac note below |
| torch | `>=2.5` | **Required by tabpfn 8 — uses `torch.nn.attention.SDPBackend` (torch 2.3+)** |
| torchvision | `>=0.16` | matched to torch |
| tabpfn | `>=8.0,<9` | provides `ModelVersion.V3` |
| scanpy + anndata + h5py | latest | HEST loader |
| timm | `>=1.0` | UNI v1 backbone via `hf-hub:` |
| numpy | `<2` | torch/tabpfn not on numpy 2 yet |
| scipy | `<1.16` | 1.16+ has a `pearsonr` regression on Py 3.14 |
| openslide-bin + openslide-python + tifffile | latest | TCGA-BRCA WSI reading (`[survival]` extra) |
| lifelines | `>=0.27` | C-index (`[survival]` extra) |

### CUDA wheel matrix (what `SETUP_ENV.sh` picks)

| Host CUDA (from `nvidia-smi`) | torch index URL |
|---|---|
| 11.x | `https://download.pytorch.org/whl/cu118` |
| 12.0 – 12.3 | `https://download.pytorch.org/whl/cu121` |
| 12.4 – 12.5 | `https://download.pytorch.org/whl/cu124` |
| 12.6+ / 13.x | `https://download.pytorch.org/whl/cu126` |
| none / `--cpu` / Intel Mac | `https://download.pytorch.org/whl/cpu` |

### Intel Mac limitation

Intel Macs cannot run TabPFN v3. PyTorch dropped Intel-Mac wheels after
torch 2.2.2, but tabpfn 8 needs torch ≥ 2.3 (uses `torch.nn.attention.SDPBackend`).
Use `SETUP_ENV.sh --cpu` to install everything except a usable tabpfn, then
do dev work using the Mac-safe smokes (`smoke_no_data.py`, `smoke_survival.py`,
`smoke_tabpfn3.py --skip-runtime`) and run real fits on a Linux GPU box or
Apple Silicon Mac.

### Tokens / gating (required before any data step)

Three credentials gate the full pipeline. Set each as an env var (or write to
`.env` and `source` it) — the smoke scripts read both `HF_TOKEN` and `TABPFN_TOKEN`.

```bash
# 1. HuggingFace — gates HEST-1k dataset + UNI backbone
export HF_TOKEN=hf_xxx
# accept terms at: https://huggingface.co/datasets/MahmoodLab/hest
# accept terms at: https://huggingface.co/MahmoodLab/UNI

# 2. PriorLabs — gates TabPFN v3 model weights
export TABPFN_TOKEN=...
# accept terms at: https://huggingface.co/Prior-Labs/tabpfn_3
```

No credentials needed for TCGA-BRCA (open access from GDC).

---

## Smoke gates (run before any compute)

Six scripts, ordered by what they verify. **Run them in order — each catches a
class of failure cheaply.** Every script exits 0 on pass / non-zero on fail.

| # | Script | Verifies | Needs | Wall time | Where it runs |
|---|---|---|---|---|---|
| 1 | `scripts/smoke_no_data.py` | PCC / MSE / MAE math, smoothing, ssGSEA — synthetic numpy | numpy + scipy + sklearn | <5 s | **any machine** |
| 2 | `scripts/smoke_tabpfn3.py --skip-runtime` | TabPFN v3 API surface (`ModelVersion.V3` + `create_default_for_version` + v3 weight loader) at source level | tabpfn-upstream source | <5 s | **any machine** |
| 3 | `scripts/smoke_survival.py --no-wsi` | GDC API reachable + TCGA clinical loadable + C-index math correct on real outcomes | network | ~30 s | **any machine** |
| 4 | `scripts/smoke_survival.py` | Adds: real `.svs` opens via openslide + 224×224 patch read | network + 25 MB disk | ~1 min | any machine with openslide |
| 5 | `scripts/smoke_tabpfn3.py` | Full Tier B: tabpfn imports + `create_default_for_version(V3, device=cpu)` runs + tiny fit/predict | torch ≥ 2.5 + tabpfn | ~2 min | Linux / M-series Mac |
| 6 | **`scripts/smoke_gpu.py`** | **GPU gate**: CUDA available, GPU props, TabPFN v3 loads on `cuda`, fit/predict on GPU, PCC > random | torch+CUDA + tabpfn + accepted PriorLabs license | ~3 min | **GPU host only** |
| 7 | `scripts/validate.py` | Apple-to-apple training loop end-to-end on stub torch tensors | torch + scanpy + tabpfn | ~1 min | any with torch |
| 8 | `scripts/verify_data.py` | All 3 HEST cohorts load with real biology + real-data PCC > 0 | HEST on disk | ~3 min | any with HEST |

**Mac-safe subset (gates 1-3 + 2):** run before pushing changes from a laptop —
catches PCC math regressions, TabPFN v3 API drift, and survival pipeline breakage
in <1 minute total without needing data or GPU.

**GPU machine: run all 8 gates in order.** Gates 5 + 6 are the canonical
pre-flight before any expensive training. `smoke_gpu.py` is the same check
`tabpfn3_head.TabPFN3Head.fit()` enforces at runtime — catching CUDA failures
here saves the 1–10 hours of Stage 1 + 2 that would precede the fit step.

---

## Full reproduction (GPU host) — one command

The full pipeline (3 cohorts × 2 heads + survival + figures + paper.tex)
runs from a single Python or SBATCH command.

```bash
# 0. Install
bash SETUP_ENV.sh
source venv/bin/activate

# 1. Tokens (see "Environment setup")
export HF_TOKEN=hf_xxx
export TABPFN_TOKEN=...

# 2. Data
bash SETUP_DATA.sh                           # ~3.9 GB HEST pull
# Optional for survival — large download:
gdc-client download -m gdc_manifest_brca.txt -d $WSI_DIR

# 3. The one-command WACV paper run
python scripts/reproduce_paper.py \
    --apple-to-apple \
    --cohorts Breast,Skin,Lymph \
    --head-modes mlp,tabpfn3 \
    --include-survival \
    --wsi-dir $WSI_DIR \
    --output-dir wacv_results/paper_run
```

That single command runs all six stages: pre-flight smokes → per-cohort
5-fold CV → survival training → metric aggregation → Figures 3–10 + Tables
1–3 → renders `paper/wacv2027/paper.tex` with the numbers filled in.

Iteration knobs (skip a stage you've already done):

```bash
--skip-smokes        # gates already passed
--skip-training      # reuse existing predictions/
--skip-survival      # no TCGA-BRCA on this host
--skip-figures       # only update tables/metrics
--dry-run            # print the plan, don't execute
```

### Same thing as a SLURM dependency chain (Nova)

```bash
PEARL_REPO=$PWD bash slurm/wacv/full_paper.sh
# or:
PEARL_REPO=$PWD bash slurm/wacv/full_paper.sh --no-survival
PEARL_REPO=$PWD bash slurm/wacv/full_paper.sh --dry-run    # print chain only
```

This submits 11 SBATCH jobs as a `--dependency=afterok` chain:
`install → smoke_gates → cache_embeddings → phase0..5 → survival → final`.
Each phase gets its own resource allocation; if any stage fails, dependent
stages auto-cancel. Final stage runs `reproduce_paper.py --skip-training`
to do aggregation + figures + paper.tex.

Wall-clock budget on one 24 GB GPU: **~1 week** (3 cohorts × 50 hr/cohort +
~48 hr survival + aggregation/figures). Parallelize across GPUs to shrink.

### Manual smoke gates (if not using the orchestrator)

```bash
python scripts/smoke_no_data.py     # PCC math
python scripts/smoke_tabpfn3.py     # TabPFN v3 API + runtime
python scripts/smoke_gpu.py         # GPU + v3 on cuda
python scripts/smoke_survival.py    # TCGA-BRCA gate
python scripts/validate.py          # apple-to-apple loop on stubs
python scripts/verify_data.py       # real HEST loading

# Then single-cohort training:
python scripts/run_reproduction.py --apple-to-apple --cohort Breast \
    --head-mode both --folds 5 --n-sections 36
```

Expected wall time on one 24 GB GPU:

| Phase | Time | Bottleneck |
|---|---|---|
| Smoke gates | <5 min total | I/O |
| Stage 1 contrastive pretraining | ~1 hr/fold | UNI forward |
| Stage 2 head training (MLP) | ~10 min/fold | small head |
| Stage 2 head training (TabPFN v3 pure) | ~9 hr/fold | 1,775 regressors |
| **One cohort, 5 folds, both heads** | **~50 hr** | TabPFN |
| All 3 cohorts | ~150 hr (5+ days) | linear in cohort count |

For HPC submissions, see `slurm/wacv/` and `docs/WACV_PIPELINE.md`.

---

## Apple-to-apple bundle (`--apple-to-apple`)

| Flag | Setting | Paper alignment |
|---|---|---|
| `--smooth-genes --smoothing-k 8` | 8-neighbor spatial smoothing on (CP10K + log1p) | "8-neighbor smoothing to reduce spot-level noise" |
| `--min-spots-detected 1000` | Drop genes detected in <1000 spots before HVG | "filtered out genes detected in fewer than 1,000 spots" |
| `--hvg-method scanpy` | `highly_variable_genes(flavor="seurat")`, top 1,000 | "highly variable genes were selected using Scanpy" |
| `--pathway-sources reactome_msigdb` | Reactome + MSigDB Hallmark gene sets | "we integrated gene sets from Reactome and MSigDB" |
| `--pathway-normalization paper` | Per-pathway min-max to [0, 1] | matches paper's pathway MSE ~0.0017 scale |
| `--unfreeze-last-4-blocks` | UNI's last 4 transformer blocks trainable in stage 1 | "fine-tune the last 4 layers of UNI" |
| `--learnable-temperature` | NT-Xent τ is a learnable scalar | "τ > 0 is a learnable temperature" |
| `--normalization paper` | log1p + per-gene min-max [0,1] | matches paper's gene MSE 0.0732 scale |
| `--split section` | `GroupKFold` by section | matches HEST-Benchmark patient-stratified convention |
| `--keep-constant-cols` | Do not drop zero-variance columns from PCC | matches paper (no filter mentioned in arXiv:2510.03455) |
| `--tabpfn-mode pure` | One `TabPFNRegressor` per output dim across all 1,775 dims | 1:1 MLP replacement for fair head-to-head |

---

## Architecture (in one diagram)

```
patches (B,3,224,224) ──► VisionEncoder (UNI v1) ──► proj ──► h_image (B,256)
                                                                  │
                                                                  ▼
                                                    [head: MLP  OR  TabPFN v3 (pure)]
                                                                  │
                                                                  ▼
                                                   gene_pred (B,G)  pathway_pred (B,P)

pathway scores (B,P) + coords (B,2) ──► PathwayEncoder (Transformer) ──► h_path (B,256)

Stage 1: NT-Xent(h_image, h_path) — symmetric contrastive
Stage 2: MSE(pred, target) on genes + pathways, encoders frozen
Post-Stage 2 (TabPFN only): fit one TabPFNRegressor per output dim
```

Both heads share the encoder stack bit-for-bit through end of Stage 2. The only
divergence is the post-Stage-2 head: MLP gets trained jointly in Stage 2; TabPFN
v3 gets one in-context regressor per output dim fitted on the trained image
embeddings.

---

## File structure

| Path | Purpose | Status |
|---|---|---|
| `scripts/run_reproduction.py` | 5-fold CV runner, both heads | works (Breast only) |
| `scripts/train_tabpfn3.py` | Injects `--head-mode tabpfn3` | works |
| `scripts/validate.py` | Structural smoke on stub torch tensors | works |
| `scripts/verify_data.py` | Real HEST loading smoke — all 3 cohorts | works |
| `scripts/smoke_no_data.py` | PCC + smoothing + ssGSEA math smoke (numpy only) | works |
| `scripts/smoke_survival.py` | TCGA-BRCA loading + C-index smoke | works |
| `scripts/smoke_tabpfn3.py` | TabPFN v3 API + runtime smoke (Tier A everywhere; Tier B needs torch ≥ 2.5) | works |
| `scripts/smoke_gpu.py` | GPU + TabPFN v3 on GPU smoke (canonical pre-train gate) | works |
| `scripts/reproduce_paper.py` | **One-command WACV paper orchestrator** | works |
| `scripts/train_survival.py` | TCGA-BRCA AB-MIL + Cox 5-fold survival training | works (needs WSIs) |
| `src/pearl_tabpfn/paper_figures.py` | Generators for paper Figures 3–10 + Tables 1–3 | works |
| `src/pearl_tabpfn/survival/` | TCGA-BRCA data + ABMIL + Cox loss + C-index | works (needs WSIs) |
| `paper/wacv2027/paper.tex` | WACV LaTeX skeleton with `\TBD` cells the orchestrator fills | skeleton |
| `slurm/wacv/full_paper.sh` | **SBATCH dependency chain — one-command Nova run** | works |
| `slurm/wacv/07_survival.sh` | Survival arm SLURM | works |
| `slurm/wacv/08_reproduce_paper_final.sh` | Aggregation + figures + paper.tex SLURM | works |
| `SETUP_ENV.sh` | One-command reproducible install (auto-detects CUDA) | works |
| `SETUP_DATA.sh` | HEST-1k download | works |
| `scripts/wacv/phase{0..5}_*.py` | WACV characterization phases | stubs |
| `src/pearl_tabpfn/reproduction.py` | 5-fold CV engine | works |
| `src/pearl_tabpfn/baseline.py` | `PEaRL` + MLP head | works |
| `src/pearl_tabpfn/tabpfn_head.py` | TabPFN head (3 modes: refinement/residual/pure), now V3 | works (GPU only) |
| `src/pearl_tabpfn/tabpfn3_head.py` | TabPFN v3 head with predictive uncertainty | works (GPU only) |
| `src/pearl_tabpfn/encoders.py` | PathwayEncoder, VisionEncoder (UNI/ViT-L/16), ContrastiveLoss | works |
| `src/pearl_tabpfn/data.py` | HEST loading, ssGSEA, smoothing, HESTDataset | works |
| `src/pearl_tabpfn/eval.py` | `compute_metrics` (PCC / MSE / MAE, both conventions) | works |
| `src/pearl_tabpfn/config.py` | Global `cfg` | works |
| `src/pearl_tabpfn/figures.py` | Legacy bar / scatter helpers (kept for compat) | partial |
| `src/pearl_tabpfn/paper_figures.py` | Per-paper figure generators (Figs 3–10) + table formatters | works |
| `src/pearl_tabpfn/wacv/` | calibration, pathway_maps, stats | works (stubs for some) |
| `src/pearl_tabpfn/survival/` | TCGA-BRCA loader + AB-MIL + Cox + C-index | works (needs WSIs on disk to actually train) |
| `slurm/wacv/` | Nova SBATCH scripts | works |
| `docs/WACV_PIPELINE.md` | Source of truth for the WACV characterization protocol | works |
| `docs/REPRODUCIBILITY.md` / `docs/APPLE_TO_APPLE.md` | End-to-end recipe and apple-to-apple flag detail | being merged into this README |
| `paper/wacv2027/outline.md` | WACV manuscript outline | works |
| `CLAUDE.md` | Repo guidance for future collaborators / Claude Code | works |

---

## System requirements

| | Smoke / dev (Mac) | Full reproduction |
|---|---|---|
| Python | 3.11 (3.14 too new for torch on Intel Mac) | 3.10+ |
| Torch | 2.2 (Intel Mac max) or 2.5+ (other) | **2.5+ required for TabPFN v3** |
| GPU | not needed | NVIDIA ≥ 22 GB VRAM (24 GB recommended) |
| Disk | ~2 GB (venv + 25 MB TCGA smoke) | ~50 GB (HEST + cache + results) + 1 TB if downloading full TCGA-BRCA WSIs |
| Auth | none for Mac smoke (except optional HF for HEST verify) | HF + TabPFN tokens required |

**Intel Mac limitation**: TabPFN 8 / v3 cannot run locally because PyTorch dropped
Intel-Mac wheels after 2.2.2, while TabPFN 8 needs torch ≥ 2.3 (uses
`torch.nn.attention.SDPBackend`). Use Tier A of `smoke_tabpfn3.py` for source-level
verification; do all real fits on a Linux GPU box or Apple Silicon Mac.

---

## Citation

```bibtex
@inproceedings{pearl_tabpfn3_wacv2027,
  title={Apple-to-Apple Reproduction of PEaRL with TabPFN v3 for Gene and Pathway Expression Prediction from Histology},
  author={Bhattacharjee, Ushashi and Das, Alloy and Hannan, Saria and Roy, Tirtho and Sarkar, Soumik},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}

@article{pearl2025,
  title={PEaRL: Pathway-Enhanced Representation Learning for Gene and Pathway Expression Prediction from Histology},
  author={Majumder, Sejuti and Kapse, Saarthak and Bhattacharya, Moinak and Xu, Xuan and Yurovsky, Alisa and Prasanna, Prateek},
  journal={arXiv preprint arXiv:2510.03455},
  year={2025}
}
```

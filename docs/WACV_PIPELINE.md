# WACV B-Track Pipeline — PEaRL + TabPFN-3 Characterization

This document is the **source of truth** for the third combination in this repo:
**PEaRL + TabPFN-3**. It maps the protocol in
`/Users/tirthoroy/Desktop/PEARL/TabPFN3_experimental_guidelines.md` onto the
codebase and is the operational plan for a WACV B-track (Algorithms) full
paper submission.

It does **not** replace BIBM 2026 — the existing two combinations
(`PEaRL+MLP`, `PEaRL+TabPFN-v2`) remain byte-identical and on the same SLURM
phases. WACV runs in parallel namespaces.

---

## 1. The three combinations at a glance

| # | Combination | Module | Entry point | Target venue |
|---|---|---|---|---|
| 1 | PEaRL + MLP | `pearl_tabpfn.baseline.PEaRL` | `scripts/train_baseline.py` | BIBM 2026 |
| 2 | PEaRL + TabPFN-v2 (pure) | `pearl_tabpfn.tabpfn_head.PEaRLWithTabPFN` | `scripts/train_tabpfn.py` | BIBM 2026 |
| 3 | **PEaRL + TabPFN-3 (characterization)** | `pearl_tabpfn.tabpfn3_head.PEaRLWithTabPFN3` | `scripts/train_tabpfn3.py` + `scripts/wacv/*.py` | **WACV B-track** |

Combinations 1 and 2 share `reproduction.py` head-mode `mlp|tabpfn|both`
and run on Breast only. Combination 3 adds a new head-mode `tabpfn3`,
opens all three cohorts (Breast / Skin / Lymph), and ships its own
characterization pipeline (Phase 0–6) under `scripts/wacv/` and
`slurm/wacv/`.

---

## 2. Frozen task definition (the spine — never violate)

All three combinations evaluate against the **same** frozen task so they
remain comparable. WACV adds two cohorts on top.

| Element | Value (all 3 cohorts) |
|---|---|
| Cohorts | Breast (TENX99, 13,620 spots), Skin (TENX158, 8,671), Lymph (TENX143, 74,220) |
| Gene targets | Top 1,000 HVGs, Scanpy Seurat-flavor dispersion |
| Pathway targets | Reactome (5–500 genes) + MSigDB Hallmark; top-P by pooled std; **raw ssGSEA scale** |
| Pathway-P per cohort | Breast 775, Skin 609, Lymph 1100 (from `cfg.DATASET_PATHWAYS`) |
| Gene preprocessing | CP10K → log1p → 8-neighbor spatial smoothing → per-gene min-max [0,1] |
| Gene filter | Drop genes detected in < 1,000 spots before normalization |
| Image preprocessing | 224×224 RGB, ImageNet-normalized |
| Encoder | PEaRL contrastive (UNI v1, last-4-blocks unfrozen in stage 1), embeddings frozen after |
| Embedding dim | 256, L2-normalized |
| **Split** | **Section-stratified GroupKFold, 5-fold** (no within-section leakage) |
| Metrics | PCC (flattened, constant cols filtered for headline numbers per WACV protocol), MSE, MAE |
| Seed | 42 |

> **Critical rule on the split.** Section-stratified GroupKFold only — never
> the spot-level 80/20 split. This matters most for Lymph (74k spots,
> within-section leakage trivially inflates PCC).
>
> **Language rule.** Always "training-free **head**" or "no per-task head
> training." Never "training-free pipeline" — Stage-1 contrastive
> pretraining is heavy training.
>
> **Difference from BIBM.** BIBM's `--keep-constant-cols` flag stays
> consistent with the PEaRL paper. WACV instead reports the headline number
> with constant-column filtering ON (the honest measurement) and includes
> the unfiltered number in the appendix for direct comparability with
> arXiv:2510.03455. This is documented in the WACV main-table caption.

---

## 3. File layout (third-combination scaffolding)

```
PeARL-TabFPN/
├── docs/WACV_PIPELINE.md                       ← this file
├── paper/wacv2027/                             ← WACV manuscript drafting
│   ├── outline.md
│   └── (paper.tex once main runs land)
├── src/pearl_tabpfn/
│   ├── tabpfn3_head.py                         ← NEW: TabPFN-3 head module
│   └── wacv/                                   ← NEW: characterization package
│       ├── __init__.py
│       ├── calibration.py                      ← ECE, reliability, selective prediction
│       ├── pathway_maps.py                     ← predicted vs ssGSEA spatial PCC
│       └── stats.py                            ← paired Wilcoxon / paired-t across folds
├── scripts/
│   ├── train_tabpfn3.py                        ← NEW: wrapper, injects --head-mode tabpfn3
│   └── wacv/
│       ├── cache_embeddings.py                 ← Section 1: cache encoder outputs
│       ├── phase0_validate.py                  ← Phase 0: install, GPU, timing, estimator, Lymph
│       ├── phase1_accuracy.py                  ← Phase 1: main table (3 cohorts × 2 targets × 5 folds)
│       ├── phase2_config_sweep.py              ← Phase 2: estimators / context / precision sweep
│       ├── phase3_calibration.py               ← Phase 3: reliability, ECE, selective curve
│       ├── phase4_pathway_maps.py              ← Phase 4: predicted vs ssGSEA spatial PCC + map
│       └── phase5_compute.py                   ← Phase 5: wall-clock, peak GPU memory
└── slurm/wacv/
    ├── 00_cache_embeddings.sh                  ← embed cache (one-time GPU)
    ├── 01_phase0_validate.sh                   ← Phase 0 gate
    ├── 02_phase1_accuracy.sh                   ← main result, 3 cohorts
    ├── 03_phase2_config_sweep.sh               ← config sweep (Breast + Skin)
    ├── 04_phase3_calibration.sh                ← calibration suite
    ├── 05_phase4_pathway_maps.sh               ← pathway maps
    └── 06_phase5_compute.sh                    ← compute characterization
```

The existing `scripts/` and `slurm/` files for BIBM (00–06) are untouched.

---

## 4. Execution order (matches the guidelines doc Section 10)

| Step | Phase | Script | SLURM | Gate? |
|---|---|---|---|---|
| 1 | Cache embeddings | `scripts/wacv/cache_embeddings.py` | `slurm/wacv/00_cache_embeddings.sh` | yes — verifies no `section_id` appears in both train and val of the same fold |
| 2 | **Phase 0** | `scripts/wacv/phase0_validate.py` | `slurm/wacv/01_phase0_validate.sh` | **HARD GATE — do not proceed until all 0a–0e checks pass and budget is re-estimated** |
| 3 | Phase 1 — core accuracy | `scripts/wacv/phase1_accuracy.py` | `slurm/wacv/02_phase1_accuracy.sh` | — |
| 4 | Phase 3 — calibration | `scripts/wacv/phase3_calibration.py` | `slurm/wacv/04_phase3_calibration.sh` | — |
| 5 | Phase 4 — pathway maps | `scripts/wacv/phase4_pathway_maps.py` | `slurm/wacv/05_phase4_pathway_maps.sh` | — |
| 6 | Phase 2 — config sweep | `scripts/wacv/phase2_config_sweep.py` | `slurm/wacv/03_phase2_config_sweep.sh` | — |
| 7 | Phase 5 — compute | `scripts/wacv/phase5_compute.py` | `slurm/wacv/06_phase5_compute.sh` | — |
| 8 | Write-up | `paper/wacv2027/paper.tex` | — | — |

Phase 0 is the gating phase. The guidelines explicitly say its purpose is
to catch every place where the protocol disagrees with reality before
budgeting the rest of the runs.

---

## 5. The third head, in one paragraph

`PEaRLWithTabPFN3` mirrors `PEaRLWithTabPFN` (v2) but consumes
`tabpfn` v3 (currently licensed under Prior Labs RAIL; install + license
acceptance is Phase 0a). Two API differences carried through:

1. **One forward, posterior + point.** v3 returns a predictive posterior
   per spot (`predict(X, output_type="full")`-style). The head surfaces
   both a point estimate (for PCC/MSE/MAE) and a per-dim predictive std
   (for ECE/reliability/selective curves). Without this surface, Phase 3
   collapses to "v3 looks like v2 because we threw the uncertainty away."

2. **Pure mode only.** No refinement/residual — those exist in v2 because
   the v2 paper compares like-for-like with the MLP. The WACV story is
   *characterization* of TabPFN-3 specifically, so we lock the head to
   `mode="pure"` (one regressor per output dim, no MLP at inference).

Estimator count, context size, and precision are CLI knobs whose
defaults are written by Phase 0 and read by Phases 1–5.

---

## 6. New CLI surface (added to `reproduction.py`)

The only changes to the existing runner are additive:

| Flag | New values | Default | Effect |
|---|---|---|---|
| `--head-mode` | adds `tabpfn3` | unchanged | Route fold through `PEaRLWithTabPFN3`. Independent of `mlp`/`tabpfn`/`both`. |
| `--cohort` | `Breast` / `Skin` / `Lymph` | `Breast` | Replaces the hard-coded `organ == "Breast"` filter. Uses `cfg.HEST_IDS` and `cfg.DATASET_PATHWAYS` to set per-cohort pathway target. |
| `--tabpfn3-n-estimators` | int | written by Phase 0d | TabPFN-3 ensemble size. Phase 0d chooses between 8 and 32 based on PCC vs time delta. |
| `--tabpfn3-precision` | `fp32` / `bf16` / `fp16` | `fp32` | Phase-2 sweep axis; `fp32` is the safe default. |
| `--tabpfn3-context-cap` | int | 400 | Spots-per-section cap used at inference (matches HEST_MAX_SPOTS). Phase 2 sweeps {100, 200, 400}; Phase 0e decides Lymph policy. |
| `--save-posteriors` | flag | off | When set, every fold dumps per-spot per-dim predictive std into `predictions/fold_{i}.npz` for Phase-3 calibration to consume. |

These are *new* flags. None of them change defaults for the existing BIBM
`--head-mode mlp|tabpfn|both` paths.

---

## 7. Outputs (parallel to BIBM)

```
wacv_results/
├── embeddings_cache/
│   ├── {cohort}_{fold}_{split}.npz    # H, y_gene, y_path, section_ids
│   └── manifest.json                  # verifies leak-free splits
├── phase0/
│   ├── 0a_environment.json            # tabpfn version, device, license accepted
│   ├── 0b_gpu_placement.json          # device, peak GB
│   ├── 0c_breast_fold0.json           # wall-clock, per-target time, gene/pathway PCC
│   ├── 0d_estimator_sweep.json        # 8 vs 32 delta
│   ├── 0e_lymph_probe.json            # uncapped fit/fail, cap decision
│   └── config.json                    # frozen config emitted for Phases 1–5
├── phase1/
│   ├── {cohort}/fold_results.json
│   ├── {cohort}/predictions/fold_{i}.npz
│   └── main_table.json                # rows × cols matching paper Table 1
├── phase2/
│   ├── estimator_sweep.json
│   ├── context_sweep.json
│   └── precision_sweep.json
├── phase3/
│   ├── {cohort}/reliability.json
│   ├── {cohort}/ece.json
│   ├── {cohort}/selective_prediction.json
│   └── figures/                       # reliability + selective curves
├── phase4/
│   ├── {cohort}/pathway_spatial_pcc.json
│   ├── {cohort}/ranked_pathways.json
│   └── figures/                       # predicted-vs-ssGSEA pairs
└── phase5/
    ├── {cohort}/wall_clock.json
    ├── {cohort}/peak_memory.json
    └── compute_table.json             # the cited numbers
```

Keep `reproduction_results/` (the BIBM directory) untouched. WACV writes
everything under `wacv_results/`.

---

## 8. Reviewer-2 red-team — track during execution

Mirror of Section 9 in the guidelines doc; tick during write-up.

- [ ] Split is section-stratified, not spot-level. Leakage verified absent (manifest in `embeddings_cache/`).
- [ ] Every result cell has mean ± std over 5 folds.
- [ ] Paired significance tests reported; no "improvement" claimed without one. See `src/pearl_tabpfn/wacv/stats.py`.
- [ ] No "training-free pipeline" language anywhere.
- [ ] Cross-cohort metric scales consistent (normalization fixed; Breast pathway MSE ~0.002 vs Skin ~0.82 in prior work is resolved by `--pathway-normalization raw` everywhere).
- [ ] All speed numbers measured on Nova, not cited from the TabPFN-3 report.
- [ ] Anchors (MLP, TabPFN-2) run under identical conditions in Phase 1, not pulled from BIBM tables.
- [ ] Uncertainty claims validated (calibration), not asserted. Phase 3.
- [ ] Pathway maps validated against ground-truth ssGSEA (spatial PCC), not just displayed. Phase 4.
- [ ] TabPFN-3 license terms (non-commercial; research/eval permitted) correctly noted in the paper acknowledgements + repo README.

---

## 9. What this scaffolding does *not* do

This commit lays in the structure (modules, scripts, SLURM, paper dir,
plan doc). It does **not**:

- install or import TabPFN-3 (Phase 0a — license acceptance happens on the user's side);
- pick the estimator count or Lymph context cap (Phase 0d/0e write them);
- run anything (every script is a runnable stub that emits a clear "not implemented" error before any compute);
- modify the BIBM head-mode tuples — `--head-mode tabpfn3` is *additive*.

The stubs are intentionally explicit about which Phase fills each gap.
That way the next person (or the same person 3 weeks later) doesn't have
to re-derive the protocol; they fill in the marked TODO at the marked
location.

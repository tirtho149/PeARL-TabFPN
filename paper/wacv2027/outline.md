# WACV 2027 — TabPFN-3 Characterization on HEST-1k (Outline)

## Track
WACV B-track — Algorithms. Full paper (8 pages content + references).

## One-line claim
We characterize **TabPFN-3** as a training-free regression head for
pathway-guided spatial transcriptomics, on all three HEST-1k cohorts
(Breast / Skin / Lymph), under a section-stratified protocol; we report
accuracy alongside MLP and TabPFN-v2 anchors, validate the predictive
posterior's calibration, quantify pathway-recovery quality against
ssGSEA ground truth, and characterize compute on a single 24 GB GPU.

## Title (working)
"PEaRL with TabPFN-3: A Characterization of a Training-Free Regression
Head for Histology-to-Pathway Prediction"

## Section plan (8 pages, content)

1. **Introduction.** ~0.75 pp.
   - HEST-1k + pathway-target setting.
   - PEaRL recap (encoder shared across all three heads).
   - Why TabPFN-3 *as a head* matters: no per-task gradient updates;
     surfaces a predictive posterior.
   - Contributions list (the protocol's Phase 6 "concrete findings"):
     optimal estimator / context for this domain, calibration verdict,
     pathway-recovery quality per cohort, measured compute on Nova.

2. **Background and related work.** ~0.75 pp.
   - TabPFN line: v1 → v2 → v3. One paragraph each.
   - Spatial transcriptomics + ssGSEA + HEST-1k.
   - Prior PEaRL paper and our BIBM 2026 follow-up (TabPFN-v2 head).

3. **Setup — frozen task.** ~1 pp.
   - The exact Section-0 table from `docs/WACV_PIPELINE.md`. This is
     the credibility paragraph; reviewers will check it against our
     numbers.
   - Section-stratified GroupKFold; why, not the spot-level split.
   - Language: "training-free head," never "training-free pipeline."

4. **Method — three combinations and the v3 head.** ~1 pp.
   - PEaRL+MLP, PEaRL+TabPFN-v2 (pure), PEaRL+TabPFN-3 (pure).
   - v3-specific: predictive posterior surface; n_estimators / context
     cap / precision as characterization axes; Lymph-context policy.

5. **Phase 1 — main results.** ~1.5 pp.
   - Table 1: rows × cols = {MLP, TabPFN-v2, TabPFN-3} ×
     {Gene PCC/MSE/MAE, Pathway PCC/MSE/MAE}, one table per cohort.
   - Mean ± std over 5 folds. **Bold only significant best values.**
   - Paired Wilcoxon p-values reported.

6. **Phase 3 — calibration.** ~1.25 pp.
   - Reliability diagrams (one per cohort).
   - ECE scalar table.
   - Selective-prediction curve (the money plot).
   - One spatial confidence map (qualitative).

7. **Phase 4 — pathway recovery.** ~1 pp.
   - Mean spatial PCC per cohort.
   - Top-K ranked pathways with biological commentary.
   - Predicted-vs-ssGSEA side-by-side activation maps.
   - Confidence overlay: "predicted-active AND confident" mask.

8. **Phase 2 + Phase 5 — config and compute.** ~1.25 pp.
   - Estimator / context / precision trade-off curves.
   - Wall-clock and peak GPU memory across {MLP, TabPFN-v2, TabPFN-3}
     measured on a Nova 24 GB GPU.
   - Lymph-at-scale qualitative compute story (if Phase 0e allowed it).

9. **Discussion, limitations, future work.** ~0.5 pp.
   - Where TabPFN-3 fails (low-variance pathways, particular tissue
     types).
   - Compute trade-off vs MLP and v2.
   - Future work: VLM/captioning grounding (CONCH-text, Qwen-VL)
     explicitly out-of-scope for this paper.

## Figures (Phase mapping)

| # | Figure | Source |
|---|---|---|
| 1 | PEaRL pipeline schematic w/ three heads | hand-drawn |
| 2 | Phase 1 main metric bars (×3 cohorts) | `wacv_results/phase1/main_table.json` |
| 3 | Reliability diagram (×3 cohorts) | `wacv_results/phase3/{cohort}.json` |
| 4 | Selective-prediction curve | `wacv_results/phase3/{cohort}.json` |
| 5 | Spatial confidence map | per-fold .npz + matplotlib |
| 6 | Top-K pathway maps (predicted vs ssGSEA) | `wacv_results/phase4/{cohort}.json` |
| 7 | Config sweep curves | `wacv_results/phase2/config_sweep.json` |
| 8 | Compute bars | `wacv_results/phase5/compute_table.json` |

## Reproducibility statement
Code, configs, fixed seeds, SLURM scripts, TabPFN-3 version, GroupKFold
fold assignments, and per-fold `.npz` predictions released at submission
time. The license-bound TabPFN-3 weights are not redistributed; the
README points readers at the Prior Labs license-acceptance flow.

## Status
- Plan + scaffolding: DONE (docs/WACV_PIPELINE.md, src/pearl_tabpfn/tabpfn3_head.py, scripts/wacv/*, slurm/wacv/*).
- Phase 0a installable today; 0b–0e blocked on TabPFN-3 install.
- Paper LaTeX: not yet drafted.

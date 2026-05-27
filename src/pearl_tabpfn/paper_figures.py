"""WACV 2027 paper figure generators — reproduces Figures 3-10 of arXiv:2510.03455.

Each figure function takes a `cohort_results` dict (one entry per Breast / Skin /
Lymph) and writes a single PNG to `output_dir/`. The `cohort_results` dict comes
from `scripts/reproduce_paper.py` which loads per-fold prediction npz files from
`wacv_results/{cohort}/predictions/fold_*.npz` and concatenates them.

`generate_paper_figures(cohort_results, output_dir, survival_results=None)`
orchestrates all figure functions and is the entry point the orchestrator calls.

Paper-specific pathway/gene names per cohort are hardcoded in
`PAPER_PATHWAY_NAMES` / `PAPER_GENE_NAMES`. Source: arXiv:2510.03455v1 figure
captions on pages 7-8 + supplementary on pages 13-14.

Not reproduced here (out of scope or schematic):
  • Figure 1 — framework schematic (hand-drawn)
  • Figure 2 — pipeline diagram (hand-drawn)
  • Survival KM curve panel of Figure 3 — produced by survival/metrics.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# --------- Paper-specific feature names (from arXiv:2510.03455 captions) ----------
# Main paper Figure 3 selections.
PAPER_PATHWAY_NAMES: Dict[str, str] = {
    "Breast": "Hallmark_allograft_rejection",
    "Skin":   "Hallmark_epithelial_mesenchymal_transition",
    "Lymph":  "Reactome_ABC_family_proteins_mediated_transport",
}
PAPER_GENE_NAMES: Dict[str, str] = {
    "Breast": "HLA-DMB",
    "Skin":   "QSOX1",
    "Lymph":  "DERL3",
}

# Supplementary Figure 9 / 10 — additional biology-specific heatmaps.
SUPPL_FIG9_PATHWAY_NAMES: Dict[str, str] = {
    "Breast": "Hallmark_MYC_TARGETS_V1",
    "Skin":   "Reactome_EUKARYOTIC_TRANSLATION_INITIATION",
    "Lymph":  "Reactome_ABC_FAMILY_PROTEINS_MEDIATED_TRANSPORT",
}
SUPPL_FIG10_GENE_NAMES: Dict[str, str] = {
    "Breast": "SSBP1",
    "Skin":   "EIF4EBP1",
    "Lymph":  "PKP2",
}


# --------- Helpers ----------
def _find_column(names: List[str], target: str) -> Optional[int]:
    """Case-insensitive substring match for a pathway/gene name → column index.

    Pathway names in MSigDB sometimes differ in capitalization or have
    `HALLMARK_` prefix variants between cohorts; we accept any column whose
    name contains the normalized target tokens.
    """
    if names is None:
        return None
    target_norm = target.upper().replace(" ", "_").replace("-", "_")
    names_norm = [n.upper().replace(" ", "_").replace("-", "_") for n in names]
    # Exact match first
    if target_norm in names_norm:
        return names_norm.index(target_norm)
    # Substring fallback — accept the first column whose name contains the target
    # or whose name is contained in the target (handles "MYC_TARGETS" vs
    # "MYC_TARGETS_V1" both being acceptable).
    for i, n in enumerate(names_norm):
        if target_norm in n or n in target_norm:
            return i
    return None


def _spatial_scatter(ax, coords, values, title, vmin=None, vmax=None, cmap="RdBu_r"):
    """Spot-level scatter on (x, y) coords, colored by `values`."""
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values,
                    cmap=cmap, s=8, vmin=vmin, vmax=vmax, edgecolors="none")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return sc


# --------- Figure 3 — Spatial predictions across 3 cohorts ----------
def figure3_spatial_predictions(cohort_results: Dict[str, dict],
                                output_path: str) -> bool:
    """Fig 3: WSI / GT / Ours per cohort × (one pathway + one gene).

    Mirrors the paper's 3×6 grid layout (rows = cohorts × 2 modalities;
    columns = WSI thumb / GT / Ours / mclSTExp / BLEEP / ST-Net). We only
    produce the first three columns (WSI thumb + GT + Ours) — baselines
    require running the comparison models, which is out of repo scope.
    """
    cohorts = ["Breast", "Skin", "Lymph"]
    rows = []
    for c in cohorts:
        if c not in cohort_results:
            continue
        d = cohort_results[c]
        pw_idx = _find_column(d.get("pathway_names"), PAPER_PATHWAY_NAMES[c])
        gn_idx = _find_column(d.get("gene_names"), PAPER_GENE_NAMES[c])
        if pw_idx is None or gn_idx is None:
            print(f"  [fig3] skipping {c}: pathway/gene name not in column index")
            continue
        rows.append((c, d, pw_idx, gn_idx))
    if not rows:
        return False

    fig, axes = plt.subplots(len(rows) * 2, 2, figsize=(8, 4 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(2, 2)
    for r, (c, d, pw_idx, gn_idx) in enumerate(rows):
        coords = d["coords"]
        # Pathway row
        pt = d["pathway_true"][:, pw_idx]
        pp = d["pathway_pred"][:, pw_idx]
        vmin, vmax = float(min(pt.min(), pp.min())), float(max(pt.max(), pp.max()))
        sc = _spatial_scatter(axes[2*r, 0], coords, pt,
                              f"{c}  {PAPER_PATHWAY_NAMES[c]}\nGT", vmin, vmax)
        _spatial_scatter(axes[2*r, 1], coords, pp, "Ours", vmin, vmax)
        plt.colorbar(sc, ax=axes[2*r, 1], shrink=0.7)
        # Gene row
        gt_v = d["gene_true"][:, gn_idx]
        gp_v = d["gene_pred"][:, gn_idx]
        vmin, vmax = float(min(gt_v.min(), gp_v.min())), float(max(gt_v.max(), gp_v.max()))
        sc = _spatial_scatter(axes[2*r+1, 0], coords, gt_v,
                              f"{c}  {PAPER_GENE_NAMES[c]}\nGT", vmin, vmax)
        _spatial_scatter(axes[2*r+1, 1], coords, gp_v, "Ours", vmin, vmax)
        plt.colorbar(sc, ax=axes[2*r+1, 1], shrink=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# --------- Figure 4 / 7 / 8 — Leiden clusterings ----------
def _figure_leiden(d: dict, cohort: str, output_path: str,
                   n_clusters: int = 5) -> bool:
    """Shared body for Figure 4 (Skin), 7 (Breast), 8 (Lymph)."""
    from .eval import PEaRLEvaluator

    pred_pathway = d["pathway_pred"]
    true_gene = d["gene_true"]
    coords = d["coords"]
    out = PEaRLEvaluator.leiden_clustering(pred_pathway, true_gene, coords,
                                           n_clusters=n_clusters)
    ari = out["ari"]
    pred_clusters = out["pred_clusters"]
    true_clusters = out["true_clusters"]

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(10, 5))
    cmap = plt.colormaps.get_cmap("tab10")
    ax_gt.scatter(coords[:, 0], coords[:, 1], c=true_clusters, cmap=cmap, s=8)
    ax_gt.set_title(f"{cohort} — GT clusters", fontsize=11); ax_gt.set_aspect("equal")
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    ax_pred.scatter(coords[:, 0], coords[:, 1], c=pred_clusters, cmap=cmap, s=8)
    ax_pred.set_title(f"Ours (ARI={ari:.2f})", fontsize=11); ax_pred.set_aspect("equal")
    ax_pred.set_xticks([]); ax_pred.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure4_leiden_skin(cohort_results, output_path) -> bool:
    if "Skin" not in cohort_results:
        return False
    return _figure_leiden(cohort_results["Skin"], "Skin", output_path)


def figure7_leiden_breast(cohort_results, output_path) -> bool:
    if "Breast" not in cohort_results:
        return False
    return _figure_leiden(cohort_results["Breast"], "Breast", output_path)


def figure8_leiden_lymph(cohort_results, output_path) -> bool:
    if "Lymph" not in cohort_results:
        return False
    return _figure_leiden(cohort_results["Lymph"], "Lymph", output_path)


# --------- Figure 5 / 6 — Correlation matrices ----------
def _figure_correlation(cohort_results, kind: str, output_path: str) -> bool:
    """Pathway-pathway (kind='pathway') or gene-gene (kind='gene') correlation
    heatmap per cohort, GT vs Ours."""
    cohorts = [c for c in ["Breast", "Skin", "Lymph"] if c in cohort_results]
    if not cohorts:
        return False
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(10, 4 * len(cohorts)))
    if len(cohorts) == 1:
        axes = axes.reshape(1, 2)
    for r, c in enumerate(cohorts):
        d = cohort_results[c]
        true = d[f"{kind}_true"]
        pred = d[f"{kind}_pred"]
        # Subsample columns if > 200 to keep the heatmap readable
        if true.shape[1] > 200:
            rng = np.random.default_rng(42)
            idx = rng.choice(true.shape[1], size=200, replace=False)
            idx.sort()
            true = true[:, idx]
            pred = pred[:, idx]
        corr_true = np.corrcoef(true.T)
        corr_pred = np.corrcoef(pred.T)
        im0 = axes[r, 0].imshow(corr_true, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[r, 0].set_title(f"{c}  GT", fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        im1 = axes[r, 1].imshow(corr_pred, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[r, 1].set_title("Ours", fontsize=10)
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        plt.colorbar(im1, ax=axes[r, 1], shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure5_pathway_correlation(cohort_results, output_path) -> bool:
    return _figure_correlation(cohort_results, "pathway", output_path)


def figure6_gene_correlation(cohort_results, output_path) -> bool:
    return _figure_correlation(cohort_results, "gene", output_path)


# --------- Figure 9 — Supplementary pathway biology heatmaps ----------
def figure9_pathway_biology_heatmaps(cohort_results, output_path) -> bool:
    """Fig 9: cohort-specific pathway spatial heatmap (MYC for Breast,
    Eukaryotic_Translation for Skin, ABC_family for Lymph). GT vs Ours."""
    cohorts = [c for c in ["Breast", "Skin", "Lymph"] if c in cohort_results]
    rows = []
    for c in cohorts:
        d = cohort_results[c]
        idx = _find_column(d.get("pathway_names"), SUPPL_FIG9_PATHWAY_NAMES[c])
        if idx is None:
            print(f"  [fig9] {c}: pathway {SUPPL_FIG9_PATHWAY_NAMES[c]} not found, skipping")
            continue
        rows.append((c, d, idx))
    if not rows:
        return False
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, 2)
    for r, (c, d, idx) in enumerate(rows):
        coords = d["coords"]
        gt_v = d["pathway_true"][:, idx]
        pp = d["pathway_pred"][:, idx]
        vmin, vmax = float(min(gt_v.min(), pp.min())), float(max(gt_v.max(), pp.max()))
        sc = _spatial_scatter(axes[r, 0], coords, gt_v,
                              f"{c}  {SUPPL_FIG9_PATHWAY_NAMES[c]}\nGT", vmin, vmax)
        _spatial_scatter(axes[r, 1], coords, pp, "Ours", vmin, vmax)
        plt.colorbar(sc, ax=axes[r, 1], shrink=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# --------- Figure 10 — Supplementary gene biology heatmaps ----------
def figure10_gene_biology_heatmaps(cohort_results, output_path) -> bool:
    """Fig 10: cohort-specific gene spatial heatmap (SSBP1, EIF4EBP1, PKP2)."""
    cohorts = [c for c in ["Breast", "Skin", "Lymph"] if c in cohort_results]
    rows = []
    for c in cohorts:
        d = cohort_results[c]
        idx = _find_column(d.get("gene_names"), SUPPL_FIG10_GENE_NAMES[c])
        if idx is None:
            print(f"  [fig10] {c}: gene {SUPPL_FIG10_GENE_NAMES[c]} not found, skipping")
            continue
        rows.append((c, d, idx))
    if not rows:
        return False
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, 2)
    for r, (c, d, idx) in enumerate(rows):
        coords = d["coords"]
        gt_v = d["gene_true"][:, idx]
        pp = d["gene_pred"][:, idx]
        vmin, vmax = float(min(gt_v.min(), pp.min())), float(max(gt_v.max(), pp.max()))
        sc = _spatial_scatter(axes[r, 0], coords, gt_v,
                              f"{c}  {SUPPL_FIG10_GENE_NAMES[c]}\nGT", vmin, vmax)
        _spatial_scatter(axes[r, 1], coords, pp, "Ours", vmin, vmax)
        plt.colorbar(sc, ax=axes[r, 1], shrink=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# --------- Survival KM curve (right panel of Fig 3) ----------
def figure_survival_km(survival_results: dict, output_path: str) -> bool:
    """Kaplan-Meier curves split by predicted risk (median split)."""
    if not survival_results or "risk_scores" not in survival_results:
        return False
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        print("  [fig_survival_km] lifelines not installed; skipping")
        return False
    risk = np.asarray(survival_results["risk_scores"])
    os_time = np.asarray(survival_results["os_time"])
    event = np.asarray(survival_results["event"])
    median_risk = float(np.median(risk))
    high = risk >= median_risk
    low = ~high
    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(os_time[high], event[high], label=f"High risk (n={high.sum()})")
    kmf.plot_survival_function(ax=ax)
    kmf.fit(os_time[low], event[low], label=f"Low risk (n={low.sum()})")
    kmf.plot_survival_function(ax=ax)
    cindex = survival_results.get("c_index_mean")
    cstd = survival_results.get("c_index_std")
    title = "TCGA-BRCA survival"
    if cindex is not None:
        title += f"  (C-index {cindex:.3f}"
        if cstd is not None:
            title += f" ± {cstd:.3f}"
        title += ")"
    ax.set_title(title); ax.set_xlabel("Days"); ax.set_ylabel("Survival probability")
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True


# --------- Table formatters ----------
def format_tables_1_2(cohort_results: Dict[str, dict],
                      output_path: str) -> dict:
    """Build Table 1 (gene) + Table 2 (pathway) rows from per-cohort metrics.

    Each cohort entry must have keys: gene_pcc_mean, gene_pcc_std, gene_mse_mean,
    gene_mse_std, gene_mae_mean, gene_mae_std, and analogous pathway_* keys.
    These come from aggregate_folds() in reproduction.py.
    """
    tables = {"table1_gene": {}, "table2_pathway": {}}
    for cohort, d in cohort_results.items():
        if "summary" not in d:
            continue
        s = d["summary"]
        gene = s.get("gene", {})
        path = s.get("pathway", {})
        tables["table1_gene"][cohort] = {
            "PCC":  gene.get("PCC"),
            "MSE":  gene.get("MSE"),
            "MAE":  gene.get("MAE"),
        }
        tables["table2_pathway"][cohort] = {
            "PCC":  path.get("PCC"),
            "MSE":  path.get("MSE"),
            "MAE":  path.get("MAE"),
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tables, f, indent=2)
    return tables


def format_table_3(survival_results: Optional[dict], output_path: str) -> dict:
    """Build Table 3 (survival C-index) from survival training output."""
    table = {}
    if survival_results and "c_index_mean" in survival_results:
        table["PEaRL+TabPFN-v3"] = {
            "c_index_mean": survival_results["c_index_mean"],
            "c_index_std":  survival_results.get("c_index_std"),
            "n_folds":      survival_results.get("n_folds", 5),
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(table, f, indent=2)
    return table


# --------- Orchestrator ----------
def generate_paper_figures(cohort_results: Dict[str, dict],
                           output_dir: str,
                           survival_results: Optional[dict] = None) -> dict:
    """Run every figure + table function. Returns a manifest of which files
    were produced. Called from scripts/reproduce_paper.py."""
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    manifest = {}

    figs = [
        ("fig3_spatial_predictions.png",     figure3_spatial_predictions),
        ("fig4_leiden_skin.png",             figure4_leiden_skin),
        ("fig5_pathway_correlation.png",     figure5_pathway_correlation),
        ("fig6_gene_correlation.png",        figure6_gene_correlation),
        ("fig7_leiden_breast.png",           figure7_leiden_breast),
        ("fig8_leiden_lymph.png",            figure8_leiden_lymph),
        ("fig9_pathway_biology.png",         figure9_pathway_biology_heatmaps),
        ("fig10_gene_biology.png",           figure10_gene_biology_heatmaps),
    ]
    for fname, fn in figs:
        out = str(od / fname)
        try:
            ok = fn(cohort_results, out)
            manifest[fname] = "ok" if ok else "skipped (missing data)"
            print(f"  {'✓' if ok else '·'} {fname}")
        except Exception as e:
            manifest[fname] = f"ERROR: {type(e).__name__}: {e}"
            print(f"  ✗ {fname}: {type(e).__name__}: {e}")

    # Survival KM (Fig 3 right panel)
    if survival_results:
        out = str(od / "fig_survival_km.png")
        try:
            ok = figure_survival_km(survival_results, out)
            manifest["fig_survival_km.png"] = "ok" if ok else "skipped"
            print(f"  {'✓' if ok else '·'} fig_survival_km.png")
        except Exception as e:
            manifest["fig_survival_km.png"] = f"ERROR: {e}"

    # Tables
    t12 = format_tables_1_2(cohort_results, str(od / "tables_1_2.json"))
    t3 = format_table_3(survival_results, str(od / "table_3.json"))
    manifest["tables_1_2.json"] = t12
    manifest["table_3.json"] = t3

    with open(od / "manifest.json", "w") as f:
        json.dump({k: (v if not isinstance(v, dict) else "see file") for k, v in manifest.items()},
                  f, indent=2)
    return manifest

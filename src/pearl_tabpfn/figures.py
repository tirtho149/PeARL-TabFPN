"""PEaRL Figure Generation: All 7 figures from paper"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

sns.set_style("whitegrid")


def figure1_model_comparison(
    results_dict: Dict[str, Dict[str, float]], output_path: str = None
):
    """
    Figure 1: Model architecture and comparison with baselines.

    Shows PEaRL framework: pathway encoder, vision encoder, contrastive learning.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results_dict.keys())
    pcc_means = [results_dict[m].get("gene_pcc_mean", 0.0) for m in methods]
    pcc_stds = [results_dict[m].get("gene_pcc_std", 0.0) for m in methods]

    colors = ["#FF6B6B" if m == "PEaRL" else "#4ECDC4" for m in methods]
    bars = ax.bar(range(len(methods)), pcc_means, yerr=pcc_stds, capsize=5, color=colors, alpha=0.8)

    ax.set_ylabel("Gene Expression PCC", fontsize=12, fontweight="bold")
    ax.set_title("Figure 1: Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure2_contrastive_curves(
    history_stage1: Dict[str, list], output_path: str = None
):
    """
    Figure 2: Contrastive pretraining loss curves.

    Shows convergence of image-pathway alignment during Stage 1.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history_stage1["train_loss"]) + 1)
    ax.plot(epochs, history_stage1["train_loss"], "o-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history_stage1["val_loss"], "s-", label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Contrastive Loss", fontsize=12, fontweight="bold")
    ax.set_title("Figure 2: Contrastive Pretraining Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure3_pathway_scatter(
    pathway_pred: np.ndarray,
    pathway_true: np.ndarray,
    output_path: str = None,
):
    """
    Figure 3: Pathway prediction scatter plots.

    Shows correlation between predicted and true pathway scores.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(min(4, pathway_pred.shape[1])):
        ax = axes[i]
        ax.scatter(pathway_true[:, i], pathway_pred[:, i], alpha=0.5, s=20)

        # Add regression line
        z = np.polyfit(pathway_true[:, i], pathway_pred[:, i], 1)
        p = np.poly1d(z)
        x_line = np.linspace(pathway_true[:, i].min(), pathway_true[:, i].max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2)

        # Compute R²
        ss_res = np.sum((pathway_pred[:, i] - p(pathway_true[:, i])) ** 2)
        ss_tot = np.sum((pathway_pred[:, i] - np.mean(pathway_pred[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        ax.set_xlabel("True", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"Pathway {i} (R²={r2:.3f})", fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure4_spatial_heatmap(
    coords: np.ndarray,
    gene_pred: np.ndarray,
    output_path: str = None,
):
    """
    Figure 4: Spatial gene expression heatmaps.

    Shows spatially resolved gene prediction across tissue.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    for i in range(min(4, gene_pred.shape[1])):
        ax = axes[i]

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=gene_pred[:, i],
            cmap="RdYlBu_r",
            s=50,
            alpha=0.8,
        )

        ax.set_aspect("equal")
        ax.set_title(f"Gene {i} Expression", fontweight="bold")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Expression", fontsize=9)

    plt.suptitle("Figure 4: Spatial Gene Expression Maps", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure5_survival_cindex(
    datasets: list,
    survival_cindex: Dict[str, float],
    output_path: str = None,
):
    """
    Figure 5: Survival prediction C-index across datasets.

    Compares prognostic utility via concordance index.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = list(survival_cindex.keys())
    cindex_vals = [survival_cindex[m] for m in methods]

    bars = ax.bar(
        range(len(methods)),
        cindex_vals,
        color="#FF6B6B",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_ylabel("Concordance Index (C-index)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 5: Survival Prediction Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylim([0.5, 0.75])
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure6_gradcam_visualization(
    coords: np.ndarray,
    gradcam_weights: np.ndarray,
    gene_pred: np.ndarray,
    output_path: str = None,
):
    """
    Figure 6: GradCAM attention visualization.

    Shows which image regions contribute to gene predictions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: GradCAM weights
    scatter = axes[0].scatter(
        coords[:, 0], coords[:, 1], c=gradcam_weights, cmap="hot", s=50, alpha=0.8
    )
    axes[0].set_aspect("equal")
    axes[0].set_title("GradCAM Attention Weights", fontweight="bold")
    plt.colorbar(scatter, ax=axes[0], label="Weight")

    # Right: Predicted expression weighted by attention
    weighted_pred = gene_pred[:, 0] * gradcam_weights
    scatter = axes[1].scatter(
        coords[:, 0], coords[:, 1], c=weighted_pred, cmap="RdYlBu_r", s=50, alpha=0.8
    )
    axes[1].set_aspect("equal")
    axes[1].set_title("Expression × Attention", fontweight="bold")
    plt.colorbar(scatter, ax=axes[1], label="Value")

    plt.suptitle("Figure 6: GradCAM Attention Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure7_pathway_attention(
    pathway_scores: np.ndarray,
    output_path: str = None,
):
    """
    Figure 7: Pathway attention heatmap.

    Shows which pathways are most important for predictions.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Compute pathway importance (variance across samples)
    pathway_importance = np.var(pathway_scores, axis=0)
    top_pathways = np.argsort(pathway_importance)[-20:]

    # Plot heatmap of top pathways
    data = pathway_scores[:, top_pathways].T
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto")

    ax.set_ylabel("Pathways", fontsize=12, fontweight="bold")
    ax.set_xlabel("Samples", fontsize=12, fontweight="bold")
    ax.set_title("Figure 7: Top 20 Pathway Attention", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score", fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_figures(
    results_dict: Dict,
    history_stage1: Dict,
    coords: np.ndarray,
    gene_pred: np.ndarray,
    pathway_pred: np.ndarray,
    gene_true: np.ndarray,
    pathway_true: np.ndarray,
    survival_cindex: Dict,
    output_dir: str = "./pearl_outputs",
):
    """Generate all 7 figures (single-model legacy mode).

    For head-to-head (PEaRL+MLP vs PEaRL+TabPFN vs PEaRL paper) use
    `generate_head_to_head_figures(...)` instead — that's the canonical
    BIBM 2026 figure set.
    """
    print("Generating figures...")

    figure1_model_comparison(
        results_dict, output_path=f"{output_dir}/fig1_model_comparison.png"
    )
    print("✓ Figure 1: Model Comparison")

    figure2_contrastive_curves(
        history_stage1, output_path=f"{output_dir}/fig2_contrastive_curves.png"
    )
    print("✓ Figure 2: Contrastive Curves")

    figure3_pathway_scatter(
        pathway_pred, pathway_true, output_path=f"{output_dir}/fig3_pathway_scatter.png"
    )
    print("✓ Figure 3: Pathway Scatter")

    figure4_spatial_heatmap(
        coords, gene_pred, output_path=f"{output_dir}/fig4_spatial_heatmap.png"
    )
    print("✓ Figure 4: Spatial Heatmap")

    figure5_survival_cindex(
        ["Breast", "Skin", "Lymph"],
        survival_cindex,
        output_path=f"{output_dir}/fig5_survival_cindex.png",
    )
    print("✓ Figure 5: Survival C-index")

    # Figure 6 (GradCAM) was a random-noise placeholder — removed to prevent
    # synthetic content being rendered as if it were a real interpretability
    # result. If you implement real GradCAM, call figure6_gradcam_visualization
    # directly with the actual attention weights.
    print("⏭ Figure 6: GradCAM (skipped — no real attention weights supplied)")

    figure7_pathway_attention(
        pathway_pred, output_path=f"{output_dir}/fig7_pathway_attention.png"
    )
    print("✓ Figure 7: Pathway Attention")

    print(f"\nAll figures saved to {output_dir}/")


# ============================================================================
# HEAD-TO-HEAD VISUALS (BIBM 2026 canonical figure set)
#
# Every figure renders three columns / series side-by-side:
#   (1) PEaRL + MLP  (ours, baseline)
#   (2) PEaRL + TabPFN (ours, MLP replaced 1:1 with TabPFN bank)
#   (3) PEaRL paper (Majumder et al. 2025, reference numbers)
#
# Input shape is identical to the single-model functions, with arrays for
# baseline and tabpfn passed in parallel.
# ============================================================================


# PEaRL paper Table 1 + Table 2 + Table 3 reference numbers (Breast cohort).
PAPER_BASELINE_BREAST = {
    "gene":    {"PCC": (0.5868, 0.0359), "MSE": (0.0732, 0.0033), "MAE": (0.1828, 0.0043)},
    "pathway": {"PCC": (0.5055, 0.0271), "MSE": (0.0017, 0.0001), "MAE": (0.0314, 0.0010)},
    "survival_c_index": (0.659, 0.027),
}


def figure_h2h_metric_bars(
    summary_baseline: Dict,
    summary_tabpfn: Dict,
    paper_reference: Dict = None,
    output_path: str = None,
):
    """Side-by-side bars: PCC / MSE / MAE for gene and pathway across the
    three approaches. summary_* dicts come from `aggregate_folds` and look like
    {target: {metric: (mean, std), ...}}.
    """
    paper_reference = paper_reference or PAPER_BASELINE_BREAST
    targets = ("gene", "pathway")
    metrics = ("PCC", "MSE", "MAE")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    methods = ["PEaRL+MLP\n(ours)", "PEaRL+TabPFN\n(ours)", "PEaRL\n(paper)"]
    colors = ["#1f77b4", "#d62728", "#7f7f7f"]
    for ri, target in enumerate(targets):
        for ci, metric in enumerate(metrics):
            ax = axes[ri, ci]
            bm, bs = summary_baseline.get(target, {}).get(metric, (np.nan, 0))
            tm, ts = summary_tabpfn.get(target, {}).get(metric, (np.nan, 0))
            pm, ps = paper_reference[target][metric]
            ax.bar(methods, [bm, tm, pm], yerr=[bs, ts, ps], capsize=5,
                   color=colors, alpha=0.85, edgecolor="black")
            arrow = "↑" if metric == "PCC" else "↓"
            ax.set_title(f"{target.title()} {metric} {arrow}", fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            for x, v in enumerate([bm, tm, pm]):
                if not np.isnan(v):
                    ax.text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle(
        "Head-to-Head: PEaRL+MLP vs PEaRL+TabPFN vs PEaRL paper",
        fontsize=14, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_h2h_pathway_scatter(
    pathway_pred_mlp: np.ndarray,
    pathway_pred_tabpfn: np.ndarray,
    pathway_true: np.ndarray,
    output_path: str = None,
    top_k: int = 4,
):
    """Pathway prediction scatter — MLP head (left col) vs TabPFN head (right col)
    on the same top-k highest-variance pathways."""
    n = min(top_k, pathway_pred_mlp.shape[1])
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    var_rank = np.argsort(pathway_true.var(axis=0))[-n:][::-1]
    for r, dim in enumerate(var_rank):
        for c, (pred, title) in enumerate(
            ((pathway_pred_mlp, "PEaRL+MLP"), (pathway_pred_tabpfn, "PEaRL+TabPFN"))
        ):
            ax = axes[r, c]
            ax.scatter(pathway_true[:, dim], pred[:, dim], alpha=0.5, s=15)
            mn = min(pathway_true[:, dim].min(), pred[:, dim].min())
            mx = max(pathway_true[:, dim].max(), pred[:, dim].max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, alpha=0.5)
            from scipy.stats import pearsonr
            r_, _ = pearsonr(pathway_true[:, dim], pred[:, dim])
            ax.set_xlabel("True"); ax.set_ylabel("Predicted")
            ax.set_title(f"{title} — Pathway {dim} (PCC={r_:.3f})", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_h2h_gene_scatter(
    gene_pred_mlp: np.ndarray,
    gene_pred_tabpfn: np.ndarray,
    gene_true: np.ndarray,
    output_path: str = None,
    top_k: int = 4,
):
    """Gene prediction scatter — MLP vs TabPFN on top-k highest-variance genes."""
    n = min(top_k, gene_pred_mlp.shape[1])
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    var_rank = np.argsort(gene_true.var(axis=0))[-n:][::-1]
    for r, dim in enumerate(var_rank):
        for c, (pred, title) in enumerate(
            ((gene_pred_mlp, "PEaRL+MLP"), (gene_pred_tabpfn, "PEaRL+TabPFN"))
        ):
            ax = axes[r, c]
            ax.scatter(gene_true[:, dim], pred[:, dim], alpha=0.5, s=15)
            mn = min(gene_true[:, dim].min(), pred[:, dim].min())
            mx = max(gene_true[:, dim].max(), pred[:, dim].max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, alpha=0.5)
            from scipy.stats import pearsonr
            r_, _ = pearsonr(gene_true[:, dim], pred[:, dim])
            ax.set_xlabel("True"); ax.set_ylabel("Predicted")
            ax.set_title(f"{title} — Gene {dim} (PCC={r_:.3f})", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_h2h_spatial_heatmap(
    coords: np.ndarray,
    gene_pred_mlp: np.ndarray,
    gene_pred_tabpfn: np.ndarray,
    gene_true: np.ndarray,
    output_path: str = None,
    top_k: int = 3,
):
    """Spatial heatmap triplet: GT / MLP / TabPFN per top-k gene."""
    n = min(top_k, gene_pred_mlp.shape[1])
    var_rank = np.argsort(gene_true.var(axis=0))[-n:][::-1]
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for r, dim in enumerate(var_rank):
        for c, (arr, title) in enumerate(
            (
                (gene_true[:, dim], "Ground Truth"),
                (gene_pred_mlp[:, dim], "PEaRL+MLP (ours)"),
                (gene_pred_tabpfn[:, dim], "PEaRL+TabPFN (ours)"),
            )
        ):
            ax = axes[r, c]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=arr, cmap="RdYlBu_r", s=20, alpha=0.85)
            ax.set_aspect("equal")
            ax.set_title(f"{title} — Gene {dim}", fontweight="bold", fontsize=10)
            plt.colorbar(sc, ax=ax, shrink=0.7)
    plt.suptitle("Head-to-Head Spatial Gene Maps", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_h2h_pathway_heatmap(
    coords: np.ndarray,
    pathway_pred_mlp: np.ndarray,
    pathway_pred_tabpfn: np.ndarray,
    pathway_true: np.ndarray,
    output_path: str = None,
    top_k: int = 3,
):
    """Spatial heatmap triplet for pathways (mirrors gene version)."""
    n = min(top_k, pathway_pred_mlp.shape[1])
    var_rank = np.argsort(pathway_true.var(axis=0))[-n:][::-1]
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for r, dim in enumerate(var_rank):
        for c, (arr, title) in enumerate(
            (
                (pathway_true[:, dim], "Ground Truth"),
                (pathway_pred_mlp[:, dim], "PEaRL+MLP (ours)"),
                (pathway_pred_tabpfn[:, dim], "PEaRL+TabPFN (ours)"),
            )
        ):
            ax = axes[r, c]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=arr, cmap="RdBu_r", s=20, alpha=0.85)
            ax.set_aspect("equal")
            ax.set_title(f"{title} — Pathway {dim}", fontweight="bold", fontsize=10)
            plt.colorbar(sc, ax=ax, shrink=0.7)
    plt.suptitle("Head-to-Head Spatial Pathway Maps", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_h2h_per_dim_pcc(
    pathway_pred_mlp: np.ndarray,
    pathway_pred_tabpfn: np.ndarray,
    pathway_true: np.ndarray,
    gene_pred_mlp: np.ndarray,
    gene_pred_tabpfn: np.ndarray,
    gene_true: np.ndarray,
    output_path: str = None,
):
    """Per-dim PCC distribution — KDE/histogram for MLP vs TabPFN. Shows
    whether TabPFN broadens or narrows the per-dim PCC distribution."""
    from scipy.stats import pearsonr

    def per_dim_pcc(pred, true):
        out = []
        for d in range(pred.shape[1]):
            if pred[:, d].std() < 1e-12 or true[:, d].std() < 1e-12:
                continue
            r, _ = pearsonr(pred[:, d], true[:, d])
            if not np.isnan(r):
                out.append(r)
        return np.array(out)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (mlp, tab, true, title) in zip(
        axes,
        (
            (gene_pred_mlp, gene_pred_tabpfn, gene_true, "Gene per-dim PCC"),
            (pathway_pred_mlp, pathway_pred_tabpfn, pathway_true, "Pathway per-dim PCC"),
        ),
    ):
        pcc_mlp = per_dim_pcc(mlp, true)
        pcc_tab = per_dim_pcc(tab, true)
        ax.hist(pcc_mlp, bins=40, alpha=0.5, label=f"PEaRL+MLP (μ={pcc_mlp.mean():.3f})", color="#1f77b4")
        ax.hist(pcc_tab, bins=40, alpha=0.5, label=f"PEaRL+TabPFN (μ={pcc_tab.mean():.3f})", color="#d62728")
        ax.axvline(pcc_mlp.mean(), color="#1f77b4", linestyle="--", linewidth=1)
        ax.axvline(pcc_tab.mean(), color="#d62728", linestyle="--", linewidth=1)
        ax.set_xlabel("PCC"); ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_h2h_correlation_matrices(
    pathway_pred_mlp: np.ndarray,
    pathway_pred_tabpfn: np.ndarray,
    pathway_true: np.ndarray,
    output_path: str = None,
    max_dims: int = 50,
):
    """Pathway-pathway correlation matrices: GT, MLP, TabPFN — diagnostic for
    whether predictions preserve the co-expression structure of the targets."""
    k = min(max_dims, pathway_true.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (data, title) in zip(
        axes,
        (
            (pathway_true[:, :k], "GT"),
            (pathway_pred_mlp[:, :k], "PEaRL+MLP"),
            (pathway_pred_tabpfn[:, :k], "PEaRL+TabPFN"),
        ),
    ):
        if data.std(axis=0).min() < 1e-12:
            keep = data.std(axis=0) > 1e-12
            data = data[:, keep]
        corr = np.corrcoef(data.T)
        sns.heatmap(corr, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, cbar=True,
                    square=True, xticklabels=False, yticklabels=False)
        ax.set_title(title, fontweight="bold")
    plt.suptitle("Pathway-Pathway Correlation Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_head_to_head_figures(
    summary_baseline: Dict,
    summary_tabpfn: Dict,
    coords: np.ndarray,
    pathway_pred_mlp: np.ndarray,
    pathway_pred_tabpfn: np.ndarray,
    pathway_true: np.ndarray,
    gene_pred_mlp: np.ndarray,
    gene_pred_tabpfn: np.ndarray,
    gene_true: np.ndarray,
    paper_reference: Dict = None,
    output_dir: str = "./pearl_outputs",
):
    """Generate the BIBM 2026 head-to-head figure set.

    Each figure compares PEaRL+MLP (ours) vs PEaRL+TabPFN (ours) vs the PEaRL
    paper's reported numbers, on the same fold's val predictions.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    print("Generating head-to-head figures...")

    figure_h2h_metric_bars(
        summary_baseline, summary_tabpfn, paper_reference,
        output_path=f"{output_dir}/fig_h2h_1_metric_bars.png",
    )
    print("✓ H2H Fig 1: Metric bars (gene + pathway × PCC/MSE/MAE)")

    figure_h2h_pathway_scatter(
        pathway_pred_mlp, pathway_pred_tabpfn, pathway_true,
        output_path=f"{output_dir}/fig_h2h_2_pathway_scatter.png",
    )
    print("✓ H2H Fig 2: Pathway scatter")

    figure_h2h_gene_scatter(
        gene_pred_mlp, gene_pred_tabpfn, gene_true,
        output_path=f"{output_dir}/fig_h2h_3_gene_scatter.png",
    )
    print("✓ H2H Fig 3: Gene scatter")

    figure_h2h_spatial_heatmap(
        coords, gene_pred_mlp, gene_pred_tabpfn, gene_true,
        output_path=f"{output_dir}/fig_h2h_4_gene_spatial.png",
    )
    print("✓ H2H Fig 4: Gene spatial heatmap")

    figure_h2h_pathway_heatmap(
        coords, pathway_pred_mlp, pathway_pred_tabpfn, pathway_true,
        output_path=f"{output_dir}/fig_h2h_5_pathway_spatial.png",
    )
    print("✓ H2H Fig 5: Pathway spatial heatmap")

    figure_h2h_per_dim_pcc(
        pathway_pred_mlp, pathway_pred_tabpfn, pathway_true,
        gene_pred_mlp, gene_pred_tabpfn, gene_true,
        output_path=f"{output_dir}/fig_h2h_6_per_dim_pcc.png",
    )
    print("✓ H2H Fig 6: Per-dim PCC distributions")

    figure_h2h_correlation_matrices(
        pathway_pred_mlp, pathway_pred_tabpfn, pathway_true,
        output_path=f"{output_dir}/fig_h2h_7_pathway_corr.png",
    )
    print("✓ H2H Fig 7: Pathway-pathway correlation matrices")

    print(f"\nAll head-to-head figures saved to {output_dir}/")

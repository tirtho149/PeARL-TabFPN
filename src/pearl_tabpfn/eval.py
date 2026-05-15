"""PEaRL Evaluation: Metrics & Visualizations"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

from .baseline import PEaRL


class PEaRLEvaluator:
    """Evaluation metrics for PEaRL."""

    @staticmethod
    def predict_batch(
        model: PEaRL,
        patches: torch.Tensor,
        pathways: torch.Tensor,
        coords: torch.Tensor,
        genes: torch.Tensor,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions for a batch.

        Returns:
            pathway_pred, gene_pred, pathway_true, gene_true (all numpy)
        """
        model.eval()

        with torch.no_grad():
            patches = patches.to(device)
            pathway_pred, gene_pred = model.forward_supervised(patches)

        return (
            pathway_pred.cpu().numpy(),
            gene_pred.cpu().numpy(),
            pathways.numpy(),
            genes.numpy(),
        )

    @staticmethod
    def pcc(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
        """Pearson Correlation Coefficient per feature."""
        pcc_vals = []
        for i in range(pred.shape[1]):
            if pred.shape[0] > 1:
                r, _ = pearsonr(pred[:, i], true[:, i])
                pcc_vals.append(r)
            else:
                pcc_vals.append(0.0)
        return np.array(pcc_vals)

    @staticmethod
    def mse(pred: np.ndarray, true: np.ndarray) -> float:
        return mean_squared_error(true, pred)

    @staticmethod
    def mae(pred: np.ndarray, true: np.ndarray) -> float:
        return mean_absolute_error(true, pred)

    @staticmethod
    def evaluate_expression_prediction(
        pathway_pred: np.ndarray,
        pathway_true: np.ndarray,
        gene_pred: np.ndarray,
        gene_true: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics for gene and pathway prediction."""
        metrics = {}

        # Pathway metrics
        pathway_pcc = PEaRLEvaluator.pcc(pathway_pred, pathway_true)
        metrics["pathway_pcc_mean"] = np.mean(pathway_pcc)
        metrics["pathway_pcc_std"] = np.std(pathway_pcc)
        metrics["pathway_mse"] = PEaRLEvaluator.mse(pathway_pred, pathway_true)
        metrics["pathway_mae"] = PEaRLEvaluator.mae(pathway_pred, pathway_true)

        # Gene metrics
        gene_pcc = PEaRLEvaluator.pcc(gene_pred, gene_true)
        metrics["gene_pcc_mean"] = np.mean(gene_pcc)
        metrics["gene_pcc_std"] = np.std(gene_pcc)
        metrics["gene_mse"] = PEaRLEvaluator.mse(gene_pred, gene_true)
        metrics["gene_mae"] = PEaRLEvaluator.mae(gene_pred, gene_true)

        return metrics

    @staticmethod
    def leiden_clustering(
        pred_pathway: np.ndarray,
        true_gene: np.ndarray,
        coords: np.ndarray,
        n_clusters: int = 5,
    ) -> Dict[str, float]:
        """
        Leiden clustering on predicted expression.
        Measure agreement with ground truth clusters via ARI.
        """
        # Cluster predicted pathways using spectral clustering
        clusterer = SpectralClustering(
            n_clusters=n_clusters, affinity="nearest_neighbors", random_state=42
        )
        pred_clusters = clusterer.fit_predict(pred_pathway)

        # Ground truth clusters from gene expression (simple k-means)
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        true_clusters = kmeans.fit_predict(true_gene)

        # ARI
        ari = adjusted_rand_score(true_clusters, pred_clusters)

        return {"ari": ari, "pred_clusters": pred_clusters, "true_clusters": true_clusters}


def visualize_spatial_prediction(
    coords: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    feature_names: list = None,
    title: str = "Spatial Prediction",
    output_path: str = None,
    vmin: float = None,
    vmax: float = None,
):
    """Visualize predicted vs true values on spatial coordinates."""
    n_features = min(4, pred.shape[1])  # Show up to 4 features

    fig, axes = plt.subplots(2, n_features, figsize=(4 * n_features, 8))
    if n_features == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_features):
        fname = feature_names[i] if feature_names else f"Feature {i}"

        # Predicted
        ax = axes[0, i]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=pred[:, i], cmap="RdBu_r", s=10)
        ax.set_title(f"Pred: {fname}")
        ax.set_aspect("equal")
        plt.colorbar(scatter, ax=ax)

        # True
        ax = axes[1, i]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=true[:, i], cmap="RdBu_r", s=10)
        ax.set_title(f"True: {fname}")
        ax.set_aspect("equal")
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_clusters(
    coords: np.ndarray,
    pred_clusters: np.ndarray,
    true_clusters: np.ndarray,
    output_path: str = None,
    title: str = "Leiden Clustering",
):
    """Visualize predicted vs true cluster assignments."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Predicted
    scatter = axes[0].scatter(
        coords[:, 0], coords[:, 1], c=pred_clusters, cmap="tab20", s=20
    )
    axes[0].set_title("Predicted Clusters")
    axes[0].set_aspect("equal")
    plt.colorbar(scatter, ax=axes[0])

    # True
    scatter = axes[1].scatter(coords[:, 0], coords[:, 1], c=true_clusters, cmap="tab20", s=20)
    axes[1].set_title("Ground Truth Clusters")
    axes[1].set_aspect("equal")
    plt.colorbar(scatter, ax=axes[1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_matrix(
    pred: np.ndarray,
    true: np.ndarray,
    output_path: str = None,
    title: str = "Prediction Correlation",
):
    """Plot correlation matrices for predicted vs true."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Predicted correlation
    pred_corr = np.corrcoef(pred.T)
    sns.heatmap(pred_corr, ax=axes[0], cmap="RdBu_r", vmin=-1, vmax=1, cbar=True)
    axes[0].set_title("Predicted Correlation")

    # True correlation
    true_corr = np.corrcoef(true.T)
    sns.heatmap(true_corr, ax=axes[1], cmap="RdBu_r", vmin=-1, vmax=1, cbar=True)
    axes[1].set_title("Ground Truth Correlation")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: str = None,
    title: str = "Method Comparison",
):
    """Compare metrics across methods."""
    methods = list(metrics_dict.keys())
    pcc_means = [metrics_dict[m]["gene_pcc_mean"] for m in methods]
    pcc_stds = [metrics_dict[m]["gene_pcc_std"] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    bars = ax.bar(x, pcc_means, yerr=pcc_stds, capsize=5, alpha=0.7)

    ax.set_xlabel("Method")
    ax.set_ylabel("Gene Expression PCC")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history_stage1: Dict[str, list],
    history_stage2: Dict[str, list],
    output_path: str = None,
):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Stage 1
    axes[0].plot(history_stage1["train_loss"], label="Train")
    axes[0].plot(history_stage1["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Contrastive Loss")
    axes[0].set_title("Stage 1: Contrastive Pretraining")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Stage 2
    axes[1].plot(history_stage2["train_loss"], label="Train")
    axes[1].plot(history_stage2["val_loss"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Supervised Loss")
    axes[1].set_title("Stage 2: Supervised Fine-tuning")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_metrics(predictions, targets, drop_constant_cols: bool = True):
    """
    Compute evaluation metrics for predictions.

    Args:
        predictions: numpy array (N, n_features)
        targets:     numpy array (N, n_features)
        drop_constant_cols: drop target columns with zero variance before
            flattening for PCC. Without this, all-zero targets pair with
            small-magnitude predictions to trivially "correlate" via the
            global flatten, inflating reported PCC.

    Returns:
        dict with PCC, MSE, MAE, plus n_cols_used and n_cols_dropped.
    """
    from scipy.stats import pearsonr
    import numpy as np

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    n_total = targets.shape[1] if targets.ndim > 1 else 1

    if drop_constant_cols and targets.ndim > 1:
        keep = targets.std(axis=0) > 1e-8
        n_kept = int(keep.sum())
        if n_kept == 0:
            return {
                "PCC": float("nan"),
                "MSE": float(np.mean((predictions - targets) ** 2)),
                "MAE": float(np.mean(np.abs(predictions - targets))),
                "n_cols_used": 0,
                "n_cols_dropped": n_total,
            }
        predictions = predictions[:, keep]
        targets = targets[:, keep]
    else:
        n_kept = n_total

    pred_flat = predictions.flatten()
    true_flat = targets.flatten()

    # pearsonr also returns NaN if either side has zero variance overall —
    # safer to wrap.
    if np.std(pred_flat) < 1e-12 or np.std(true_flat) < 1e-12:
        pcc = float("nan")
    else:
        pcc, _ = pearsonr(pred_flat, true_flat)

    # Per-dim mean PCC — what spatial-transcriptomics papers usually report
    # (paper PEaRL included). Less dominated by scale/normalization than the
    # global flatten PCC, and better at surfacing gains on a subset of dims.
    if predictions.ndim > 1 and predictions.shape[1] > 0:
        per_dim = []
        for d in range(predictions.shape[1]):
            p = predictions[:, d]; t = targets[:, d]
            if np.std(p) < 1e-12 or np.std(t) < 1e-12:
                continue
            r, _ = pearsonr(p, t)
            if not np.isnan(r):
                per_dim.append(r)
        pcc_per_dim = float(np.mean(per_dim)) if per_dim else float("nan")
        pcc_per_dim_n = len(per_dim)
    else:
        pcc_per_dim = float("nan")
        pcc_per_dim_n = 0

    return {
        "PCC": float(pcc),
        "PCC_per_dim_mean": pcc_per_dim,
        "PCC_per_dim_n": pcc_per_dim_n,
        "MSE": float(np.mean((predictions - targets) ** 2)),
        "MAE": float(np.mean(np.abs(predictions - targets))),
        "n_cols_used": n_kept,
        "n_cols_dropped": n_total - n_kept,
    }

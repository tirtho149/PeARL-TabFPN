"""Pathway-activation analysis for TabPFN-3 (WACV Phase 4).

The honest answer to "which pathway is the model picking up." Uses fold
predictions (no model internals), comparing the predicted spatial
activation pattern to the ssGSEA ground-truth pattern *on the same
spots*. The headline number is the per-pathway *spatial* PCC; the
qualitative companion is a side-by-side activation map.

This module operates on the `.npz` fold-prediction artifacts produced
by `reproduction.py`. Required keys (per fold):

  - `pathway_pred`  (N, P)   predicted ssGSEA-scale activations
  - `pathway_true`  (N, P)   ground-truth ssGSEA activations
  - `coords`        (N, 2)   per-spot spatial coordinates (used by the
                             optional spatial-map renderer; not needed
                             for the spatial-PCC metric itself)

The renderer is split into a separate function so the headline metric
remains compute-free of matplotlib — useful when the figure step runs
on a node without a display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PathwayActivationRank:
    """Output of `rank_pathways_by_spatial_pcc`."""

    pathway_index: np.ndarray  # original column indices, sorted by PCC desc
    spatial_pcc: np.ndarray  # the PCC value at each index, same order
    pathway_name: Optional[List[str]] = None  # mirrors `pathway_index` if names provided


def pathway_spatial_pcc(
    pred: np.ndarray, true: np.ndarray
) -> np.ndarray:
    """Per-pathway *spatial* PCC between predicted and true activation.

    For each pathway column p, compute the Pearson correlation between
    the predicted activation vector `pred[:, p]` and the ground-truth
    `true[:, p]` *across spots*. This is the right comparison for "did
    the model recover the spatial pattern of pathway p"; it is *not*
    the same as the headline `compute_metrics` flatten PCC.

    Returns
    -------
    np.ndarray of shape (P,). Constant-target columns (zero variance)
    return NaN — they are excluded from the ranking by
    `rank_pathways_by_spatial_pcc`.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape}, true {t.shape}")
    n, P = p.shape
    out = np.full(P, np.nan, dtype=np.float64)
    for j in range(P):
        x = p[:, j]
        y = t[:, j]
        sx = x.std()
        sy = y.std()
        if sx == 0.0 or sy == 0.0:
            continue
        out[j] = float(np.corrcoef(x, y)[0, 1])
    return out


def rank_pathways_by_spatial_pcc(
    pred: np.ndarray,
    true: np.ndarray,
    pathway_names: Optional[Sequence[str]] = None,
) -> PathwayActivationRank:
    """Rank pathways by `pathway_spatial_pcc` (descending).

    NaN PCC values (constant columns) are pushed to the end. Phase 4
    reports the top-K (typically K=20) with names; the full ranking is
    saved as JSON for the appendix.
    """
    pcc = pathway_spatial_pcc(pred, true)
    # Stable sort with NaN last.
    # np.argsort puts NaN at the end with kind='mergesort' on most numpy
    # builds, but we be explicit to avoid surprises.
    mask_nan = np.isnan(pcc)
    valid = np.where(~mask_nan)[0]
    invalid = np.where(mask_nan)[0]
    order_valid = valid[np.argsort(-pcc[valid], kind="mergesort")]
    order = np.concatenate([order_valid, invalid])
    sorted_pcc = pcc[order]
    names_out: Optional[List[str]] = None
    if pathway_names is not None:
        if len(pathway_names) != pcc.size:
            raise ValueError(
                f"pathway_names len {len(pathway_names)} does not match "
                f"P={pcc.size}"
            )
        names_out = [pathway_names[i] for i in order.tolist()]
    return PathwayActivationRank(
        pathway_index=order.astype(np.int64),
        spatial_pcc=sorted_pcc,
        pathway_name=names_out,
    )


def confident_active_mask(
    pred: np.ndarray, std: np.ndarray, threshold_pred: float, max_std: float
) -> np.ndarray:
    """Boolean mask of "predicted active AND model confident" spots.

    Used to overlay confidence on the spatial activation map. A spot is
    flagged when:
      - `pred[s, p] >= threshold_pred` (above activation threshold), and
      - `std[s, p]  <= max_std`        (within the confidence band).

    Both thresholds are read from the Phase-4 config (the protocol
    suggests data-driven cutoffs: threshold_pred = 80th percentile of
    `pred[:, p]`, max_std = 20th percentile of `std[:, p]`).
    """
    pred = np.asarray(pred, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if pred.shape != std.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, std {std.shape}")
    return (pred >= threshold_pred) & (std <= max_std)


def summary_for_fold(
    pred: np.ndarray,
    true: np.ndarray,
    pathway_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
) -> Dict[str, object]:
    """Phase-4 per-fold summary dict, JSON-serializable."""
    rank = rank_pathways_by_spatial_pcc(pred, true, pathway_names=pathway_names)
    valid_mask = ~np.isnan(rank.spatial_pcc)
    n_valid = int(valid_mask.sum())
    mean_pcc = (
        float(rank.spatial_pcc[valid_mask].mean()) if n_valid > 0 else float("nan")
    )
    median_pcc = (
        float(np.median(rank.spatial_pcc[valid_mask])) if n_valid > 0 else float("nan")
    )
    top = rank.pathway_index[:top_k].tolist()
    top_names = (
        rank.pathway_name[:top_k] if rank.pathway_name is not None else None
    )
    return {
        "n_pathways": int(rank.pathway_index.size),
        "n_valid": n_valid,
        "mean_spatial_pcc": mean_pcc,
        "median_spatial_pcc": median_pcc,
        "ranking_top_k": top,
        "ranking_top_k_names": top_names,
        "ranking_top_k_pcc": rank.spatial_pcc[:top_k].tolist(),
    }

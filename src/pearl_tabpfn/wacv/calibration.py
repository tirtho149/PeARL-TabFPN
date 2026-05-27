"""Calibration / uncertainty analysis for TabPFN-3 (WACV Phase 3).

The WACV characterization paper's headline non-accuracy contribution is
that TabPFN-3 returns a predictive posterior and that posterior is
*trustworthy*. This module quantifies trustworthiness:

  - `reliability_bins`     — predicted vs empirical interval coverage
  - `expected_calibration_error` — scalar ECE summary
  - `selective_prediction` — error vs coverage as the most-uncertain spots
                             are discarded (the "money plot")

All three operate on the per-fold `.npz` artifacts emitted by
`reproduction.py` when `--save-posteriors` is set. Required keys:

  - `pathway_pred` / `gene_pred`    (N, D)  predicted point estimates
  - `pathway_true` / `gene_true`    (N, D)  ground-truth targets
  - `pathway_std`  / `gene_std`     (N, D)  predictive std (v3 only)

The PEaRL+MLP baseline has no `*_std`, so calibration metrics are only
meaningful for the TabPFN-3 combination. Phase 3 also computes the
*conformal-style* anchor for the MLP baseline (residual std on a holdout
fold) — that lives next to the v3 numbers in Table 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class ReliabilityBins:
    """Output of `reliability_bins`."""

    # Each entry is one nominal coverage level (e.g. 0.5, 0.6, ..., 0.95).
    nominal: np.ndarray
    # Empirical fraction of true values inside the predicted interval.
    empirical: np.ndarray
    # Number of (spot × dim) cells contributing at each level (constant).
    n_cells: int


def reliability_bins(
    pred: np.ndarray,
    true: np.ndarray,
    std: np.ndarray,
    nominal_levels: Optional[np.ndarray] = None,
) -> ReliabilityBins:
    """Predicted vs empirical interval coverage across nominal levels.

    For a Gaussian posterior with mean `pred` and std `std`, the
    `nominal`-coverage interval at each cell is
    `pred ± z(nominal) * std`. Empirical coverage at each level is the
    fraction of cells where `true` actually falls inside that interval.

    A well-calibrated model has empirical ≈ nominal; deviations are read
    off the reliability diagram.

    Notes
    -----
    - Operates on flattened (N × D,) views — all output dims are pooled.
      A per-dim variant lives in the protocol notes; this one is the
      headline reliability plot.
    - Cells with `std <= 0` are treated as a point prediction and are
      counted as covered iff `pred == true` (rare in practice).
    """
    if nominal_levels is None:
        nominal_levels = np.array(
            [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64
        )
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    s = np.asarray(std, dtype=np.float64).reshape(-1)
    if not (p.shape == t.shape == s.shape):
        raise ValueError(
            f"shape mismatch: pred {p.shape}, true {t.shape}, std {s.shape}"
        )
    emp = np.empty_like(nominal_levels)
    for i, lvl in enumerate(nominal_levels):
        # Two-sided Gaussian interval: pred ± z * std where
        # z = Φ^{-1}((1 + lvl) / 2).
        z = norm.ppf((1.0 + lvl) / 2.0)
        lo = p - z * s
        hi = p + z * s
        emp[i] = float(np.mean((t >= lo) & (t <= hi)))
    return ReliabilityBins(
        nominal=nominal_levels.astype(np.float64),
        empirical=emp,
        n_cells=p.size,
    )


def expected_calibration_error(
    pred: np.ndarray,
    true: np.ndarray,
    std: np.ndarray,
    nominal_levels: Optional[np.ndarray] = None,
) -> float:
    """ECE — mean absolute gap between nominal and empirical coverage.

    Conventional definition for regression: ECE = mean_l |emp(l) - l|
    over the `nominal_levels` grid. A perfectly calibrated model has
    ECE = 0. Reported as a scalar per cohort in Table 2.
    """
    rb = reliability_bins(pred, true, std, nominal_levels=nominal_levels)
    return float(np.mean(np.abs(rb.empirical - rb.nominal)))


@dataclass
class SelectivePrediction:
    """Output of `selective_prediction`."""

    coverage: np.ndarray  # fraction of spots retained, in [0, 1]
    error: np.ndarray  # mean per-spot error at that coverage
    # `error_metric` ∈ {"mse", "mae"} — recorded so the plot caption is
    # unambiguous.
    error_metric: str


def selective_prediction(
    pred: np.ndarray,
    true: np.ndarray,
    std: np.ndarray,
    error_metric: str = "mse",
    n_steps: int = 21,
) -> SelectivePrediction:
    """Error vs coverage as the most-uncertain spots are discarded.

    Procedure
    ---------
    1. For each spot, compute a per-spot scalar uncertainty as the mean
       predictive std across output dims.
    2. Sort spots ascending by uncertainty.
    3. Sweep coverage ∈ {1.0, 1.0 - 1/n_steps, ..., 1/n_steps}; at each
       coverage retain that fraction of *most-confident* spots, compute
       the mean per-spot error, and emit a (coverage, error) point.

    A downward-trending curve means uncertainty is informative: throwing
    away the model's least-confident spots reduces error. A flat curve
    means uncertainty is uninformative even if ECE is small.

    The "area under the selective curve" is the natural single-number
    summary; Phase 3 figures plot the curve and Table 2 reports the AUC.
    """
    if error_metric not in ("mse", "mae"):
        raise ValueError("error_metric must be one of 'mse', 'mae'")
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    if not (p.shape == t.shape == s.shape):
        raise ValueError(
            f"shape mismatch: pred {p.shape}, true {t.shape}, std {s.shape}"
        )
    # Per-spot uncertainty: mean predictive std across dims.
    per_spot_unc = s.mean(axis=1)
    order = np.argsort(per_spot_unc, kind="mergesort")
    p_sorted, t_sorted = p[order], t[order]
    n = p.shape[0]

    coverages = np.linspace(1.0, 1.0 / n_steps, n_steps)
    errors = np.empty_like(coverages)
    for i, cov in enumerate(coverages):
        k = max(1, int(np.floor(cov * n)))
        diff = p_sorted[:k] - t_sorted[:k]
        if error_metric == "mse":
            errors[i] = float(np.mean(diff * diff))
        else:
            errors[i] = float(np.mean(np.abs(diff)))
    return SelectivePrediction(
        coverage=coverages.astype(np.float64),
        error=errors.astype(np.float64),
        error_metric=error_metric,
    )


def summary_for_target(
    pred: np.ndarray, true: np.ndarray, std: np.ndarray
) -> Dict[str, object]:
    """Convenience: all three metrics, one call, JSON-serializable.

    Used by `scripts/wacv/phase3_calibration.py` to emit a single per-
    cohort, per-target (gene or pathway) summary dict.
    """
    rb = reliability_bins(pred, true, std)
    ece = expected_calibration_error(pred, true, std)
    sp_mse = selective_prediction(pred, true, std, error_metric="mse")
    sp_mae = selective_prediction(pred, true, std, error_metric="mae")
    return {
        "reliability": {
            "nominal": rb.nominal.tolist(),
            "empirical": rb.empirical.tolist(),
            "n_cells": rb.n_cells,
        },
        "ece": ece,
        "selective_prediction_mse": {
            "coverage": sp_mse.coverage.tolist(),
            "error": sp_mse.error.tolist(),
        },
        "selective_prediction_mae": {
            "coverage": sp_mae.coverage.tolist(),
            "error": sp_mae.error.tolist(),
        },
    }

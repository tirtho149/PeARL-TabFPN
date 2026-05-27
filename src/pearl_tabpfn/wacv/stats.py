"""Paired significance tests across CV folds (WACV Phase 1).

The WACV protocol demands a paired test between TabPFN-3 and each
anchor (MLP, TabPFN-v2) before any difference can be called an
improvement. Five folds is small enough that the parametric paired-t
and the non-parametric Wilcoxon signed-rank give different answers; we
report both and let the paper caption pick.

This module is intentionally tiny — `scipy.stats` already does the
heavy lifting. It exists so the call sites in Phase-1 scripts have a
single, documented surface and the test selection isn't re-invented in
six places.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


@dataclass
class PairedTestResult:
    """One paired-test outcome."""

    mean_diff: float  # mean(a - b) across folds; positive => a > b
    std_diff: float  # std(a - b)
    n_pairs: int  # number of folds
    paired_t_stat: float
    paired_t_p: float
    wilcoxon_stat: Optional[float]  # None when ties / n<6 make it ill-defined
    wilcoxon_p: Optional[float]


def paired_compare(a: np.ndarray, b: np.ndarray) -> PairedTestResult:
    """Compare two per-fold scalar arrays (e.g. PCC across 5 folds).

    `a` is the candidate (TabPFN-3 in WACV usage); `b` is the anchor
    (MLP or TabPFN-v2). Positive `mean_diff` means the candidate is
    better. Wilcoxon is the recommended primary test when n is small
    (WACV's 5 folds), with paired-t reported alongside for context.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a {a.shape}, b {b.shape}")
    if a.size < 2:
        raise ValueError("need at least 2 paired samples")
    diff = a - b
    t_stat, t_p = ttest_rel(a, b)
    try:
        # `zero_method="wilcox"` drops zero differences before ranking,
        # which is the standard convention but can leave n very small.
        w_stat, w_p = wilcoxon(a, b, zero_method="wilcox")
        w_stat_f: Optional[float] = float(w_stat)
        w_p_f: Optional[float] = float(w_p)
    except ValueError:
        # All-zero differences or fewer than the minimum sample size:
        # surfaces in JSON as null rather than a fake number.
        w_stat_f = None
        w_p_f = None
    return PairedTestResult(
        mean_diff=float(diff.mean()),
        std_diff=float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
        n_pairs=int(diff.size),
        paired_t_stat=float(t_stat),
        paired_t_p=float(t_p),
        wilcoxon_stat=w_stat_f,
        wilcoxon_p=w_p_f,
    )


def to_dict(r: PairedTestResult) -> Dict[str, object]:
    """JSON-serializable dict for paired_compare output."""
    return {
        "mean_diff": r.mean_diff,
        "std_diff": r.std_diff,
        "n_pairs": r.n_pairs,
        "paired_t_stat": r.paired_t_stat,
        "paired_t_p": r.paired_t_p,
        "wilcoxon_stat": r.wilcoxon_stat,
        "wilcoxon_p": r.wilcoxon_p,
    }

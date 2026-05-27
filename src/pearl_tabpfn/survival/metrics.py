"""C-index helpers — wraps lifelines.utils.concordance_index for survival
analysis. Convention: higher predicted risk → shorter expected survival, so
we pass -risk to lifelines (which expects higher score = better outcome)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def fold_c_index(risks: np.ndarray, os_time: np.ndarray, events: np.ndarray) -> float:
    """C-index for a single validation fold."""
    from lifelines.utils import concordance_index
    return float(concordance_index(os_time, -risks, events))


def concordance_index_5fold(fold_outputs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
                            ) -> dict:
    """Compute per-fold and aggregated C-index.

    Args:
        fold_outputs: list of (val_risks, val_os_time, val_events) per fold
    Returns:
        dict with c_index_per_fold, c_index_mean, c_index_std, n_folds
    """
    cs = [fold_c_index(*o) for o in fold_outputs]
    return {
        "c_index_per_fold": cs,
        "c_index_mean": float(np.mean(cs)),
        "c_index_std": float(np.std(cs)),
        "n_folds": len(cs),
    }

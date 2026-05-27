#!/usr/bin/env python
"""WACV Phase 4 — pathway activation analysis.

Consumes Phase 1 predictions, computes per-pathway spatial PCC (predicted
vs ssGSEA ground-truth), ranks pathways, and emits a summary JSON +
ranked list per cohort. Side-by-side activation maps are rendered by a
separate (matplotlib-only) script — see WACV_PIPELINE.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pearl_tabpfn.wacv import pathway_maps  # noqa: E402


def _pool_folds(phase1_dir: str) -> Dict[str, np.ndarray]:
    """Concatenate pathway_pred / pathway_true across folds."""
    pred_dir = os.path.join(phase1_dir, "predictions")
    paths = sorted(glob.glob(os.path.join(pred_dir, "fold_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No fold predictions under {pred_dir}")
    preds, trues = [], []
    for p in paths:
        f = np.load(p, allow_pickle=False)
        preds.append(f["pathway_pred"])
        trues.append(f["pathway_true"])
    return {
        "pred": np.concatenate(preds, axis=0),
        "true": np.concatenate(trues, axis=0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohorts", nargs="+", choices=["Breast", "Skin", "Lymph"],
        default=["Breast", "Skin", "Lymph"],
    )
    p.add_argument("--phase1-dir", default="./wacv_results/phase1")
    p.add_argument("--output-dir", default="./wacv_results/phase4")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summary: Dict[str, Any] = {"cohorts": {}}
    for cohort in args.cohorts:
        cohort_dir = os.path.join(args.phase1_dir, cohort, "tabpfn3")
        if not os.path.exists(cohort_dir):
            print(f"[phase4] skipping {cohort}: {cohort_dir} not found")
            continue
        pooled = _pool_folds(cohort_dir)
        per_cohort = pathway_maps.summary_for_fold(
            pooled["pred"], pooled["true"], top_k=args.top_k
        )
        out = os.path.join(args.output_dir, f"{cohort}.json")
        with open(out, "w") as f:
            json.dump(per_cohort, f, indent=2, default=str)
        summary["cohorts"][cohort] = {
            "mean_spatial_pcc": per_cohort["mean_spatial_pcc"],
            "median_spatial_pcc": per_cohort["median_spatial_pcc"],
            "n_valid": per_cohort["n_valid"],
            "out": out,
        }
        print(
            f"[phase4] {cohort}: mean spatial PCC="
            f"{per_cohort['mean_spatial_pcc']:.4f}, "
            f"n_valid={per_cohort['n_valid']}/{per_cohort['n_pathways']} -> {out}"
        )

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nPhase 4 summary written to {args.output_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

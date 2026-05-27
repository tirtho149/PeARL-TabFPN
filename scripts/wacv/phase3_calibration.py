#!/usr/bin/env python
"""WACV Phase 3 — calibration & uncertainty.

Consumes the per-fold prediction `.npz` artifacts emitted by Phase 1
when --save-posteriors is set, and emits:

  - Reliability bins (nominal vs empirical coverage), per cohort.
  - Expected Calibration Error (scalar), per cohort.
  - Selective-prediction curve (error vs coverage), per cohort.

The headline plot is the selective-prediction curve — uncertainty being
*informative* matters more for the WACV story than uncertainty being
present.
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

from pearl_tabpfn.wacv import calibration  # noqa: E402


def _load_fold_predictions(phase1_dir: str) -> List[Dict[str, np.ndarray]]:
    """Load every fold .npz emitted under {phase1_dir}/predictions/."""
    pred_dir = os.path.join(phase1_dir, "predictions")
    paths = sorted(glob.glob(os.path.join(pred_dir, "fold_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No fold predictions under {pred_dir}. Did Phase 1 run "
            "with --save-posteriors?"
        )
    folds = []
    for p in paths:
        f = np.load(p, allow_pickle=False)
        folds.append({k: f[k] for k in f.files})
    return folds


def _stack_folds(folds: List[Dict[str, np.ndarray]], key: str) -> np.ndarray:
    """Concatenate per-fold arrays along axis 0.

    Missing key raises with a helpful pointer to Phase 1.
    """
    if not all(key in f for f in folds):
        raise KeyError(
            f"Key {key!r} missing from at least one fold .npz. "
            "Phase 1 must run with --save-posteriors AND --head-mode "
            "tabpfn3 (or both3) to populate predictive std."
        )
    return np.concatenate([f[key] for f in folds], axis=0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohorts", nargs="+", choices=["Breast", "Skin", "Lymph"],
        default=["Breast", "Skin", "Lymph"],
    )
    p.add_argument(
        "--phase1-dir", default="./wacv_results/phase1",
        help="Root of Phase 1 outputs. Each cohort is expected at "
             "{phase1_dir}/{cohort}/tabpfn3/.",
    )
    p.add_argument("--output-dir", default="./wacv_results/phase3")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summary: Dict[str, Any] = {"cohorts": {}}
    for cohort in args.cohorts:
        cohort_dir = os.path.join(args.phase1_dir, cohort, "tabpfn3")
        if not os.path.exists(cohort_dir):
            print(f"[phase3] skipping {cohort}: {cohort_dir} not found")
            continue
        folds = _load_fold_predictions(cohort_dir)
        pred_p = _stack_folds(folds, "pathway_pred")
        true_p = _stack_folds(folds, "pathway_true")
        std_p = _stack_folds(folds, "pathway_std")
        pred_g = _stack_folds(folds, "gene_pred")
        true_g = _stack_folds(folds, "gene_true")
        std_g = _stack_folds(folds, "gene_std")

        per_cohort: Dict[str, Any] = {
            "pathway": calibration.summary_for_target(pred_p, true_p, std_p),
            "gene":    calibration.summary_for_target(pred_g, true_g, std_g),
            "n_folds": len(folds),
            "n_spots_total": int(pred_p.shape[0]),
        }
        out = os.path.join(args.output_dir, f"{cohort}.json")
        with open(out, "w") as f:
            json.dump(per_cohort, f, indent=2, default=str)
        summary["cohorts"][cohort] = {
            "pathway_ece": per_cohort["pathway"]["ece"],
            "gene_ece":    per_cohort["gene"]["ece"],
            "out": out,
        }
        print(
            f"[phase3] {cohort}: pathway ECE={per_cohort['pathway']['ece']:.4f} "
            f"gene ECE={per_cohort['gene']['ece']:.4f} -> {out}"
        )

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nPhase 3 summary written to {args.output_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

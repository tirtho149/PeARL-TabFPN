#!/usr/bin/env python
"""WACV Phase 2 — single-model configuration sweep (Breast + Skin).

Axes (one figure per axis, mean ± std band):

  - Estimators: {2, 4, 8, 16, 32}
  - Context size: spots-per-section ∈ {100, 200, 400}
    (headline number stays at the protocol-default cap; apple-to-apple
    boundary is checked here)
  - Precision: {fp32, bf16, fp16}
  - (Optional) Context selection ablation: all vs random vs similarity-
    selected. Only if Phase 0 budget allows.

Lymph is skipped unless the per-fold runs are cheap (Phase 0e decides).

Status: SCAFFOLDED. The sweep delegates to `scripts/train_tabpfn3.py`
once per axis point, with --tabpfn3-* flags varying. Aggregation reads
per-fold PCC arrays from each run's reproduction_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ESTIMATOR_GRID = [2, 4, 8, 16, 32]
CONTEXT_GRID = [100, 200, 400]
PRECISION_GRID = ["fp32", "bf16", "fp16"]


def _run_v3(cohort: str, out_dir: str, n_sections: int, folds: int,
            n_estimators: int, precision: str, context_cap: int) -> None:
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "scripts/train_tabpfn3.py"),
        "--apple-to-apple",
        "--cohort", cohort,
        "--n-sections", str(n_sections),
        "--folds", str(folds),
        "--head-mode", "tabpfn3",
        "--tabpfn3-n-estimators", str(n_estimators),
        "--tabpfn3-precision", precision,
        "--tabpfn3-context-cap", str(context_cap),
        "--output-dir", out_dir,
    ]
    print(f"\n[phase2] {os.path.relpath(out_dir)} :: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _load_per_fold_pcc(out_dir: str) -> Dict[str, List[float]]:
    with open(os.path.join(out_dir, "reproduction_results.json")) as f:
        r = json.load(f)
    folds = r.get("per_fold", [])
    return {
        "gene": [float(f.get("tabpfn3", {}).get("gene", {}).get("PCC", float("nan")))
                 for f in folds],
        "pathway": [float(f.get("tabpfn3", {}).get("pathway", {}).get("PCC", float("nan")))
                    for f in folds],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohorts", nargs="+", choices=["Breast", "Skin", "Lymph"],
        default=["Breast", "Skin"],
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-sections", type=int, default=None)
    p.add_argument("--output-dir", default="./wacv_results/phase2")
    p.add_argument(
        "--axes", nargs="+",
        choices=["estimators", "context", "precision"],
        default=["estimators", "context", "precision"],
    )
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = "./wacv_results/phase0/config.json"
    if not os.path.exists(config_path):
        print(
            f"[phase2] WARNING: {config_path} not found. Phase 0 must "
            "complete before Phase 2 so the headline (default) config is "
            "frozen. Proceeding with built-in defaults — DO NOT publish "
            "these numbers without the Phase-0 frozen config."
        )

    summary: Dict[str, Any] = {"axes": {}}

    for cohort in args.cohorts:
        n_sections = args.n_sections or (36 if cohort == "Breast" else 24)

        if "estimators" in args.axes:
            axis_out: Dict[str, Any] = {}
            for n_est in ESTIMATOR_GRID:
                d = os.path.join(args.output_dir, cohort, f"est_{n_est}")
                _run_v3(cohort, d, n_sections, args.folds,
                        n_estimators=n_est, precision="fp32",
                        context_cap=400)
                axis_out[str(n_est)] = _load_per_fold_pcc(d)
            summary["axes"].setdefault("estimators", {})[cohort] = axis_out

        if "context" in args.axes:
            axis_out = {}
            for cap in CONTEXT_GRID:
                d = os.path.join(args.output_dir, cohort, f"ctx_{cap}")
                _run_v3(cohort, d, n_sections, args.folds,
                        n_estimators=8, precision="fp32", context_cap=cap)
                axis_out[str(cap)] = _load_per_fold_pcc(d)
            summary["axes"].setdefault("context", {})[cohort] = axis_out

        if "precision" in args.axes:
            axis_out = {}
            for prec in PRECISION_GRID:
                d = os.path.join(args.output_dir, cohort, f"prec_{prec}")
                _run_v3(cohort, d, n_sections, args.folds,
                        n_estimators=8, precision=prec, context_cap=400)
                axis_out[prec] = _load_per_fold_pcc(d)
            summary["axes"].setdefault("precision", {})[cohort] = axis_out

    out = os.path.join(args.output_dir, "config_sweep.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nPhase 2 sweep summary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

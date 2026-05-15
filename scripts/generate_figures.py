#!/usr/bin/env python
"""Entry point: generate head-to-head figures from saved fold predictions.

Run AFTER `run_reproduction.py` finishes (or after running baseline + tabpfn
as separate jobs and merging the predictions/ dirs).

Usage:
    python scripts/generate_figures.py \\
        --results-dir ./results/reproduction_apple_to_apple \\
        --fold 0 \\
        --output-dir ./results/reproduction_apple_to_apple/figures

Produces fig_h2h_1_metric_bars.png ... fig_h2h_7_pathway_corr.png — the
seven figures the BIBM 2026 paper references.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from pearl_tabpfn.figures import generate_head_to_head_figures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="Directory containing reproduction_results.json and predictions/")
    p.add_argument("--fold", type=int, default=0,
                   help="Which fold's predictions to use for the per-spot figures (default 0).")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    results_path = os.path.join(args.results_dir, "reproduction_results.json")
    if not os.path.isfile(results_path):
        print(f"ERROR: missing {results_path}", file=sys.stderr)
        return 1
    preds_path = os.path.join(args.results_dir, "predictions", f"fold_{args.fold}.npz")
    if not os.path.isfile(preds_path):
        print(f"ERROR: missing {preds_path}", file=sys.stderr)
        print(f"  Available folds in predictions/:", file=sys.stderr)
        pdir = os.path.join(args.results_dir, "predictions")
        if os.path.isdir(pdir):
            for f in sorted(os.listdir(pdir)):
                print(f"    {f}", file=sys.stderr)
        return 1

    with open(results_path) as f:
        res = json.load(f)
    preds = np.load(preds_path)

    required = ["coords", "pathway_pred_mlp", "pathway_pred_tabpfn",
                "pathway_true", "gene_pred_mlp", "gene_pred_tabpfn", "gene_true"]
    missing = [k for k in required if k not in preds.files]
    if missing:
        print(f"ERROR: {preds_path} is missing keys: {missing}", file=sys.stderr)
        print(f"  Did you run both --head-mode mlp and --head-mode tabpfn?", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    generate_head_to_head_figures(
        summary_baseline=res["summary"]["baseline"],
        summary_tabpfn=res["summary"]["tabpfn"],
        coords=preds["coords"],
        pathway_pred_mlp=preds["pathway_pred_mlp"],
        pathway_pred_tabpfn=preds["pathway_pred_tabpfn"],
        pathway_true=preds["pathway_true"],
        gene_pred_mlp=preds["gene_pred_mlp"],
        gene_pred_tabpfn=preds["gene_pred_tabpfn"],
        gene_true=preds["gene_true"],
        output_dir=args.output_dir,
    )
    print(f"\nFigures saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

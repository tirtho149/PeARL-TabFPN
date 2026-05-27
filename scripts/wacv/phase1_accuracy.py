#!/usr/bin/env python
"""WACV Phase 1 — core accuracy characterization (main result table).

Protocol (docs/WACV_PIPELINE.md):

  - Run TabPFN-3 across all 3 cohorts × both target types × 5 folds.
  - Report mean ± std across folds in every cell.
  - Include PEaRL-MLP and TabPFN-v2 as anchors, run under identical
    conditions in this same phase (NOT pulled from BIBM results).
  - Paired Wilcoxon / paired-t between TabPFN-3 and each anchor across
    folds. Report p-values.

This script orchestrates three runs per cohort and assembles the main
table at the end. The per-cohort run uses --head-mode mlp, tabpfn, and
tabpfn3 in sequence; --apple-to-apple bundles every paper-faithful
preset.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pearl_tabpfn.wacv.stats import paired_compare, to_dict  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run_one(cohort: str, head_mode: str, out_root: str,
             n_sections: int, folds: int, save_posteriors: bool) -> str:
    """Invoke reproduction.main via the canonical wrapper.

    Returns the output directory that contains
    reproduction_results.json. Does NOT fail on TabPFN-3 not-installed
    until the subprocess actually tries to import it — by design, so a
    user can dry-run head_mode=mlp without tabpfn on the path.
    """
    out_dir = os.path.join(out_root, cohort, head_mode)
    os.makedirs(out_dir, exist_ok=True)
    script = "scripts/train_tabpfn3.py" if head_mode in ("tabpfn3", "both3") \
        else "scripts/run_reproduction.py"
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, script),
        "--apple-to-apple",
        "--cohort", cohort,
        "--n-sections", str(n_sections),
        "--folds", str(folds),
        "--head-mode", head_mode,
        "--output-dir", out_dir,
    ]
    if save_posteriors:
        cmd.append("--save-posteriors")
    print(f"\n[phase1] {cohort} / {head_mode}\n  cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_dir


def _load_fold_pccs(out_dir: str, variant: str) -> Dict[str, List[float]]:
    """Per-fold PCC arrays for `variant` from reproduction_results.json.

    Returns {'gene': [...], 'pathway': [...]} length = #folds.
    """
    with open(os.path.join(out_dir, "reproduction_results.json")) as f:
        results = json.load(f)
    folds = results.get("per_fold", [])
    out = {"gene": [], "pathway": []}
    for fold in folds:
        v = fold.get(variant, {})
        out["gene"].append(float(v.get("gene", {}).get("PCC", float("nan"))))
        out["pathway"].append(float(v.get("pathway", {}).get("PCC", float("nan"))))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohorts", nargs="+", choices=["Breast", "Skin", "Lymph"],
        default=["Breast", "Skin", "Lymph"],
    )
    p.add_argument(
        "--n-sections", type=int, default=None,
        help="Per-cohort section count. Defaults: Breast 36, Skin/Lymph 24.",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--output-dir", default="./wacv_results/phase1")
    p.add_argument(
        "--skip-anchors", action="store_true",
        help="Skip the MLP and TabPFN-v2 anchor runs (useful when "
             "iterating only on the v3 side).",
    )
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summary: Dict[str, Any] = {"cohorts": {}}

    for cohort in args.cohorts:
        n_sections = args.n_sections or (36 if cohort == "Breast" else 24)
        cohort_summary: Dict[str, Any] = {"n_sections": n_sections}

        # 1) MLP anchor.
        if not args.skip_anchors:
            mlp_dir = _run_one(cohort, "mlp", args.output_dir,
                               n_sections, args.folds, save_posteriors=False)
            cohort_summary["mlp"] = _load_fold_pccs(mlp_dir, "baseline")

            # 2) TabPFN-v2 anchor.
            v2_dir = _run_one(cohort, "tabpfn", args.output_dir,
                              n_sections, args.folds, save_posteriors=False)
            cohort_summary["tabpfn_v2"] = _load_fold_pccs(v2_dir, "tabpfn")

        # 3) TabPFN-3 candidate (always run).
        v3_dir = _run_one(cohort, "tabpfn3", args.output_dir,
                          n_sections, args.folds, save_posteriors=True)
        cohort_summary["tabpfn_v3"] = _load_fold_pccs(v3_dir, "tabpfn3")

        # Paired tests: v3 vs each anchor, gene and pathway.
        if not args.skip_anchors:
            cohort_summary["paired"] = {
                "v3_vs_mlp_gene": to_dict(
                    paired_compare(cohort_summary["tabpfn_v3"]["gene"],
                                   cohort_summary["mlp"]["gene"])),
                "v3_vs_mlp_pathway": to_dict(
                    paired_compare(cohort_summary["tabpfn_v3"]["pathway"],
                                   cohort_summary["mlp"]["pathway"])),
                "v3_vs_v2_gene": to_dict(
                    paired_compare(cohort_summary["tabpfn_v3"]["gene"],
                                   cohort_summary["tabpfn_v2"]["gene"])),
                "v3_vs_v2_pathway": to_dict(
                    paired_compare(cohort_summary["tabpfn_v3"]["pathway"],
                                   cohort_summary["tabpfn_v2"]["pathway"])),
            }
        summary["cohorts"][cohort] = cohort_summary

    main_table = os.path.join(args.output_dir, "main_table.json")
    with open(main_table, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nPhase 1 main table written to {main_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

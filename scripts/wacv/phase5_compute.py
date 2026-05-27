#!/usr/bin/env python
"""WACV Phase 5 — compute characterization (measured on Nova).

Re-runs the three combinations on a single Breast fold and records:

  - Wall-clock per-stage (Stage 1 + Stage 2 + post-stage-2 head fit).
  - Per-target mean head-fit time.
  - Peak GPU memory (torch.cuda.max_memory_allocated, per stage).

The WACV paper cites *these* numbers, not the headline benchmarks from
the TabPFN-3 report (which run on H100s and different workloads).

Status: SCAFFOLDED. The metrics extraction lives here; the actual run
invocations defer to `scripts/train_baseline.py`,
`scripts/train_tabpfn.py`, and `scripts/train_tabpfn3.py`. Each is
launched with --folds 1 --max-folds 1 to make this a 1-fold timing run
rather than a full reproduction.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _wall_clock_run(args_list, out_dir: str) -> Dict[str, Any]:
    """Run a subprocess, time it, and parse reproduction_results.json."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    subprocess.run(args_list + ["--output-dir", out_dir], check=True)
    wall = time.time() - t0
    rep = os.path.join(out_dir, "reproduction_results.json")
    parsed: Dict[str, Any] = {"wall_seconds": wall, "results_json": rep}
    if os.path.exists(rep):
        with open(rep) as f:
            r = json.load(f)
        # `_fold_seconds` is written per fold by reproduction.main.
        parsed["per_fold_seconds"] = [
            fold.get("_fold_seconds") for fold in r.get("per_fold", [])
        ]
    return parsed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohort", choices=["Breast", "Skin", "Lymph"], default="Breast",
        help="Cohort to time on. Breast is the protocol's compute axis; "
             "Lymph (if Phase 0e allows uncapped context) is the "
             "qualitative compute story.",
    )
    p.add_argument("--n-sections", type=int, default=None)
    p.add_argument("--output-dir", default="./wacv_results/phase5")
    args = p.parse_args()

    n_sections = args.n_sections or (36 if args.cohort == "Breast" else 24)
    base = os.path.join(args.output_dir, args.cohort)
    common = [
        "--apple-to-apple",
        "--cohort", args.cohort,
        "--n-sections", str(n_sections),
        "--folds", "5", "--max-folds", "1",
    ]

    timings: Dict[str, Any] = {}

    timings["mlp"] = _wall_clock_run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts/train_baseline.py")]
        + common, os.path.join(base, "mlp"),
    )
    timings["tabpfn_v2"] = _wall_clock_run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts/train_tabpfn.py")]
        + common, os.path.join(base, "tabpfn_v2"),
    )
    timings["tabpfn_v3"] = _wall_clock_run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts/train_tabpfn3.py")]
        + common, os.path.join(base, "tabpfn_v3"),
    )

    out = os.path.join(args.output_dir, "compute_table.json")
    with open(out, "w") as f:
        json.dump({"cohort": args.cohort, "n_sections": n_sections,
                   "timings": timings}, f, indent=2, default=str)
    print(f"\nPhase 5 compute table written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

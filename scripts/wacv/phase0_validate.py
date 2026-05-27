#!/usr/bin/env python
"""WACV Phase 0 — gating validation (do this first, ~1 day).

Implements the five sub-checks from
docs/WACV_PIPELINE.md (mirroring the guidelines doc):

  0a. TabPFN-3 install + license — confirm the package imports, record
      the version, and assert the license was accepted.
  0b. GPU placement check — run a trivial fit/predict and confirm it
      executes on GPU, not CPU fallback. Record device + peak memory.
  0c. Single-fold timing on Breast — run TabPFN-3 per-target across all
      targets for ONE Breast fold. Record wall-clock, per-target mean,
      peak GPU memory, gene PCC, pathway PCC.
  0d. Estimator sanity — 8 vs 32 — same fold, two estimator counts;
      compare PCC and time deltas. Decision rule: if PCC drop from
      32→8 is within ~0.005, use 8 everywhere. Else keep 32.
  0e. Lymph memory probe — one Lymph fold at capped context and one
      uncapped (~74k spots). Decision rule: uncapped if it fits, else
      capped to match Breast/Skin.

The script writes a fixed-config JSON at the end:
  wacv_results/phase0/config.json
which Phases 1–5 read. **Do not proceed past Phase 0 until this JSON
exists.**

This is the gating phase. Current scaffold:
  - 0a runs today (import check + version record)
  - 0b–0e raise NotImplementedError once 0a passes, pointing at the
    exact runner invocation that fills them.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch  # noqa: E402

PHASE_DIR_DEFAULT = "./wacv_results/phase0"


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  wrote {path}")


def check_0a_install(out_dir: str) -> Dict[str, Any]:
    print("\n[0a] TabPFN-3 install + license")
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import tabpfn  # noqa: F401

        info["tabpfn_version"] = getattr(tabpfn, "__version__", "unknown")
        info["tabpfn_installed"] = True
        print(f"  tabpfn {info['tabpfn_version']} importable")
    except ImportError as e:
        info["tabpfn_installed"] = False
        info["import_error"] = str(e)
        print(
            "  tabpfn NOT importable. Install:\n"
            "    pip install tabpfn>=7\n"
            "  and accept the Prior Labs RAIL license at\n"
            "    https://github.com/PriorLabs/TabPFN"
        )
    _write_json(os.path.join(out_dir, "0a_environment.json"), info)
    return info


def check_0b_gpu_placement(out_dir: str) -> Dict[str, Any]:
    print("\n[0b] GPU placement check")
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)
    raise NotImplementedError(
        "0b requires running a trivial TabPFN-3 fit/predict and "
        "recording peak GPU memory. Fill in once 0a passes: "
        "instantiate TabPFNRegressor on device='cuda', fit on a "
        "small synthetic (X, y), measure torch.cuda.max_memory_allocated."
    )


def check_0c_single_fold_breast(out_dir: str) -> Dict[str, Any]:
    print("\n[0c] Single-fold timing — Breast")
    raise NotImplementedError(
        "0c invokes the runner: \n"
        "    python scripts/train_tabpfn3.py --apple-to-apple \\\n"
        "        --cohort Breast --n-sections 36 --folds 5 --max-folds 1 \\\n"
        "        --save-posteriors \\\n"
        "        --output-dir wacv_results/phase0/0c_breast_fold0\n"
        "and parses reproduction_results.json for wall-clock, per-target "
        "time, peak memory, gene PCC, pathway PCC. Wire after 0a/0b."
    )


def check_0d_estimator_sweep(out_dir: str) -> Dict[str, Any]:
    print("\n[0d] Estimator sanity — 8 vs 32")
    raise NotImplementedError(
        "0d runs 0c twice with --tabpfn3-n-estimators 8 and 32, compares "
        "gene/pathway PCC and wall-clock. Decision rule per protocol:\n"
        "  PCC delta < 0.005 → use 8 everywhere (write to config.json)\n"
        "  else              → keep 32 (and update budget in Phases 1–5)"
    )


def check_0e_lymph_probe(out_dir: str) -> Dict[str, Any]:
    print("\n[0e] Lymph memory probe")
    raise NotImplementedError(
        "0e runs ONE Lymph fold capped (--tabpfn3-context-cap 400) AND "
        "attempts ONE uncapped (--tabpfn3-context-cap 0). Records OOM "
        "vs success. Decision rule:\n"
        "  uncapped fits → Lymph-at-scale becomes a WACV showcase axis\n"
        "  OOM           → cap Lymph to 400, note constraint in paper"
    )


def emit_frozen_config(out_dir: str, decisions: Dict[str, Any]) -> None:
    path = os.path.join(out_dir, "config.json")
    payload = {
        "phase": "0",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decisions": decisions,
        "consumed_by": ["phase1_accuracy", "phase2_config_sweep",
                        "phase3_calibration", "phase4_pathway_maps",
                        "phase5_compute"],
    }
    _write_json(path, payload)
    print(
        "\nPhase 0 GATE: this config.json now drives Phases 1–5. "
        "Re-scope the run budget against the real numbers before "
        "submitting Phase 1."
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=PHASE_DIR_DEFAULT)
    p.add_argument(
        "--only", choices=["0a", "0b", "0c", "0d", "0e", "all"], default="all",
        help="Run a single sub-check (useful while wiring) or all.",
    )
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    decisions: Dict[str, Any] = {}
    if args.only in ("0a", "all"):
        info_0a = check_0a_install(args.output_dir)
        decisions["0a"] = info_0a
        if not info_0a.get("tabpfn_installed"):
            print(
                "\nPhase 0 BLOCKED at 0a — install tabpfn and accept the "
                "license before retrying."
            )
            return 1

    if args.only in ("0b", "all"):
        decisions["0b"] = check_0b_gpu_placement(args.output_dir)
    if args.only in ("0c", "all"):
        decisions["0c"] = check_0c_single_fold_breast(args.output_dir)
    if args.only in ("0d", "all"):
        decisions["0d"] = check_0d_estimator_sweep(args.output_dir)
    if args.only in ("0e", "all"):
        decisions["0e"] = check_0e_lymph_probe(args.output_dir)

    emit_frozen_config(args.output_dir, decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

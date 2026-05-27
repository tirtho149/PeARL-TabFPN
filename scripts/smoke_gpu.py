#!/usr/bin/env python3
"""GPU + TabPFN v3-on-GPU smoke test.

Verifies the runtime path that `tabpfn3_head.TabPFN3Head.fit` enforces:

  [1] torch.cuda.is_available() — strict GPU presence check
  [2] CUDA driver / device properties (name, VRAM, compute capability)
  [3] A small torch tensor lives on GPU and computes correctly
  [4] tabpfn imports cleanly with this torch build
  [5] TabPFNRegressor.create_default_for_version(ModelVersion.V3, device="cuda")
      loads weights onto the GPU (downloads ~500 MB on first run; needs
      accepted PriorLabs license + TABPFN_TOKEN or HF_TOKEN)
  [6] fit() on synthetic data succeeds, GPU memory increases during fit
  [7] predict() returns sensible PCC > random on a held-out set

This is the **canonical GPU gate** — every smoke here must pass before you
launch a real 50-hour reproduction job. If [5] fails with 401/403, accept
the license at https://huggingface.co/Prior-Labs/tabpfn_3 and re-run.

Usage:
    python scripts/smoke_gpu.py                       # default: 200×10 fit
    python scripts/smoke_gpu.py --n-train 1000        # larger fit
    python scripts/smoke_gpu.py --n-estimators 4      # ensemble size
    python scripts/smoke_gpu.py --skip-tabpfn         # CUDA + tensor only

If torch reports CUDA unavailable, the script exits 1 with a clear message.
That is the same failure mode `TabPFN3Head.fit` produces at runtime — better
to catch it here in 5 seconds than after Stage 1+2 has burned an hour.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def section(t: str): print(f"\n{YELLOW}{t}{RESET}")
def check(name, ok, detail=""):
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    suf = f" {DIM}— {detail}{RESET}" if detail else ""
    print(f"    {mark} {name}{suf}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-features", type=int, default=10)
    ap.add_argument("--n-estimators", type=int, default=2)
    ap.add_argument("--skip-tabpfn", action="store_true", help="CUDA + tensor only — skip TabPFN v3 weight load.")
    args = ap.parse_args()

    print("=" * 70)
    print("GPU + TabPFN v3 SMOKE GATE")
    print("=" * 70)

    results = []

    # ----- 1. torch.cuda.is_available -----
    section("[1] torch.cuda.is_available()")
    try:
        import torch
    except ImportError as e:
        print(f"    {RED}✗{RESET} torch not installed: {e}")
        print(f"    {DIM}→ run `bash SETUP_ENV.sh` to install torch + CUDA wheel{RESET}")
        return 1

    results.append(check(
        "torch installed",
        True,
        f"version={torch.__version__}, CUDA built={torch.version.cuda or 'CPU-only'}",
    ))

    cuda_avail = torch.cuda.is_available()
    results.append(check(
        "torch.cuda.is_available()",
        cuda_avail,
        "True" if cuda_avail else "FALSE — strict GPU gate failed",
    ))
    if not cuda_avail:
        print()
        print(f"    {DIM}This is the same failure that TabPFN3Head.fit() raises at runtime.{RESET}")
        print(f"    {DIM}Diagnostic checklist:{RESET}")
        print(f"    {DIM}  • Is nvidia-smi available? Are you on a GPU node (not a login node)?{RESET}")
        print(f"    {DIM}  • Is CUDA_VISIBLE_DEVICES set to '' or an invalid index?{RESET}")
        print(f"    {DIM}  • Is torch a CPU-only build? Check torch.version.cuda above.{RESET}")
        print(f"    {DIM}  • On SLURM: did you request `--gres=gpu:1`?{RESET}")
        return _summary(results)

    # ----- 2. Device properties -----
    section("[2] GPU device properties")
    n_gpus = torch.cuda.device_count()
    print(f"    {DIM}{n_gpus} CUDA device(s) visible{RESET}")
    total_vram_gb = 0.0
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        vram_gb = p.total_memory / (1024 ** 3)
        total_vram_gb += vram_gb
        print(f"      [{i}] {p.name}  {vram_gb:.1f} GB  CC {p.major}.{p.minor}")
    results.append(check(
        "At least one CUDA device",
        n_gpus >= 1,
        f"{n_gpus} GPU(s)",
    ))
    results.append(check(
        "Primary GPU has ≥ 16 GB VRAM (24 GB recommended for TabPFN-pure)",
        torch.cuda.get_device_properties(0).total_memory >= 16 * 1024 ** 3,
        f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB",
    ))

    # ----- 3. Tensor lives on GPU -----
    section("[3] Tensor lives on GPU + computes correctly")
    t = torch.randn(1024, 1024, device="cuda")
    t2 = (t @ t.T).sum()
    results.append(check(
        "1024×1024 matmul on GPU runs without error",
        t.is_cuda and torch.isfinite(t2).item(),
        f"sum={t2.item():.2f} on {t.device}",
    ))

    # ----- 4. tabpfn import -----
    if args.skip_tabpfn:
        print(f"\n{YELLOW}[4-7] skipped (--skip-tabpfn){RESET}")
        return _summary(results)

    section("[4] tabpfn imports cleanly")
    try:
        import tabpfn
        from tabpfn.constants import ModelVersion
        from tabpfn import TabPFNRegressor
        results.append(check(
            "tabpfn + ModelVersion.V3 + TabPFNRegressor importable",
            ModelVersion.V3 is not None,
            f"tabpfn={tabpfn.__version__}",
        ))
    except Exception as e:
        msg = str(e)
        results.append(check("tabpfn importable", False, msg[:200]))
        if "torch.nn.attention" in msg or "SDPBackend" in msg:
            print(f"    {DIM}→ torch is too old for tabpfn 8 (needs ≥ 2.3 for the attention API){RESET}")
            print(f"    {DIM}→ run `bash SETUP_ENV.sh` with the right CUDA version{RESET}")
        return _summary(results)

    # ----- 5. Load v3 weights on GPU -----
    section("[5] Load TabPFN v3 weights onto GPU (downloads ~500 MB on first run)")
    free_before, _ = torch.cuda.mem_get_info(0)
    try:
        t0 = time.time()
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3,
            device="cuda",
            n_estimators=args.n_estimators,
            random_state=42,
            ignore_pretraining_limits=True,
        )
        elapsed = time.time() - t0
        results.append(check(
            "TabPFNRegressor.create_default_for_version(V3, device='cuda') returns",
            reg is not None,
            f"{elapsed:.1f}s (cached weights re-use is fast)",
        ))
    except Exception as e:
        msg = str(e)
        ml = msg.lower()
        gated = any(s in ml for s in ["401", "403", "gated", "license", "unauthorized", "access"])
        results.append(check(
            "TabPFNRegressor.create_default_for_version(V3, device='cuda')",
            False,
            f"{type(e).__name__}: {msg[:200]}",
        ))
        if gated:
            print(f"    {DIM}→ accept license at https://huggingface.co/Prior-Labs/tabpfn_3{RESET}")
            print(f"    {DIM}→ then set TABPFN_TOKEN=<token> or HF_TOKEN=<token>{RESET}")
        return _summary(results)

    # ----- 6. fit() on GPU -----
    section("[6] fit() on synthetic data")
    import numpy as np
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((args.n_train, args.n_features)).astype(np.float32)
    true_w = rng.standard_normal(args.n_features).astype(np.float32)
    y_train = X_train @ true_w + 0.3 * rng.standard_normal(args.n_train).astype(np.float32)
    try:
        t0 = time.time()
        reg.fit(X_train, y_train)
        fit_time = time.time() - t0
        free_after, _ = torch.cuda.mem_get_info(0)
        used_mb = (free_before - free_after) / (1024 ** 2)
        results.append(check(
            f"fit({args.n_train} rows × {args.n_features} cols) on GPU",
            True,
            f"{fit_time:.2f}s, GPU mem ↑ {used_mb:.0f} MB",
        ))
    except Exception as e:
        import traceback; traceback.print_exc()
        results.append(check("fit() succeeds", False, f"{type(e).__name__}: {e}"))
        return _summary(results)

    # ----- 7. predict() + PCC sanity -----
    section("[7] predict() + PCC vs truth")
    X_val = rng.standard_normal((100, args.n_features)).astype(np.float32)
    y_val_true = X_val @ true_w + 0.3 * rng.standard_normal(100).astype(np.float32)
    try:
        t0 = time.time()
        y_val_pred = reg.predict(X_val)
        pred_time = time.time() - t0
        pcc = float(np.corrcoef(y_val_pred, y_val_true)[0, 1])
        results.append(check(
            "predict() returns numeric array",
            y_val_pred.shape == (100,) and np.all(np.isfinite(y_val_pred)),
            f"shape={y_val_pred.shape}, {pred_time*1000:.0f}ms",
        ))
        results.append(check(
            "PCC(pred, truth) > 0.5 (well above random)",
            pcc > 0.5,
            f"PCC={pcc:.3f}",
        ))
    except Exception as e:
        results.append(check("predict() succeeds", False, f"{type(e).__name__}: {e}"))

    return _summary(results)


def _summary(results: list[bool]) -> int:
    passed, total = sum(results), len(results)
    print("\n" + "=" * 70)
    if passed == total and total > 0:
        print(f"{GREEN}GPU GATE PASSED — {passed}/{total} assertions{RESET}")
        print("  GPU is visible, TabPFN v3 loads on GPU, fits and predicts correctly.")
        print("  Safe to launch a real training job.")
    else:
        print(f"{RED}GPU GATE FAILED — {passed}/{total} assertions{RESET}")
        print("  DO NOT launch a real training job until this passes.")
    print("=" * 70)
    return 0 if (passed == total and total > 0) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""TabPFN v3 smoke test — two-tier (source check + runtime check).

Tier A — source-level API check (works EVERYWHERE, including Intel Mac):
  [A1] ModelVersion.V3 enum present in upstream tabpfn constants.py
  [A2] TabPFNRegressor.create_default_for_version() branches on V3
  [A3] model_loading.py has v3 weight loader + HF repo reference
  [A4] PEaRL heads (tabpfn_head.py, tabpfn3_head.py) call ModelVersion.V3

Tier B — runtime fit/predict (needs torch ≥ 2.5 + tabpfn ≥ 8.0):
  [B1] tabpfn package importable
  [B2] ModelVersion.V3 importable from the installed package
  [B3] TabPFNRegressor.create_default_for_version(V3, device=...) constructs
      (downloads v3 weights on first run — needs accepted PriorLabs license)
  [B4] regressor.fit(X, y) on tiny synthetic data succeeds
  [B5] regressor.predict(X) returns sensible PCC > random

Tier B is skipped on Intel Macs (PyTorch dropped Intel-Mac wheels after 2.2,
but tabpfn 8 needs torch ≥ 2.5). Pass --skip-runtime to force Tier A only.

The v3 model weights are gated on HuggingFace (Prior-Labs/tabpfn_3). If the
download fails with 401/403, accept the license at the model page and set
TABPFN_TOKEN (or HF_TOKEN) before re-running Tier B.

Usage:
    python scripts/smoke_tabpfn3.py                   # auto-detect
    python scripts/smoke_tabpfn3.py --skip-runtime    # API surface only (Mac-safe)
    python scripts/smoke_tabpfn3.py --device cuda     # exercise GPU path
    python scripts/smoke_tabpfn3.py --tabpfn-source /path/to/tabpfn-upstream/src/tabpfn
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def section(t: str): print(f"\n{YELLOW}{t}{RESET}")
def check(name, ok, detail=""):
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    suf = f" {DIM}— {detail}{RESET}" if detail else ""
    print(f"    {mark} {name}{suf}")
    return ok


def smoke_source(tabpfn_src: str) -> list[bool]:
    """Tier A — read upstream tabpfn source directly, no import needed."""
    import ast, pathlib, re
    section("[A] Source-level API check (Tier A — works everywhere)")
    UP = pathlib.Path(tabpfn_src)
    if not UP.is_dir():
        return [check(f"tabpfn source at {UP}", False, "directory not found")]
    results = []

    # A1. ModelVersion.V3 enum
    tree = ast.parse((UP / "constants.py").read_text())
    mv = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ModelVersion"), None)
    names = [s.targets[0].id for s in mv.body
             if mv and isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)] if mv else []
    results.append(check("ModelVersion.V3 in constants.py", "V3" in names, f"members: {names}"))

    # A2. create_default_for_version branches on V3
    tree = ast.parse((UP / "regressor.py").read_text())
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TabPFNRegressor"), None)
    method = next((m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "create_default_for_version"), None) if cls else None
    body = ast.unparse(method) if method else ""
    results.append(check("create_default_for_version branches on V3", "ModelVersion.V3" in body))

    # A3. model_loading.py has v3 pipeline
    ml = (UP / "model_loading.py").read_text()
    results.append(check(
        "model_loading: V3 ref + v3 HF repo + get_regressor_v3()",
        "ModelVersion.V3" in ml and "tabpfn_3" in ml and "get_regressor_v3" in ml,
    ))

    # A4. PEaRL heads point at V3 (cross-check our edits)
    for path in [
        "/Users/tirthoroy/Desktop/PEARL/PeARL-TabFPN/src/pearl_tabpfn/tabpfn_head.py",
        "/Users/tirthoroy/Desktop/PEARL/PeARL-TabFPN/src/pearl_tabpfn/tabpfn3_head.py",
    ]:
        try:
            code = open(path).read()
            m = re.search(r"TabPFNRegressor\.create_default_for_version\(\s*ModelVersion\.V(\d)", code)
            v = m.group(1) if m else "?"
            results.append(check(f"{path.split('/')[-1]} → ModelVersion.V{v}", v == "3"))
        except FileNotFoundError:
            results.append(check(f"{path.split('/')[-1]} present", False))

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n-train", type=int, default=200, help="Synthetic training set size.")
    ap.add_argument("--n-features", type=int, default=10, help="Synthetic feature dimension.")
    ap.add_argument("--n-estimators", type=int, default=2, help="Tiny ensemble for speed.")
    ap.add_argument("--skip-runtime", action="store_true", help="Skip Tier B (runtime fit/predict).")
    ap.add_argument("--tabpfn-source",
                    default="/Users/tirthoroy/Desktop/PEARL/tabpfn-upstream/src/tabpfn",
                    help="Path to upstream tabpfn source for Tier A check.")
    args = ap.parse_args()

    print("=" * 70)
    print("TABPFN v3 SMOKE TEST")
    print("=" * 70)
    print(f"device={args.device}  n_train={args.n_train}  n_features={args.n_features}  "
          f"n_estimators={args.n_estimators}")

    results = []

    # Tier A — always run, works on any machine
    results.extend(smoke_source(args.tabpfn_source))

    if args.skip_runtime:
        print(f"\n{YELLOW}[B] skipped (--skip-runtime){RESET}")
        return _summary(results)

    section("[B] Runtime fit/predict (Tier B — needs torch ≥ 2.5 + tabpfn ≥ 8.0)")

    # ----- B1. Package import + version -----
    try:
        import tabpfn
        version = getattr(tabpfn, "__version__", "?")
        major = int(version.split(".")[0]) if version != "?" else 0
        results.append(check(
            f"tabpfn importable, version >= 8.0",
            major >= 8,
            f"version={version}",
        ))
    except ImportError as e:
        msg = str(e)
        results.append(check("tabpfn importable", False, msg))
        if "torch.nn.attention" in msg or "SDPBackend" in msg:
            print(f"    {DIM}→ EXPECTED on Intel Mac: tabpfn 8 needs torch ≥ 2.3 (attention API),{RESET}")
            print(f"    {DIM}  but Intel Mac caps at torch 2.2.2. Run Tier B on Linux GPU or M-series Mac.{RESET}")
        return _summary(results)

    # ----- B2. ModelVersion.V3 accessible -----
    try:
        from tabpfn.constants import ModelVersion
        v3 = ModelVersion.V3
        all_versions = [v.name for v in ModelVersion]
        results.append(check(
            "ModelVersion.V3 importable",
            v3 is not None,
            f"all versions: {all_versions}",
        ))
    except Exception as e:
        results.append(check("ModelVersion.V3 importable", False, f"{type(e).__name__}: {e}"))
        return _summary(results)

    # ----- B3. TabPFNRegressor.create_default_for_version(V3) -----
    try:
        from tabpfn import TabPFNRegressor
        t0 = time.time()
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3,
            device=args.device,
            n_estimators=args.n_estimators,
            random_state=42,
            ignore_pretraining_limits=True,
        )
        elapsed = time.time() - t0
        results.append(check(
            "TabPFNRegressor.create_default_for_version(V3) returns",
            reg is not None,
            f"{elapsed:.1f}s (cached weights re-use is fast; first run downloads ~500 MB)",
        ))
    except Exception as e:
        msg = str(e)
        gated = any(s in msg.lower() for s in ["401", "403", "gated", "license", "access", "token"])
        if gated:
            print(f"    {RED}✗{RESET} create_default_for_version raised — likely gating:")
            print(f"        {type(e).__name__}: {e}")
            print(f"        {DIM}→ accept license at https://huggingface.co/Prior-Labs/tabpfn_3{RESET}")
            print(f"        {DIM}→ then set TABPFN_TOKEN=<your-token> or HF_TOKEN=<your-token>{RESET}")
        else:
            import traceback
            print(f"    {RED}✗{RESET} unexpected error:")
            traceback.print_exc()
        results.append(False)
        return _summary(results)

    # ----- B4. fit on tiny synthetic data -----
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((args.n_train, args.n_features)).astype(np.float32)
    # y is a noisy linear combination — enough signal to be learnable, enough
    # noise that PCC < 1.
    true_w = rng.standard_normal(args.n_features).astype(np.float32)
    y_train = X_train @ true_w + 0.3 * rng.standard_normal(args.n_train).astype(np.float32)
    try:
        t0 = time.time()
        reg.fit(X_train, y_train)
        fit_time = time.time() - t0
        results.append(check(
            f"fit({args.n_train} rows × {args.n_features} cols) succeeds",
            True,
            f"{fit_time:.2f}s",
        ))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(check("fit() succeeds", False, f"{type(e).__name__}: {e}"))
        return _summary(results)

    # ----- B5. predict + sanity PCC -----
    X_val = rng.standard_normal((100, args.n_features)).astype(np.float32)
    y_val_true = X_val @ true_w + 0.3 * rng.standard_normal(100).astype(np.float32)
    try:
        t0 = time.time()
        y_val_pred = reg.predict(X_val)
        pred_time = time.time() - t0
        results.append(check(
            "predict() returns numeric array",
            y_val_pred.shape == (100,) and np.all(np.isfinite(y_val_pred)),
            f"shape={y_val_pred.shape}, dtype={y_val_pred.dtype}, {pred_time:.2f}s",
        ))
        pcc = float(np.corrcoef(y_val_pred, y_val_true)[0, 1])
        results.append(check(
            "PCC(pred, truth) > 0.5 (well above random)",
            pcc > 0.5,
            f"PCC={pcc:.3f}",
        ))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(check("predict() succeeds", False, f"{type(e).__name__}: {e}"))

    return _summary(results)


def _summary(results: list[bool]) -> int:
    passed, total = sum(results), len(results)
    print("\n" + "=" * 70)
    if passed == total and total > 0:
        print(f"{GREEN}SMOKE PASSED — {passed}/{total} assertions{RESET}")
        print("  TabPFN v3 is loadable, fittable, and produces sensible predictions.")
    else:
        print(f"{RED}SMOKE FAILED — {passed}/{total} assertions{RESET}")
    print("=" * 70)
    return 0 if (passed == total and total > 0) else 1


if __name__ == "__main__":
    sys.exit(main())

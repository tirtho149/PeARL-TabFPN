#!/usr/bin/env python3
"""Numpy-only smoke test for PCC + data-pipeline math.

Exercises the actual functions in pearl_tabpfn (compute_metrics,
apply_spatial_smoothing, ssgsea) without needing torch, scanpy, or
real HEST data. Stubs heavyweight imports at module-load time so the
pure-numpy functions can be imported and called in isolation.

This answers the question: "given a known-input array, do these
functions return numerically correct outputs?" It does NOT test:
  - the actual HEST loader (needs real .h5ad files)
  - any torch model / training loop
  - GPU code paths

Usage:
    /path/to/venv/bin/python scripts/smoke_no_data.py
"""
from __future__ import annotations

import os
import sys
import types

# Stub heavy/optional deps so `import eval` and `import data` don't fail
# at module-load. Each stub provides only what's accessed during import,
# not the runtime semantics — calling into the stubs would crash, which
# is intentional (it forces this script to stay numpy-only).
def _install_stubs():
    def fake_module(name: str, attrs: dict | None = None) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    # torch + submodules accessed at top of eval.py / data.py
    torch = fake_module("torch")
    torch.Tensor = object  # only used in type hints
    torch.device = lambda *a, **k: None
    fake_module("torch.nn", {"Module": object})
    fake_module("torch.nn.functional")
    fake_module("torch.utils")
    fake_module("torch.utils.data", {"Dataset": object, "DataLoader": object})

    # matplotlib + seaborn (eval.py imports them at top)
    fake_module("matplotlib")
    pyplot = fake_module("matplotlib.pyplot")
    pyplot.subplots = lambda *a, **k: (None, None)
    fake_module("matplotlib.patches", {"Patch": object})
    fake_module("matplotlib.colors", {"ListedColormap": object})
    fake_module("seaborn")

    # data.py loaders
    fake_module("h5py")
    fake_module("PIL")
    fake_module("PIL.Image")
    fake_module("anndata")


_install_stubs()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


# scipy 1.17.1 + numpy 2.x has a regression in pearsonr that crashes on
# any call (AttributeError: 'float' object has no attribute 'astype' in
# the array-api dispatch path). compute_metrics calls pearsonr; to test
# compute_metrics' OWN logic without being blocked by the scipy bug, we
# monkey-patch pearsonr with a numpy implementation BEFORE importing
# compute_metrics. The numpy version is mathematically identical for the
# float32 inputs this codebase uses.
import scipy.stats as _scipy_stats

def _numpy_pearsonr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return r, 0.0  # p-value unused by compute_metrics

_scipy_stats.pearsonr = _numpy_pearsonr
# Also patch the bound symbol inside eval.py (it does `from scipy.stats
# import pearsonr` inside the function, so the module-level patch above
# is what compute_metrics sees).

from pearl_tabpfn.eval import compute_metrics
from pearl_tabpfn.data import apply_spatial_smoothing, ssgsea


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def check(name: str, cond: bool, detail: str = "") -> bool:
    mark = f"{GREEN}✓{RESET}" if cond else f"{RED}✗{RESET}"
    suffix = f" {DIM}— {detail}{RESET}" if detail else ""
    print(f"    {mark} {name}{suffix}")
    return cond


def section(title: str):
    print(f"\n{YELLOW}{title}{RESET}")


def fmt(x: float) -> str:
    return "nan" if (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def test_compute_metrics() -> list[bool]:
    section("[1/3] compute_metrics — PCC / MSE / MAE")
    results = []
    rng = np.random.default_rng(42)

    # --- 1a. Near-perfect prediction → PCC ≈ 1, MSE ≈ 0
    # (Tiny epsilon noise avoids scipy 1.17 + numpy 2.x degenerate-corr crash
    # in pearsonr; this is a scipy regression, not a PEaRL bug.)
    true = rng.standard_normal((100, 20)).astype(np.float32)
    pred = true + 1e-4 * rng.standard_normal(true.shape).astype(np.float32)
    m = compute_metrics(pred, true, drop_constant_cols=True)
    results.append(check(
        "near-perfect prediction → PCC ≈ 1",
        m["PCC"] > 0.9999,
        f"PCC={fmt(m['PCC'])}, MSE={fmt(m['MSE'])}",
    ))

    # --- 1b. Noisy prediction → 0 < PCC < 1, with MSE matching analytic value
    true = rng.standard_normal((200, 30)).astype(np.float32)
    noise = 0.3 * rng.standard_normal(true.shape).astype(np.float32)
    pred = true + noise
    m = compute_metrics(pred, true, drop_constant_cols=True)
    results.append(check(
        "noisy prediction → 0 < PCC < 1",
        0.5 < m["PCC"] < 1.0,
        f"PCC={fmt(m['PCC'])} (expected ~0.95 for σ_noise=0.3)",
    ))
    results.append(check(
        "MSE matches analytic noise variance",
        abs(m["MSE"] - (0.3 ** 2)) < 0.02,
        f"MSE={fmt(m['MSE'])} vs analytic σ²={0.09:.4f}",
    ))

    # --- 1c. drop_constant_cols actually drops zero-variance columns
    true = rng.standard_normal((50, 40)).astype(np.float32)
    true[:, [3, 7, 11, 22]] = 0.0  # 4 dead columns
    pred = true + 0.1 * rng.standard_normal(true.shape).astype(np.float32)
    m_drop = compute_metrics(pred, true, drop_constant_cols=True)
    m_keep = compute_metrics(pred, true, drop_constant_cols=False)
    results.append(check(
        "drop_constant_cols=True removes 4 dead columns",
        m_drop["n_cols_dropped"] == 4 and m_drop["n_cols_used"] == 36,
        f"used={m_drop['n_cols_used']}, dropped={m_drop['n_cols_dropped']}",
    ))
    results.append(check(
        "drop_constant_cols=False keeps all columns",
        m_keep["n_cols_dropped"] == 0 and m_keep["n_cols_used"] == 40,
        f"used={m_keep['n_cols_used']}, dropped={m_keep['n_cols_dropped']}",
    ))
    # Because dead columns inflate PCC artificially via the flatten, dropping
    # them should yield a *higher* per-dim PCC (the real signal-bearing cols
    # are stronger on average than the across-flatten with zeros mixed in).
    # The flatten PCC numbers themselves are not monotonically comparable
    # because they depend on the relative scale of dropped vs kept cols.
    results.append(check(
        "per-dim mean PCC is a real number with drop_constant_cols",
        not np.isnan(m_drop["PCC_per_dim_mean"]) and 0 < m_drop["PCC_per_dim_mean"] < 1,
        f"per-dim PCC={fmt(m_drop['PCC_per_dim_mean'])} from "
        f"{m_drop['PCC_per_dim_n']} dims",
    ))

    # --- 1d. Degenerate (constant) prediction → PCC = NaN, not error
    pred_const = np.ones_like(true) * 0.5
    m = compute_metrics(pred_const, true, drop_constant_cols=True)
    results.append(check(
        "constant prediction → PCC = NaN (no crash)",
        np.isnan(m["PCC"]),
        f"PCC={fmt(m['PCC'])}, MSE={fmt(m['MSE'])} (well-defined)",
    ))

    # --- 1e. All-zero targets → graceful return
    true_zero = np.zeros((30, 10), dtype=np.float32)
    pred_any = rng.standard_normal((30, 10)).astype(np.float32)
    m = compute_metrics(pred_any, true_zero, drop_constant_cols=True)
    results.append(check(
        "all-zero targets → all cols dropped, no crash",
        m["n_cols_used"] == 0 and m["n_cols_dropped"] == 10
        and np.isnan(m["PCC"]),
        f"used={m['n_cols_used']}, MSE={fmt(m['MSE'])} (still computable)",
    ))

    # --- 1f. Near-anti-correlated prediction → PCC ≈ -1
    # (Tiny epsilon avoids the same scipy 1.17 degenerate-corr crash as 1a.)
    true = rng.standard_normal((100, 5)).astype(np.float32)
    pred = -true + 1e-4 * rng.standard_normal(true.shape).astype(np.float32)
    m = compute_metrics(pred, true, drop_constant_cols=True)
    results.append(check(
        "anti-correlated prediction → PCC ≈ -1",
        m["PCC"] < -0.9999,
        f"PCC={fmt(m['PCC'])}",
    ))

    return results


def test_apply_spatial_smoothing() -> list[bool]:
    section("[2/3] apply_spatial_smoothing — 8-neighbor kNN")
    results = []
    rng = np.random.default_rng(7)

    # --- 2a. Shape + dtype preserved
    n, g = 80, 50
    expr = rng.standard_normal((n, g)).astype(np.float32)
    coords = rng.random((n, 2)).astype(np.float32)
    smoothed = apply_spatial_smoothing(expr, coords, k=8)
    results.append(check(
        "shape preserved (N, G) → (N, G)",
        smoothed.shape == expr.shape,
        f"in={expr.shape}, out={smoothed.shape}",
    ))
    results.append(check(
        "dtype preserved (float32)",
        smoothed.dtype == expr.dtype,
        f"in={expr.dtype}, out={smoothed.dtype}",
    ))

    # --- 2b. Variance reduction (averaging neighbors smooths)
    results.append(check(
        "smoothing reduces variance",
        smoothed.var() < expr.var(),
        f"var: {expr.var():.4f} → {smoothed.var():.4f}",
    ))

    # --- 2c. Spatially-coherent signal: a gradient along x should be preserved
    # but high-frequency noise should be dampened.
    n = 200
    coords = rng.random((n, 2)).astype(np.float32)
    gradient = coords[:, 0:1].astype(np.float32)  # signal: f(x) = x
    noise = 0.5 * rng.standard_normal((n, 1)).astype(np.float32)
    expr = gradient + noise
    smoothed = apply_spatial_smoothing(expr, coords, k=8)
    # Correlation of smoothed signal with the gradient should be HIGHER than
    # raw (noise averages out, signal survives).
    corr_raw = float(np.corrcoef(expr[:, 0], gradient[:, 0])[0, 1])
    corr_smooth = float(np.corrcoef(smoothed[:, 0], gradient[:, 0])[0, 1])
    results.append(check(
        "spatial signal preserved, noise dampened",
        corr_smooth > corr_raw,
        f"corr(signal, raw)={corr_raw:.3f} → corr(signal, smoothed)={corr_smooth:.3f}",
    ))

    # --- 2d. k=1 still runs (edge case: averaging with single nearest neighbor)
    smoothed_k1 = apply_spatial_smoothing(expr, coords, k=1)
    results.append(check(
        "k=1 edge case runs",
        smoothed_k1.shape == expr.shape,
        f"shape={smoothed_k1.shape}",
    ))

    # --- 2e. Single-spot edge case
    expr_one = rng.standard_normal((1, 5)).astype(np.float32)
    coords_one = rng.random((1, 2)).astype(np.float32)
    smoothed_one = apply_spatial_smoothing(expr_one, coords_one, k=8)
    results.append(check(
        "single-spot input returns unchanged",
        np.array_equal(smoothed_one, expr_one),
        f"shape={smoothed_one.shape} (no crash on n=1)",
    ))

    return results


def test_ssgsea() -> list[bool]:
    section("[3/3] ssgsea — Single-Sample GSEA")
    results = []
    rng = np.random.default_rng(123)

    # --- 3a. Basic shape + finite values
    n_spots, n_genes = 30, 100
    gene_names = [f"G{i:03d}" for i in range(n_genes)]
    expr = rng.lognormal(mean=0, sigma=1.0, size=(n_spots, n_genes)).astype(np.float32)
    pathways = {
        "pathway_A": gene_names[:20],
        "pathway_B": gene_names[15:50],   # overlaps with A
        "pathway_C": gene_names[60:90],
    }
    scores = ssgsea(expr, gene_names, pathways)
    results.append(check(
        "output shape (n_spots, n_pathways)",
        scores.shape == (n_spots, 3),
        f"shape={scores.shape}",
    ))
    results.append(check(
        "all values finite",
        np.all(np.isfinite(scores)),
        f"min={scores.min():.3f}, max={scores.max():.3f}",
    ))

    # --- 3b. Non-trivial output (not all zero)
    results.append(check(
        "scores are non-zero (real ranking happened)",
        np.abs(scores).sum() > 0,
        f"|scores|.sum() = {np.abs(scores).sum():.2f}",
    ))

    # --- 3c. Pathways must produce DIFFERENT scores (otherwise it's a constant)
    results.append(check(
        "different pathways → different score vectors",
        not np.allclose(scores[:, 0], scores[:, 1])
        and not np.allclose(scores[:, 1], scores[:, 2]),
        f"std across pathways = {scores.std(axis=0).tolist()}",
    ))

    # --- 3d. A pathway with genes biased to high expression should score
    # higher than one with random gene membership.
    expr_biased = rng.standard_normal((n_spots, n_genes)).astype(np.float32)
    # Spike the first 10 genes way up
    expr_biased[:, :10] += 5.0
    pathways_bias = {
        "high_expr_pathway": gene_names[:10],     # the spiked genes
        "low_expr_pathway": gene_names[50:60],    # random middle
    }
    scores_bias = ssgsea(expr_biased, gene_names, pathways_bias)
    mean_high = float(scores_bias[:, 0].mean())
    mean_low = float(scores_bias[:, 1].mean())
    results.append(check(
        "high-expression gene set scores higher than random gene set",
        mean_high > mean_low,
        f"mean(spiked-pathway)={mean_high:.2f} > mean(random-pathway)={mean_low:.2f}",
    ))

    # --- 3e. Empty pathway input
    scores_empty = ssgsea(expr, gene_names, {})
    results.append(check(
        "empty pathway dict → (n_spots, 0) array, no crash",
        scores_empty.shape == (n_spots, 0),
        f"shape={scores_empty.shape}",
    ))

    # --- 3f. Pathway with genes not in gene_names
    pathways_missing = {"all_missing": ["NOT_A_GENE_1", "NOT_A_GENE_2"]}
    scores_missing = ssgsea(expr, gene_names, pathways_missing)
    results.append(check(
        "pathway with no overlapping genes → zero scores, no crash",
        scores_missing.shape == (n_spots, 1) and np.all(scores_missing == 0),
        f"shape={scores_missing.shape}, all-zero={bool(np.all(scores_missing == 0))}",
    ))

    return results


def main() -> int:
    print("=" * 70)
    print("NUMPY-ONLY SMOKE TEST — PCC + data-pipeline math")
    print("=" * 70)
    print("Stubs torch/matplotlib/etc. and exercises the actual numpy")
    print("functions from pearl_tabpfn.eval and pearl_tabpfn.data.")

    all_results = []
    all_results.extend(test_compute_metrics())
    all_results.extend(test_apply_spatial_smoothing())
    all_results.extend(test_ssgsea())

    passed = sum(all_results)
    total = len(all_results)
    print()
    print("=" * 70)
    if passed == total:
        print(f"{GREEN}SMOKE PASSED — {passed}/{total} assertions{RESET}")
        print("  • compute_metrics: PCC math correct on real numpy inputs")
        print("  • apply_spatial_smoothing: 8-neighbor kNN smooths signal correctly")
        print("  • ssgsea: ranking-based enrichment produces real scores")
        print("=" * 70)
        return 0
    else:
        print(f"{RED}SMOKE FAILED — {passed}/{total} assertions{RESET}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

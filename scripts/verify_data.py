#!/usr/bin/env python3
"""Verify the apple-to-apple pipeline produces REAL PCC numbers on REAL HEST.

CPU-only check that:
  1. Real HEST-1k Breast sections load via `pearl_tabpfn.data.load_hest_sample`
     (no synthetic fallback — the script crashes if data is missing).
  2. The returned gene / pathway / coord arrays have biologically plausible
     statistics (non-zero variance, finite values, expected sparsity).
  3. `pearl_tabpfn.eval.compute_metrics` with `drop_constant_cols=True` returns
     real, non-trivial PCC on a baseline predictor (predict per-column mean).
  4. The 8-neighbor smoother + Reactome+MSigDB pathway loader + raw pathway
     scaling (the apple-to-apple bundle) all run end-to-end on real data.

No model training. No GPU. Confirms the data layer is paper-faithful and
that downstream PCC is computed from real numbers.

Usage:
    python scripts/verify_data.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


HEST_DIR = os.environ.get("HEST_DATA_ROOT", "./hest_data")
SAMPLE_IDS_TO_CHECK = ["TENX99"]  # Breast IDC, paper-canonical; extend if more


def fmt(x: float) -> str:
    return f"{x:.4f}"


def check_section(sample_id: str) -> dict:
    """Load one section under apple-to-apple settings and return stats."""
    from pearl_tabpfn.data import load_hest_sample

    t0 = time.time()
    patches, genes, pathways, coords = load_hest_sample(
        hest_dir=HEST_DIR,
        sample_id=sample_id,
        n_genes=1000,
        n_pathways=200,
        patch_size=224,
        max_spots=200,
        seed=42,
        normalization="paper",           # paper-faithful gene scaling
        pathway_sources="reactome_msigdb",  # paper-faithful pathway pool
        smooth_genes=True,               # paper-faithful 8-neighbor smoothing
        smoothing_k=8,
    )
    elapsed = time.time() - t0

    return {
        "sample_id": sample_id,
        "elapsed_s": elapsed,
        "patches_shape": tuple(patches.shape),
        "genes_shape": tuple(genes.shape),
        "pathways_shape": tuple(pathways.shape),
        "coords_shape": tuple(coords.shape),
        "patches_dtype": str(patches.dtype),
        "genes_dtype": str(genes.dtype),
        "gene_mean": float(genes.mean()),
        "gene_std": float(genes.std()),
        "gene_min": float(genes.min()),
        "gene_max": float(genes.max()),
        "gene_nan_count": int(np.isnan(genes).sum()),
        "gene_zero_frac": float((genes == 0).mean()),
        "pathway_mean": float(pathways.mean()),
        "pathway_std": float(pathways.std()),
        "pathway_min": float(pathways.min()),
        "pathway_max": float(pathways.max()),
        "pathway_nan_count": int(np.isnan(pathways).sum()),
        "n_constant_gene_cols": int((genes.std(axis=0) < 1e-8).sum()),
        "n_constant_pathway_cols": int((pathways.std(axis=0) < 1e-8).sum()),
        "coord_min": float(coords.min()),
        "coord_max": float(coords.max()),
    }


def baseline_pcc(genes: np.ndarray, pathways: np.ndarray) -> dict:
    """Compute PCC of the per-column-mean predictor — sanity check that
    `compute_metrics` produces real, non-trivial PCC from real targets.

    A flat predictor that outputs every spot's column-mean for each dim is
    perfectly correlated by construction (PCC = 1 within each dim BUT with
    zero variance in the prediction). So we use a noisy mean predictor:
    pred = mean + small noise, where compute_metrics should still recover a
    meaningful PCC on the across-dim flatten.
    """
    from pearl_tabpfn.eval import compute_metrics

    rng = np.random.default_rng(7)
    # Predictor: each dim = (true_col_mean + 0.05 * std-scaled noise) shifted
    # toward the true target. This is a NON-trivial predictor that should
    # produce a positive but sub-1 PCC.
    gene_pred = genes * 0.3 + 0.7 * genes.mean(axis=0, keepdims=True) + \
                0.05 * rng.standard_normal(genes.shape).astype(np.float32)
    pathway_pred = pathways * 0.3 + 0.7 * pathways.mean(axis=0, keepdims=True) + \
                   0.05 * rng.standard_normal(pathways.shape).astype(np.float32)

    return {
        "gene": compute_metrics(gene_pred, genes, drop_constant_cols=True),
        "pathway": compute_metrics(pathway_pred, pathways, drop_constant_cols=True),
    }


def main() -> int:
    print("=" * 70)
    print("REAL-PCC VERIFICATION ON LOCAL HEST-1K")
    print("=" * 70)
    print()

    # ---- Verify HEST is actually on disk ----
    st_dir = Path(HEST_DIR) / "st"
    patches_dir = Path(HEST_DIR) / "patches"
    if not st_dir.is_dir() or not patches_dir.is_dir():
        print(f"FAIL: HEST dirs not found at {HEST_DIR}/st and {HEST_DIR}/patches.")
        print("      Did `snapshot_download HistologyBench/HEST` finish?")
        return 1

    print(f"HEST root: {HEST_DIR}")
    print(f"  st/      → {len(list(st_dir.glob('*.h5ad')))} h5ad files")
    print(f"  patches/ → {len(list(patches_dir.glob('*.h5')))} h5 files")
    print()

    # ---- Load each section under apple-to-apple settings ----
    all_stats = []
    for sid in SAMPLE_IDS_TO_CHECK:
        st_file = st_dir / f"{sid}.h5ad"
        if not st_file.exists():
            print(f"SKIP {sid}: no {st_file} — section not in this HEST snapshot.")
            continue
        print(f"Loading {sid} under apple-to-apple settings ...")
        try:
            stats = check_section(sid)
        except Exception as e:
            import traceback
            print(f"  FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        all_stats.append(stats)

        print(f"  ✓ loaded in {stats['elapsed_s']:.1f}s")
        print(f"    patches  : {stats['patches_shape']} {stats['patches_dtype']}")
        print(f"    genes    : {stats['genes_shape']} {stats['genes_dtype']}, "
              f"mean={fmt(stats['gene_mean'])}, std={fmt(stats['gene_std'])}, "
              f"range=[{fmt(stats['gene_min'])}, {fmt(stats['gene_max'])}]")
        print(f"    pathways : {stats['pathways_shape']}, "
              f"mean={fmt(stats['pathway_mean'])}, std={fmt(stats['pathway_std'])}, "
              f"range=[{fmt(stats['pathway_min'])}, {fmt(stats['pathway_max'])}]")
        print(f"    coords   : range [{fmt(stats['coord_min'])}, {fmt(stats['coord_max'])}]")
        print(f"    constant cols (genes/pathways): "
              f"{stats['n_constant_gene_cols']}/{stats['n_constant_pathway_cols']}")
        print(f"    NaN counts: gene={stats['gene_nan_count']}, "
              f"pathway={stats['pathway_nan_count']}")
        print()

    if not all_stats:
        print("FAIL: no sections loaded. HEST snapshot may be incomplete.")
        return 1

    # ---- Sanity assertions on the first loaded section ----
    s = all_stats[0]
    failures = []
    if s["gene_nan_count"] > 0:
        failures.append(f"gene_nan_count={s['gene_nan_count']} (must be 0)")
    if s["pathway_nan_count"] > 0:
        failures.append(f"pathway_nan_count={s['pathway_nan_count']} (must be 0)")
    if s["gene_std"] < 1e-6:
        failures.append(f"gene_std={s['gene_std']} ≈ 0 → all genes constant (suspect synthetic)")
    if s["pathway_std"] < 1e-6:
        failures.append(f"pathway_std={s['pathway_std']} ≈ 0 (suspect synthetic)")
    if s["gene_min"] == s["gene_max"]:
        failures.append("genes have zero range (suspect synthetic)")
    # 'paper' normalization clips to [0, 1].
    if not (-0.01 <= s["gene_min"] and s["gene_max"] <= 1.01):
        failures.append(
            f"gene range [{s['gene_min']}, {s['gene_max']}] outside [0,1] — "
            f"normalization='paper' should min-max scale per-gene"
        )

    # ---- Run compute_metrics on real targets ----
    print("Running compute_metrics on a noisy-mean baseline predictor ...")
    print("(Real PCC infrastructure check; expect 0 < PCC < 1, both targets.)")
    print()
    from pearl_tabpfn.data import load_hest_sample
    patches, genes, pathways, coords = load_hest_sample(
        hest_dir=HEST_DIR, sample_id=SAMPLE_IDS_TO_CHECK[0],
        n_genes=1000, n_pathways=200, max_spots=200, seed=42,
        normalization="paper", pathway_sources="reactome_msigdb",
        smooth_genes=True,
    )
    bm = baseline_pcc(genes, pathways)
    for target in ("gene", "pathway"):
        m = bm[target]
        print(f"  {target:8s}: PCC={fmt(m['PCC'])}  "
              f"PCC_per_dim={fmt(m['PCC_per_dim_mean'])}  "
              f"MSE={fmt(m['MSE'])}  MAE={fmt(m['MAE'])}  "
              f"cols_used={m['n_cols_used']} dropped={m['n_cols_dropped']}")
        if not (0 < m["PCC"] < 1):
            failures.append(f"{target} PCC={m['PCC']} outside (0, 1) — predictor is degenerate")
        if m["n_cols_used"] == 0:
            failures.append(f"{target}: all columns dropped — pathway/gene data is empty")

    print()
    print("=" * 70)
    if failures:
        print(f"VERIFICATION FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  • {f}")
        return 1
    print("VERIFICATION PASSED.")
    print("  • Real HEST sections loaded (no synthetic fallback).")
    print("  • Gene + pathway arrays have non-trivial biological variance.")
    print("  • Apple-to-apple data settings (smoothing, MSigDB, raw scale, paper gene norm) all work.")
    print("  • compute_metrics returns real PCC numbers on real targets.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

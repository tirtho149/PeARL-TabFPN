"""Stage 1 — export a PeARL HEST-1k cohort for the LA-3B regressor.

Reuses the corrected pooled pipeline so the LA model trains/evals on EXACTLY the
same cohort, targets, and 5-fold spot splits as PeARL — only the image encoder
differs downstream. Works for any cohort (breast | skin | lymph).

    python la_regression/export_dataset.py --cohort breast

Output: la_regression/la_dataset_<cohort>.npz with
    raw_patches (N,224,224,3) uint8 | genes (N,1000) | pathways (N,P) |
    section_ids (N,) | fold (N,)  [KFold shuffle seed=42, matches reproduction.py]

Run in the PeARL venv (has scanpy/anndata). GPU not required.
"""
import os
import sys
import argparse

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pearl_tabpfn.data import load_hest_multi_sample  # noqa: E402
from pearl_tabpfn.reproduction import select_cohort_section_ids, COHORTS  # noqa: E402

HERE = os.path.dirname(__file__)
HEST = os.path.join(HERE, "..", "hest_data")
META = os.path.join(HEST, "HEST_v1_1_0.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["breast", "skin", "lymph"], default="breast")
    args = ap.parse_args()
    spec = COHORTS[args.cohort]
    out = os.path.join(HERE, f"la_dataset_{args.cohort}.npz")

    ids = select_cohort_section_ids(META, args.cohort, seed=42)
    print(f"[{args.cohort}] selected {len(ids)} sections, n_pathways={spec['n_pathways']}")

    patches, genes, pathways, coords, section_ids, raw = load_hest_multi_sample(
        hest_dir=HEST, sample_ids=ids,
        n_genes=1000, n_pathways=spec["n_pathways"],
        max_spots_per_section=10 ** 9, normalization="paper", seed=42,
        pathway_sources="reactome_msigdb", pathway_normalization="minmax",
        smooth_genes=True, smoothing_k=8, min_spots_detected=1000,
        hvg_method="scanpy", return_raw_patches=True,
    )
    n = raw.shape[0]
    print(f"Pooled: {n} spots | genes {genes.shape} | pathways {pathways.shape} | raw {raw.shape}")

    fold = np.full(n, -1, dtype=np.int64)
    for fi, (_tr, val) in enumerate(KFold(5, shuffle=True, random_state=42).split(np.arange(n))):
        fold[val] = fi
    assert (fold >= 0).all()

    np.savez(out, raw_patches=raw, genes=genes.astype(np.float32),
             pathways=pathways.astype(np.float32),
             section_ids=section_ids.astype(np.int64), fold=fold)
    print(f"Saved {out} ({os.path.getsize(out)/1e9:.2f} GB); "
          f"fold sizes {[int((fold==i).sum()) for i in range(5)]}")


if __name__ == "__main__":
    main()

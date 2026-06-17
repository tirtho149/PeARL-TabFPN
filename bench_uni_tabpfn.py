"""
UNI -> TabPFN: minimal architecture, no training before TabPFN.

Pipeline:
  patches (cached) --> UNI (frozen) -> 1024-d raw features
                    +  K=32 spatial K-NN mean (per-section, by coords) -> 1024-d
                    -> concat -> 2048-d
                    -> PCA fit on train, project both to 256-d
                    -> TabPFN per output dim (n_estimators=1, context cap 4096)
                    -> gene_pred (1000) + pathway_pred (775)

No contrastive stage 1, no supervised stage 2, no MLP head, no refinement.
Just the encoder (UNI, frozen) + a deterministic feature reducer (PCA) +
TabPFN as the head.

Reuses the raw_features cache built by bench_spatial_raw.py if present.
Single 80/20 fold (same as the prior benches: KFold seed=42).

Cost estimate: per-dim TabPFN at ~2s, x 1,775 dims = ~1 hour per fold.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

from pearl_tabpfn.config import cfg
from pearl_tabpfn.data import HESTDataset, load_hest_multi_sample
from pearl_tabpfn.encoders import VisionEncoder
from pearl_tabpfn.eval import compute_metrics
from pearl_tabpfn.reproduction import extract_features, select_breast_section_ids


RAW_CACHE_KEYS = ["raw_features", "pathways", "genes", "coords", "section_ids"]


# ----------------------------------------------------------------------------
# Cache (raw UNI features + targets + coords)
# ----------------------------------------------------------------------------


def build_raw_cache(args, device, cache_path: str):
    print(f"\nNo raw-feature cache at {cache_path} -- building ...")
    sample_ids = select_breast_section_ids(args.metadata_csv, args.n_sections, seed=args.seed)
    t0 = time.time()
    patches, genes, pathways, coords, section_ids = load_hest_multi_sample(
        hest_dir=args.data_dir,
        sample_ids=sample_ids,
        n_genes=args.n_genes,
        n_pathways=args.n_pathways,
        max_spots_per_section=args.max_spots_per_section,
        normalization=args.normalization,
        seed=args.seed,
    )
    print(f"  data loaded in {time.time()-t0:.1f}s; patches={patches.shape}")

    t0 = time.time()
    encoder = VisionEncoder(
        embed_dim=cfg.EMBED_DIM, pretrained=True, backbone=args.encoder, freeze_backbone=True
    ).to(device)
    ds = HESTDataset(patches, genes, pathways, coords, sample_id="multi")
    features = extract_features(encoder, ds, device, args.feat_batch_size)
    del encoder
    torch.cuda.empty_cache()
    print(f"  features {tuple(features.shape)} in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        raw_features=features.numpy().astype(np.float32),
        pathways=pathways.astype(np.float32),
        genes=genes.astype(np.float32),
        coords=coords.astype(np.float64),
        section_ids=np.asarray(section_ids),
    )
    print(f"  cache saved ({os.path.getsize(cache_path)/1e6:.1f} MB)")


def load_raw_cache(cache_path: str):
    print(f"Loading raw-feature cache from {cache_path} ...")
    z = np.load(cache_path, allow_pickle=True)
    out = {k: z[k] for k in RAW_CACHE_KEYS}
    print(f"  raw_features {out['raw_features'].shape}")
    return out


# ----------------------------------------------------------------------------
# Per-section spatial K-NN mean (same as bench_spatial_raw.py)
# ----------------------------------------------------------------------------


def knn_neighbor_mean(
    features: np.ndarray, coords: np.ndarray, section_ids: np.ndarray, K: int
) -> np.ndarray:
    if K == 0:
        return features.copy()
    out = np.zeros_like(features)
    for sid in np.unique(section_ids):
        mask = section_ids == sid
        idx_in_section = np.where(mask)[0]
        sec_coords = coords[idx_in_section]
        sec_feat = features[idx_in_section]
        if len(idx_in_section) < 2:
            out[idx_in_section] = sec_feat
            continue
        tree = BallTree(sec_coords)
        k_eff = min(K + 1, len(sec_coords))
        _, knn = tree.query(sec_coords, k=k_eff)
        if k_eff > 1:
            out[idx_in_section] = sec_feat[knn[:, 1:]].mean(axis=1)
        else:
            out[idx_in_section] = sec_feat
    return out


# ----------------------------------------------------------------------------
# TabPFN per output dim
# ----------------------------------------------------------------------------


def fit_predict_tabpfn(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
    n_estimators: int, max_ctx: int | None, device: str, seed: int = 42,
    label: str = "",
) -> tuple[np.ndarray, float, float]:
    """For each output dim d: TabPFNRegressor.fit(X_train, y_train[:, d]) then
    predict(X_val). Returns (preds (n_val, n_dims), fit_time, predict_time).

    Subsamples X_train to max_ctx samples before each fit (single random sample
    shared across dims for determinism — TabPFN uses its own internal data each
    estimator, so this is just our context-cap policy).
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    n_dims = y_train.shape[1]
    n_train = X_train.shape[0]

    # Context cap: one subsampling, shared across dims.
    if max_ctx is not None and n_train > max_ctx:
        rng = np.random.default_rng(seed)
        ctx_idx = rng.choice(n_train, size=max_ctx, replace=False)
        X_ctx = X_train[ctx_idx]
        y_ctx = y_train[ctx_idx]
    else:
        X_ctx, y_ctx = X_train, y_train

    print(f"    {label}: TabPFN context X {X_ctx.shape}, predict on X_val {X_val.shape}, dims={n_dims}")

    preds = np.zeros((X_val.shape[0], n_dims), dtype=np.float32)
    t_fit_total = 0.0
    t_pred_total = 0.0
    log_every = max(1, n_dims // 20)
    t_start = time.time()
    for d in range(n_dims):
        # Use TabPFN v2 explicitly; v3 requires a license token (TABPFN_TOKEN)
        # and v2 is what the prior bench scripts used under tabpfn 7.1.1.
        r = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device=device,
            n_estimators=n_estimators,
            random_state=42 + d,
            ignore_pretraining_limits=True,
        )
        t0 = time.time()
        r.fit(X_ctx, y_ctx[:, d])
        t_fit_total += time.time() - t0
        t0 = time.time()
        preds[:, d] = r.predict(X_val)
        t_pred_total += time.time() - t0
        if (d + 1) % log_every == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (d + 1) * (n_dims - d - 1)
            print(f"      [{label}] dim {d+1}/{n_dims}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

    return preds, t_fit_total, t_pred_total


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./hest_data")
    p.add_argument("--metadata-csv", default="./hest_data/HEST_v1_1_0.csv")
    p.add_argument("--output-dir", default="./bench_uni_tabpfn_results")
    p.add_argument("--raw-cache-path", default="./bench_spatial_raw_results/raw_cache.npz",
                   help="Cache built by bench_spatial_raw.py; reused if present.")
    p.add_argument("--n-sections", type=int, default=36)
    p.add_argument("--max-spots-per-section", type=int, default=400)
    p.add_argument("--n-genes", type=int, default=1000)
    p.add_argument("--n-pathways", type=int, default=775)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--feat-batch-size", type=int, default=64)
    p.add_argument("--encoder", choices=["uni", "vit"], default="uni")
    p.add_argument("--normalization", choices=["paper", "paper_log1p_only", "paper_zscore"],
                   default="paper_log1p_only")
    p.add_argument("--K", type=int, default=32, help="Spatial K-NN neighbors")
    p.add_argument("--pca-dim", type=int, default=256)
    p.add_argument("--tabpfn-n-estimators", type=int, default=1)
    p.add_argument("--tabpfn-max-ctx", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny sanity check: K=8, PCA 64, top 5 dims, ctx 512.")
    p.add_argument("--force-rebuild", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.K = 8
        args.pca_dim = 64
        args.tabpfn_max_ctx = 512

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tabpfn_device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"TabPFN device: {tabpfn_device}; K={args.K}, PCA dim={args.pca_dim}, "
          f"n_est={args.tabpfn_n_estimators}, ctx_cap={args.tabpfn_max_ctx}")

    # ---- Load or build raw UNI features ----
    if not os.path.exists(args.raw_cache_path) or args.force_rebuild:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        build_raw_cache(args, device, args.raw_cache_path)
    raw = load_raw_cache(args.raw_cache_path)
    raw_features = raw["raw_features"]
    pathways = raw["pathways"]
    genes = raw["genes"]
    coords = raw["coords"]
    section_ids = np.asarray(raw["section_ids"])
    n = raw_features.shape[0]

    # ---- Spatial K-NN aggregation ----
    t0 = time.time()
    neighbor_means = knn_neighbor_mean(raw_features, coords, section_ids, args.K)
    if args.K == 0:
        aug_features = raw_features
    else:
        aug_features = np.concatenate([raw_features, neighbor_means], axis=1).astype(np.float32)
    t_knn = time.time() - t0
    print(f"K={args.K} aug features: {aug_features.shape} in {t_knn:.1f}s")

    # ---- Fold split (same as prior benches) ----
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(iter(kf.split(np.arange(n))))
    print(f"Fold: train={len(train_idx)}, val={len(val_idx)}")

    X_train_aug = aug_features[train_idx]
    X_val_aug = aug_features[val_idx]
    yp_train = pathways[train_idx]; yp_val = pathways[val_idx]
    yg_train = genes[train_idx];    yg_val = genes[val_idx]

    # ---- PCA fit on train, apply to both ----
    t0 = time.time()
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    X_train = pca.fit_transform(X_train_aug).astype(np.float32)
    X_val = pca.transform(X_val_aug).astype(np.float32)
    t_pca = time.time() - t0
    var_explained = float(pca.explained_variance_ratio_.sum())
    print(f"PCA: train {X_train_aug.shape} -> {X_train.shape} in {t_pca:.1f}s "
          f"(explained variance: {var_explained:.3f})")

    if args.smoke:
        print("\nSMOKE: keeping top 5 pathway and 5 gene dims only.")
        yp_train = yp_train[:, :5]; yp_val = yp_val[:, :5]
        yg_train = yg_train[:, :5]; yg_val = yg_val[:, :5]

    # ---- Pathway TabPFN ----
    print(f"\n=== Pathway TabPFN ({yp_train.shape[1]} dims) ===")
    pp_pred, p_fit_t, p_pred_t = fit_predict_tabpfn(
        X_train, yp_train, X_val,
        n_estimators=args.tabpfn_n_estimators,
        max_ctx=args.tabpfn_max_ctx,
        device=tabpfn_device, seed=args.seed, label="pathway",
    )
    print(f"  pathway: fit={p_fit_t:.1f}s, predict={p_pred_t:.1f}s")

    # ---- Gene TabPFN ----
    print(f"\n=== Gene TabPFN ({yg_train.shape[1]} dims) ===")
    gp_pred, g_fit_t, g_pred_t = fit_predict_tabpfn(
        X_train, yg_train, X_val,
        n_estimators=args.tabpfn_n_estimators,
        max_ctx=args.tabpfn_max_ctx,
        device=tabpfn_device, seed=args.seed, label="gene",
    )
    print(f"  gene: fit={g_fit_t:.1f}s, predict={g_pred_t:.1f}s")

    # ---- Metrics ----
    pwy_metrics = compute_metrics(pp_pred, yp_val)
    gn_metrics = compute_metrics(gp_pred, yg_val)
    print(f"\nResults:")
    print(f"  gene    PCC={gn_metrics['PCC']:.4f}  MSE={gn_metrics['MSE']:.4f}  MAE={gn_metrics['MAE']:.4f}")
    print(f"  pathway PCC={pwy_metrics['PCC']:.4f}  MSE={pwy_metrics['MSE']:.4f}  MAE={pwy_metrics['MAE']:.4f}")

    # ---- Save + report ----
    out = {
        "args": vars(args),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "K": int(args.K),
        "pca_dim": int(args.pca_dim),
        "pca_variance_explained": var_explained,
        "tabpfn_n_estimators": int(args.tabpfn_n_estimators),
        "tabpfn_max_ctx": int(args.tabpfn_max_ctx),
        "t_knn_s": float(t_knn),
        "t_pca_s": float(t_pca),
        "t_pathway_fit_s": float(p_fit_t),
        "t_pathway_predict_s": float(p_pred_t),
        "t_gene_fit_s": float(g_fit_t),
        "t_gene_predict_s": float(g_pred_t),
        "t_total_s": float(t_knn + t_pca + p_fit_t + p_pred_t + g_fit_t + g_pred_t),
        "pathway": pwy_metrics,
        "gene": gn_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "bench_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    render_report(out, os.path.join(args.output_dir, "BENCHMARK_REPORT.md"))
    print(f"\nResults saved to {args.output_dir}/")


def render_report(out: Dict, path: str):
    lines = []
    lines.append("# UNI -> K-NN -> PCA -> TabPFN")
    lines.append("")
    lines.append("Minimal architecture: no contrastive pretraining, no MLP head, no refinement.")
    lines.append("Just frozen UNI features, spatial K-NN aggregation (per-section), PCA fit on train,")
    lines.append("TabPFN per output dim.")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append(f"- K (spatial K-NN neighbors): {out['K']}")
    lines.append(f"- PCA dim: {out['pca_dim']} (explained variance: {out['pca_variance_explained']:.3f})")
    lines.append(f"- TabPFN n_estimators: {out['tabpfn_n_estimators']}")
    lines.append(f"- TabPFN max context: {out['tabpfn_max_ctx']}")
    lines.append(f"- n_train: {out['n_train']}, n_val: {out['n_val']}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Target | PCC | MSE | MAE |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| gene    | {out['gene']['PCC']:.4f} | {out['gene']['MSE']:.4f} | {out['gene']['MAE']:.4f} |"
    )
    lines.append(
        f"| pathway | {out['pathway']['PCC']:.4f} | {out['pathway']['MSE']:.4f} | {out['pathway']['MAE']:.4f} |"
    )
    lines.append("")
    lines.append("## Reference: prior best on the same fold")
    lines.append("")
    lines.append("| Method | gene PCC | pathway PCC |")
    lines.append("|---|---:|---:|")
    lines.append("| MLP-only baseline (PEaRL + 2-stage training) | 0.7596 | 0.6684 |")
    lines.append("| **+ raw K-NN K=32 + MLP** (prior best) | **0.7621** | **0.6755** |")
    lines.append(
        f"| **UNI -> K-NN -> PCA -> TabPFN (this run)** | **{out['gene']['PCC']:.4f}** | **{out['pathway']['PCC']:.4f}** |"
    )
    lines.append("")
    lines.append("## Time breakdown")
    lines.append("")
    lines.append("| Stage | Seconds |")
    lines.append("|---|---:|")
    lines.append(f"| Spatial K-NN aggregation | {out['t_knn_s']:.1f} |")
    lines.append(f"| PCA fit + transform | {out['t_pca_s']:.1f} |")
    lines.append(f"| TabPFN fit (pathway) | {out['t_pathway_fit_s']:.1f} |")
    lines.append(f"| TabPFN predict (pathway) | {out['t_pathway_predict_s']:.1f} |")
    lines.append(f"| TabPFN fit (gene) | {out['t_gene_fit_s']:.1f} |")
    lines.append(f"| TabPFN predict (gene) | {out['t_gene_predict_s']:.1f} |")
    lines.append(f"| **Total** | **{out['t_total_s']:.1f}** |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

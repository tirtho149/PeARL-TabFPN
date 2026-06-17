"""
5-fold CV confirmation of raw-space spatial K-NN at K=32.

bench_spatial_raw.py established the single-fold result (gene PCC +0.0025,
pathway PCC +0.0071 at K=32 vs K=0). This script runs the SAME pipeline
across all 5 folds with K in {0, 32} only, to confirm the result isn't
fold-luck.

Pipeline per fold (identical to bench_spatial_raw.py):

  raw_uni -> [optional K-NN concat] -> PEaRLCached(feat_dim)
    Stage 1 contrastive (NT-Xent) -> trains feat_proj + pathway_encoder
    Stage 2 supervised (MSE)      -> trains MLP heads, encoders frozen
    Evaluate (gene PCC, pathway PCC, MSE, MAE)

Reuses raw_features cache from bench_spatial_raw_results. Reports per-fold
metrics plus mean ± std for each K, plus the K=32 - K=0 delta with the
fold-paired standard error.
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
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

from pearl_tabpfn.config import cfg
from pearl_tabpfn.data import HESTDataset, load_hest_multi_sample
from pearl_tabpfn.encoders import VisionEncoder
from pearl_tabpfn.eval import compute_metrics
from pearl_tabpfn.reproduction import (
    PEaRLCached,
    extract_features,
    make_tensor_loader,
    select_breast_section_ids,
    stage1_contrastive,
    stage2_supervised,
)
from bench_lightgbm import collect_arrays


RAW_CACHE_KEYS = ["raw_features", "pathways", "genes", "coords", "section_ids"]


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


def load_raw_cache(cache_path: str):
    print(f"Loading raw-feature cache from {cache_path} ...")
    z = np.load(cache_path, allow_pickle=True)
    return {k: z[k] for k in RAW_CACHE_KEYS}


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
# Per-fold runner
# ----------------------------------------------------------------------------


def run_one_fold_one_K(
    K: int, fold_idx: int, raw: Dict, args, device, train_idx, val_idx,
) -> Dict:
    print(f"\n--- Fold {fold_idx+1}/{args.folds} | K={K} | train={len(train_idx)} val={len(val_idx)} ---")
    raw_feats = raw["raw_features"]
    pathways = raw["pathways"]
    genes = raw["genes"]
    coords = raw["coords"]
    section_ids = np.asarray(raw["section_ids"])

    t0 = time.time()
    if K == 0:
        aug_features = raw_feats.astype(np.float32)
    else:
        neighbor_means = knn_neighbor_mean(raw_feats, coords, section_ids, K)
        aug_features = np.concatenate([raw_feats, neighbor_means], axis=1).astype(np.float32)
    t_agg = time.time() - t0
    feat_dim = aug_features.shape[1]

    features_t = torch.from_numpy(aug_features)
    pathways_t = torch.from_numpy(pathways)
    genes_t = torch.from_numpy(genes)
    coords_t = torch.from_numpy(coords.astype(np.float32))

    train_loader = make_tensor_loader(
        features_t, pathways_t, genes_t, coords_t, train_idx, args.batch_size, True
    )
    val_loader = make_tensor_loader(
        features_t, pathways_t, genes_t, coords_t, val_idx, args.batch_size, False
    )

    # Fresh model per fold so seeds match the single-fold bench
    torch.manual_seed(args.seed + fold_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + fold_idx)
    model = PEaRLCached(
        feat_dim=feat_dim,
        n_pathways=pathways.shape[1],
        n_genes=genes.shape[1],
        embed_dim=cfg.EMBED_DIM,
        pathway_hidden=cfg.PATHWAY_HIDDEN,
        head_type="mlp",
    ).to(device)

    t0 = time.time()
    stage1_contrastive(
        model, train_loader, val_loader, device,
        args.epochs_stage1, args.patience, args.lr, args.weight_decay, args.temperature,
    )
    t_stage1 = time.time() - t0

    t0 = time.time()
    stage2_supervised(
        model, train_loader, val_loader, device,
        args.epochs_stage2, args.patience, args.lr, args.weight_decay, "mlp",
    )
    t_stage2 = time.time() - t0

    model.eval()
    _, _, _, val_mlp_pp, val_mlp_gp = collect_arrays(model, val_loader, device)
    pwy = compute_metrics(val_mlp_pp, pathways[val_idx])
    gn = compute_metrics(val_mlp_gp, genes[val_idx])
    print(f"  stage1={t_stage1:.0f}s stage2={t_stage2:.0f}s | "
          f"gene PCC={gn['PCC']:.4f}, pathway PCC={pwy['PCC']:.4f}")

    return {
        "fold": int(fold_idx),
        "K": int(K),
        "feat_dim": int(feat_dim),
        "t_aggregate_s": float(t_agg),
        "t_stage1_s": float(t_stage1),
        "t_stage2_s": float(t_stage2),
        "pathway": pwy,
        "gene": gn,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./hest_data")
    p.add_argument("--metadata-csv", default="./hest_data/HEST_v1_1_0.csv")
    p.add_argument("--output-dir", default="./bench_spatial_raw_cv_results")
    p.add_argument("--raw-cache-path", default="./bench_spatial_raw_results/raw_cache.npz")
    p.add_argument("--n-sections", type=int, default=36)
    p.add_argument("--max-spots-per-section", type=int, default=400)
    p.add_argument("--n-genes", type=int, default=1000)
    p.add_argument("--n-pathways", type=int, default=775)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs-stage1", type=int, default=100)
    p.add_argument("--epochs-stage2", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--feat-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--encoder", choices=["uni", "vit"], default="uni")
    p.add_argument("--normalization", choices=["paper", "paper_log1p_only", "paper_zscore"],
                   default="paper_log1p_only")
    p.add_argument("--K-list", type=str, default="0,32",
                   help="Comma-separated K values to evaluate across all folds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-rebuild", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    K_values = [int(k.strip()) for k in args.K_list.split(",")]
    print(f"Device: {device}; folds={args.folds}; K values={K_values}")

    if not os.path.exists(args.raw_cache_path) or args.force_rebuild:
        build_raw_cache(args, device, args.raw_cache_path)
    raw = load_raw_cache(args.raw_cache_path)
    n = raw["raw_features"].shape[0]
    print(f"  raw_features {raw['raw_features'].shape}")

    # Same KFold seed as the single-fold benches; fold 0 here reproduces those.
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_splits = list(kf.split(np.arange(n)))

    results = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        for K in K_values:
            r = run_one_fold_one_K(K, fold_idx, raw, args, device, train_idx, val_idx)
            results.append(r)
            with open(os.path.join(args.output_dir, "cv_results.json"), "w") as f:
                json.dump({
                    "args": vars(args),
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2, default=str)

    render_report(results, K_values, args.folds, os.path.join(args.output_dir, "BENCHMARK_REPORT.md"))
    print(f"\nResults saved to {args.output_dir}/")


def aggregate(results: List[Dict], K: int, n_folds: int) -> Dict:
    """Return mean ± std of gene PCC and pathway PCC across folds for a given K."""
    rows = [r for r in results if r["K"] == K]
    rows.sort(key=lambda x: x["fold"])
    gene_pccs = np.array([r["gene"]["PCC"] for r in rows], dtype=np.float64)
    pwy_pccs = np.array([r["pathway"]["PCC"] for r in rows], dtype=np.float64)
    return {
        "K": K,
        "n_folds": len(rows),
        "gene_pcc_per_fold": gene_pccs.tolist(),
        "pathway_pcc_per_fold": pwy_pccs.tolist(),
        "gene_pcc_mean": float(np.mean(gene_pccs)),
        "gene_pcc_std": float(np.std(gene_pccs, ddof=1)) if len(gene_pccs) > 1 else 0.0,
        "pathway_pcc_mean": float(np.mean(pwy_pccs)),
        "pathway_pcc_std": float(np.std(pwy_pccs, ddof=1)) if len(pwy_pccs) > 1 else 0.0,
    }


def render_report(results: List[Dict], K_values: List[int], n_folds: int, path: str):
    aggs = {K: aggregate(results, K, n_folds) for K in K_values}

    lines = []
    lines.append("# Spatial K-NN in Raw UNI Space — 5-fold CV confirmation")
    lines.append("")
    lines.append(f"Same fold split (KFold seed=42, n_splits={n_folds}) as the single-fold benches.")
    lines.append("Fold 0 reproduces the bench_spatial_raw single-fold numbers; folds 1-4 are new.")
    lines.append("Each row trains PEaRLCached from scratch (stage 1 + stage 2) on")
    lines.append("`concat(raw_uni_self, mean(raw_uni_K_neighbors))`. K=0 is the original pipeline.")
    lines.append("")
    lines.append("## Per-fold metrics")
    lines.append("")
    lines.append("| Fold | K | gene PCC | pathway PCC |")
    lines.append("|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda x: (x["fold"], x["K"])):
        lines.append(
            f"| {r['fold']+1} | {r['K']} | {r['gene']['PCC']:.4f} | {r['pathway']['PCC']:.4f} |"
        )
    lines.append("")
    lines.append("## Aggregated (mean ± std across folds)")
    lines.append("")
    lines.append("| K | gene PCC | pathway PCC |")
    lines.append("|---:|---|---|")
    for K in K_values:
        a = aggs[K]
        lines.append(
            f"| {K} | {a['gene_pcc_mean']:.4f} ± {a['gene_pcc_std']:.4f} | "
            f"{a['pathway_pcc_mean']:.4f} ± {a['pathway_pcc_std']:.4f} |"
        )
    lines.append("")

    # If exactly two K values, compute fold-paired delta
    if len(K_values) == 2 and 0 in K_values:
        K_other = [k for k in K_values if k != 0][0]
        rows_0 = sorted([r for r in results if r["K"] == 0], key=lambda x: x["fold"])
        rows_other = sorted([r for r in results if r["K"] == K_other], key=lambda x: x["fold"])
        gene_diffs = np.array([
            ro["gene"]["PCC"] - r0["gene"]["PCC"]
            for r0, ro in zip(rows_0, rows_other)
        ], dtype=np.float64)
        pwy_diffs = np.array([
            ro["pathway"]["PCC"] - r0["pathway"]["PCC"]
            for r0, ro in zip(rows_0, rows_other)
        ], dtype=np.float64)
        gm, gs = float(np.mean(gene_diffs)), float(np.std(gene_diffs, ddof=1)) if len(gene_diffs) > 1 else 0.0
        pm, ps = float(np.mean(pwy_diffs)), float(np.std(pwy_diffs, ddof=1)) if len(pwy_diffs) > 1 else 0.0
        # paired-sample SE
        gse = gs / np.sqrt(max(len(gene_diffs), 1))
        pse = ps / np.sqrt(max(len(pwy_diffs), 1))
        lines.append(f"## Fold-paired delta (K={K_other} - K=0)")
        lines.append("")
        lines.append("| Target | mean Δ PCC | std Δ | SE | per-fold Δ |")
        lines.append("|---|---:|---:|---:|---|")
        lines.append(
            f"| gene | {gm:+.4f} | {gs:.4f} | {gse:.4f} | "
            + " ".join(f"{d:+.4f}" for d in gene_diffs) + " |"
        )
        lines.append(
            f"| pathway | {pm:+.4f} | {ps:.4f} | {pse:.4f} | "
            + " ".join(f"{d:+.4f}" for d in pwy_diffs) + " |"
        )
        lines.append("")
        # Headline interpretation
        gene_sig = "✓ confirmed" if gm > 0 and gm - gse > 0 else "✗ within noise"
        pwy_sig  = "✓ confirmed" if pm > 0 and pm - pse > 0 else "✗ within noise"
        lines.append(f"- gene: {gene_sig} (mean +{gm:.4f} ± SE {gse:.4f})")
        lines.append(f"- pathway: {pwy_sig} (mean +{pm:.4f} ± SE {pse:.4f})")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

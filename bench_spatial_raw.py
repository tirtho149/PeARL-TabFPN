"""
K-NN spatial augmentation in RAW UNI feature space (vs projected space).

The earlier bench_spatial.py did K-NN on the 256-d projected embedding —
that's smoothing a representation the projection already learned to make
discriminative. This script does K-NN BEFORE the projection: on the raw
1024-d UNI features. The trained projection then gets to learn from the
aggregated signal, which is the standard pattern in ST literature.

For each K in {0, 4, 8, 16, 32}:
  raw_feat (1024) ──┐
                    ├── concat ── feat_proj ── ... heads
  knn_mean (1024) ──┘   (2048)    Linear(2048→256)

K=0 is the original pipeline (no concat, feat_proj input dim = 1024).

Wall time: ~3 min feature extraction (cached after first run) + ~2.5 min
per K (stage 1 contrastive + stage 2 supervised on cached features).
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
from pearl_tabpfn.eval import compute_metrics
from pearl_tabpfn.encoders import VisionEncoder
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
    print(f"\nNo raw-feature cache at {cache_path} — building ...")
    sample_ids = select_breast_section_ids(args.metadata_csv, args.n_sections, seed=args.seed)
    print(f"  selected {len(sample_ids)} sections")

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
    print(f"  raw cache saved to {cache_path} ({os.path.getsize(cache_path)/1e6:.1f} MB)")


def load_raw_cache(cache_path: str):
    print(f"Loading raw-feature cache from {cache_path} ...")
    z = np.load(cache_path, allow_pickle=True)
    out = {k: z[k] for k in RAW_CACHE_KEYS}
    print(f"  raw_features {out['raw_features'].shape}, "
          f"pathways {out['pathways'].shape}, genes {out['genes'].shape}")
    return out


# ----------------------------------------------------------------------------
# Spatial K-NN in raw space
# ----------------------------------------------------------------------------


def knn_neighbor_mean(
    features: np.ndarray, coords: np.ndarray, section_ids: np.ndarray, K: int
) -> np.ndarray:
    """Per-section K-NN by Euclidean coord distance. Returns mean of K nearest
    neighbors (self excluded). For K=0 returns features unchanged.
    """
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
# Train one K configuration
# ----------------------------------------------------------------------------


def run_one_K(K: int, raw: Dict, args, device, train_idx, val_idx) -> Dict:
    print(f"\n=== K={K} ===")

    raw_feats = raw["raw_features"]
    pathways = raw["pathways"]
    genes = raw["genes"]
    coords = raw["coords"]
    section_ids = np.asarray(raw["section_ids"])

    t0 = time.time()
    neighbor_means = knn_neighbor_mean(raw_feats, coords, section_ids, K)
    if K == 0:
        aug_features = raw_feats.astype(np.float32)
    else:
        aug_features = np.concatenate([raw_feats, neighbor_means], axis=1).astype(np.float32)
    t_aggregate = time.time() - t0
    feat_dim = aug_features.shape[1]
    print(f"  aug features: shape {aug_features.shape}, aggregated in {t_aggregate:.1f}s")

    # Tensors
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
    print(f"  stage 1 done in {t_stage1:.1f}s")

    t0 = time.time()
    stage2_supervised(
        model, train_loader, val_loader, device,
        args.epochs_stage2, args.patience, args.lr, args.weight_decay, "mlp",
    )
    t_stage2 = time.time() - t0
    print(f"  stage 2 done in {t_stage2:.1f}s")

    # Evaluate
    model.eval()
    _, _, _, val_mlp_pp, val_mlp_gp = collect_arrays(model, val_loader, device)
    pwy = compute_metrics(val_mlp_pp, pathways[val_idx])
    gn = compute_metrics(val_mlp_gp, genes[val_idx])
    print(f"  gene PCC={gn['PCC']:.4f}, pathway PCC={pwy['PCC']:.4f}")

    return {
        "K": K,
        "feat_dim": int(feat_dim),
        "t_aggregate_s": float(t_aggregate),
        "t_stage1_s": float(t_stage1),
        "t_stage2_s": float(t_stage2),
        "t_total_s": float(t_aggregate + t_stage1 + t_stage2),
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
    p.add_argument("--output-dir", default="./bench_spatial_raw_results")
    p.add_argument("--raw-cache-path", default="./bench_spatial_raw_results/raw_cache.npz")
    p.add_argument("--n-sections", type=int, default=36)
    p.add_argument("--max-spots-per-section", type=int, default=400)
    p.add_argument("--n-genes", type=int, default=1000)
    p.add_argument("--n-pathways", type=int, default=775)
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
    p.add_argument("--K-list", type=str, default="0,4,8,16,32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-rebuild", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    K_values = [int(k.strip()) for k in args.K_list.split(",")]
    print(f"Device: {device}; K values: {K_values}")

    if not os.path.exists(args.raw_cache_path) or args.force_rebuild:
        build_raw_cache(args, device, args.raw_cache_path)
    raw = load_raw_cache(args.raw_cache_path)
    n = raw["raw_features"].shape[0]

    # Same KFold split as the prior benches
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(iter(kf.split(np.arange(n))))
    print(f"Fold: train={len(train_idx)}, val={len(val_idx)}")

    results = []
    for K in K_values:
        r = run_one_K(K, raw, args, device, train_idx, val_idx)
        results.append(r)
        with open(os.path.join(args.output_dir, "spatial_raw_results.json"), "w") as f:
            json.dump({
                "args": vars(args),
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, default=str)

    render_report(results, os.path.join(args.output_dir, "BENCHMARK_REPORT.md"))
    print(f"\nResults saved to {args.output_dir}/")


def render_report(results, path):
    baseline = next((r for r in results if r["K"] == 0), None)
    lines = []
    lines.append("# Spatial K-NN in Raw UNI Feature Space")
    lines.append("")
    lines.append("Each row trains a fresh PEaRLCached (stage 1 contrastive + stage 2 supervised) on")
    lines.append("`concat(raw_uni_self, mean(raw_uni_K_nearest_neighbors))`. The projection layer gets")
    lines.append("to learn from the aggregated signal — different from the projected-space K-NN which")
    lines.append("smoothed an already-learned representation.")
    lines.append("")
    lines.append("Neighbors are picked by Euclidean distance in pixel coords, restricted to the same HEST section.")
    lines.append("K=0 is the original pipeline (no augmentation; feat_proj input dim 1024).")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| K | feat_dim | gene PCC | Δ gene vs K=0 | pwy PCC | Δ pwy vs K=0 | gene MSE | gene MAE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        dg = r["gene"]["PCC"] - baseline["gene"]["PCC"] if baseline else 0.0
        dp = r["pathway"]["PCC"] - baseline["pathway"]["PCC"] if baseline else 0.0
        lines.append(
            f"| {r['K']} | {r['feat_dim']} | {r['gene']['PCC']:.4f} | {dg:+.4f} | "
            f"{r['pathway']['PCC']:.4f} | {dp:+.4f} | {r['gene']['MSE']:.4f} | {r['gene']['MAE']:.4f} |"
        )
    lines.append("")
    lines.append("## Time")
    lines.append("")
    lines.append("| K | aggregate | stage 1 | stage 2 | total |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['K']} | {r['t_aggregate_s']:.1f}s | {r['t_stage1_s']:.1f}s | "
            f"{r['t_stage2_s']:.1f}s | {r['t_total_s']:.1f}s |"
        )
    lines.append("")
    if baseline:
        best = max(results, key=lambda r: r["gene"]["PCC"])
        if best["K"] != 0:
            lines.append(
                f"**Best gene PCC**: K={best['K']} → {best['gene']['PCC']:.4f} "
                f"(Δ {best['gene']['PCC']-baseline['gene']['PCC']:+.4f} vs K=0). "
            )
        else:
            lines.append(
                f"**K=0 wins.** Spatial aggregation in raw UNI feature space does not improve gene PCC "
                f"on this setup. All K > 0: Δ ≤ 0 vs K=0."
            )

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

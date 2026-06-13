"""
Spatial K-NN augmentation benchmark.

Hypothesis: gene expression is spatially smooth — adjacent spots co-vary.
The MLP head currently sees one spot at a time and ignores 2D coords at
inference. Augmenting each spot's embedding with the mean of its K nearest
neighbors' embeddings should add information the head literally has but
doesn't use.

For each K in {0, 4, 8, 16, 32}: BallTree K-NN on coords, augment feature
as concat(self_emb, mean(neighbor_embs)), train a fresh MLP head on the
augmented feature, evaluate on the same val fold as the prior benches.

K=0 doubles as a pipeline sanity check (no augmentation → should match the
cached MLP baseline within stochastic-training noise).

K-NN neighbor selection uses ALL spots' coords + embeddings. No label
leakage: embeddings are deterministic functions of patches; only train-set
labels are used to fit the head.
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
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

from pearl_data import load_hest_multi_sample
from pearl_eval import compute_metrics
from run_paper_reproduction import select_breast_section_ids


CACHE_KEYS = [
    "X_tr", "yp_tr", "yg_tr", "mlp_pp_tr", "mlp_gp_tr",
    "val_embeds", "val_mlp_pp", "val_mlp_gp", "val_yp", "val_yg",
]


# ----------------------------------------------------------------------------
# Data loading: cache (embeddings + targets) + fresh HEST load (coords only)
# ----------------------------------------------------------------------------


def load_cache(path: str) -> Dict[str, np.ndarray]:
    print(f"Loading cache from {path} ...")
    z = np.load(path)
    cache = {k: z[k] for k in CACHE_KEYS}
    print(f"  X_tr {cache['X_tr'].shape}, val_embeds {cache['val_embeds'].shape}")
    return cache


def load_coords_and_fold(args, n_expected: int):
    """Load HEST data just to get coords + section_ids. Recompute the same
    KFold split the cache was built with so train/val indices align with the
    cached X_tr / val_embeds rows."""
    sample_ids = select_breast_section_ids(args.metadata_csv, args.n_sections, seed=args.seed)
    print(f"Loading HEST data for coords ({len(sample_ids)} sections) ...")
    t0 = time.time()
    _patches, _genes, _pathways, coords, section_ids = load_hest_multi_sample(
        hest_dir=args.data_dir,
        sample_ids=sample_ids,
        n_genes=args.n_genes,
        n_pathways=args.n_pathways,
        max_spots_per_section=args.max_spots_per_section,
        normalization=args.normalization,
        seed=args.seed,
    )
    print(f"  data loaded in {time.time()-t0:.1f}s; N={len(coords)}")
    assert len(coords) == n_expected, f"coord N {len(coords)} != cache N {n_expected}"
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(iter(kf.split(np.arange(len(coords)))))
    print(f"  fold split: train={len(train_idx)}, val={len(val_idx)}")
    return coords.astype(np.float64), section_ids, train_idx, val_idx


def reassemble_all_embeds(X_tr, val_embeds, train_idx, val_idx, n_total) -> np.ndarray:
    """Recover the (N_total, embed_dim) array indexed by original spot index."""
    d = X_tr.shape[1]
    out = np.zeros((n_total, d), dtype=np.float32)
    out[train_idx] = X_tr
    out[val_idx] = val_embeds
    return out


# ----------------------------------------------------------------------------
# Spatial K-NN aggregation
# ----------------------------------------------------------------------------


def knn_neighbor_mean(
    embeds: np.ndarray, coords: np.ndarray, section_ids: np.ndarray, K: int
) -> np.ndarray:
    """Return (N, embed_dim) array where row i is the mean embedding of i's K
    nearest neighbors by Euclidean coord distance, RESTRICTED to neighbors in
    the same section (no cross-section neighbors — coords are not
    section-comparable). Self is excluded.

    For K=0, returns the embeddings unchanged (so concat(self, self) downstream).
    """
    n = len(coords)
    out = np.zeros_like(embeds)
    if K == 0:
        out[:] = embeds
        return out

    # Process each section independently — coords are per-section pixel coords
    # in HEST, so a tree across sections would compare apples to oranges.
    for sid in np.unique(section_ids):
        mask = section_ids == sid
        idx_in_section = np.where(mask)[0]
        if len(idx_in_section) < 2:
            out[idx_in_section] = embeds[idx_in_section]
            continue
        sec_coords = coords[idx_in_section]
        sec_embeds = embeds[idx_in_section]
        tree = BallTree(sec_coords)
        k_eff = min(K + 1, len(sec_coords))  # +1 for self
        _, knn = tree.query(sec_coords, k=k_eff)
        # knn[i, 0] is self (distance 0); take rows 1..k_eff
        if k_eff > 1:
            neighbor_idx = knn[:, 1:]
            out[idx_in_section] = sec_embeds[neighbor_idx].mean(axis=1)
        else:
            out[idx_in_section] = sec_embeds
    return out


# ----------------------------------------------------------------------------
# Fresh MLP head
# ----------------------------------------------------------------------------


class SpatialHead(nn.Module):
    """MLP head with separate pathway and gene outputs. Mirrors the architecture
    of the cached baseline (Linear → ReLU → Linear) for both outputs."""

    def __init__(self, in_dim: int, hidden_dim: int, n_pathways: int, n_genes: int):
        super().__init__()
        self.pathway_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_pathways),
        )
        self.gene_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, x):
        return self.pathway_head(x), self.gene_head(x)


def train_head(
    X_train: np.ndarray, yp_train: np.ndarray, yg_train: np.ndarray,
    X_val: np.ndarray, yp_val: np.ndarray, yg_val: np.ndarray,
    device, epochs: int = 100, patience: int = 15, lr: float = 1e-3,
    weight_decay: float = 1e-3, batch_size: int = 128, hidden_dim: int = 256, seed: int = 42,
):
    torch.manual_seed(seed)
    n_pathways = yp_train.shape[1]
    n_genes = yg_train.shape[1]
    in_dim = X_train.shape[1]
    head = SpatialHead(in_dim, hidden_dim, n_pathways, n_genes).to(device)
    opt = optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    Xt = torch.from_numpy(X_train).float().to(device)
    ypt = torch.from_numpy(yp_train).float().to(device)
    ygt = torch.from_numpy(yg_train).float().to(device)
    Xv = torch.from_numpy(X_val).float().to(device)
    ypv = torch.from_numpy(yp_val).float().to(device)
    ygv = torch.from_numpy(yg_val).float().to(device)

    n = Xt.shape[0]
    best = float("inf")
    best_state = None
    bad = 0
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            pp, gp = head(Xt[idx])
            loss = mse(pp, ypt[idx]) + mse(gp, ygt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        head.eval()
        with torch.no_grad():
            pp, gp = head(Xv)
            v = (mse(pp, ypv) + mse(gp, ygv)).item()
        if v < best - 1e-5:
            best = v; bad = 0
            best_state = {k: t.detach().cpu().clone() for k, t in head.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pp, gp = head(Xv)
    return pp.cpu().numpy(), gp.cpu().numpy(), epoch + 1


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-path", default="./bench_tabpfn_results/bench_cache.npz")
    p.add_argument("--output-dir", default="./bench_spatial_results")
    p.add_argument("--data-dir", default="./hest_data")
    p.add_argument("--metadata-csv", default="./hest_data/HEST_v1_1_0.csv")
    p.add_argument("--n-sections", type=int, default=36)
    p.add_argument("--max-spots-per-section", type=int, default=400)
    p.add_argument("--n-genes", type=int, default=1000)
    p.add_argument("--n-pathways", type=int, default=775)
    p.add_argument("--normalization", choices=["paper", "paper_log1p_only", "paper_zscore"],
                   default="paper_log1p_only")
    p.add_argument("--K-list", type=str, default="0,4,8,16,32",
                   help="Comma-separated K values to try.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    K_values = [int(k.strip()) for k in args.K_list.split(",")]
    print(f"Device: {device}; K values: {K_values}")

    # --- Load cache + coords ---
    cache = load_cache(args.cache_path)
    n_total = cache["X_tr"].shape[0] + cache["val_embeds"].shape[0]
    coords, section_ids, train_idx, val_idx = load_coords_and_fold(args, n_total)
    section_ids = np.asarray(section_ids)

    # Reassemble all embeddings + all targets in original spot ordering
    all_embeds = reassemble_all_embeds(cache["X_tr"], cache["val_embeds"], train_idx, val_idx, n_total)

    # MLP-only reference (from cache)
    mlp_baseline = {
        "pathway": compute_metrics(cache["val_mlp_pp"], cache["val_yp"]),
        "gene":    compute_metrics(cache["val_mlp_gp"], cache["val_yg"]),
    }
    print(f"\nMLP-only baseline (cached): gene PCC={mlp_baseline['gene']['PCC']:.4f}, "
          f"pathway PCC={mlp_baseline['pathway']['PCC']:.4f}")

    # --- For each K: aggregate, train, evaluate ---
    results = []
    for K in K_values:
        print(f"\n=== K={K} ===")
        t0 = time.time()
        neighbor_means = knn_neighbor_mean(all_embeds, coords, section_ids, K)
        if K == 0:
            X_aug = all_embeds  # 256-d
        else:
            X_aug = np.concatenate([all_embeds, neighbor_means], axis=1).astype(np.float32)  # 512-d
        t_aug = time.time() - t0
        print(f"  aug features: shape {X_aug.shape}, aggregated in {t_aug:.1f}s")

        # Recover train/val targets
        X_train = X_aug[train_idx]
        X_val = X_aug[val_idx]
        yp_train = cache["yp_tr"]  # already in train_idx order
        yg_train = cache["yg_tr"]
        yp_val = cache["val_yp"]
        yg_val = cache["val_yg"]

        t0 = time.time()
        pp, gp, n_epochs = train_head(
            X_train, yp_train, yg_train, X_val, yp_val, yg_val,
            device, epochs=args.epochs, patience=args.patience, lr=args.lr,
            weight_decay=args.weight_decay, batch_size=args.batch_size,
            hidden_dim=args.hidden_dim, seed=args.seed,
        )
        t_train = time.time() - t0
        pwy = compute_metrics(pp, yp_val)
        gn = compute_metrics(gp, yg_val)
        print(f"  trained {n_epochs} epochs in {t_train:.1f}s")
        print(f"  gene PCC={gn['PCC']:.4f} (Δ {gn['PCC']-mlp_baseline['gene']['PCC']:+.4f}), "
              f"pathway PCC={pwy['PCC']:.4f} (Δ {pwy['PCC']-mlp_baseline['pathway']['PCC']:+.4f})")

        results.append({
            "K": K,
            "in_dim": int(X_aug.shape[1]),
            "n_epochs": int(n_epochs),
            "t_aggregate_s": float(t_aug),
            "t_train_s": float(t_train),
            "pathway": pwy,
            "gene": gn,
            "delta_gene_PCC": float(gn["PCC"] - mlp_baseline["gene"]["PCC"]),
            "delta_pathway_PCC": float(pwy["PCC"] - mlp_baseline["pathway"]["PCC"]),
        })

        # incremental write
        with open(os.path.join(args.output_dir, "spatial_results.json"), "w") as f:
            json.dump({
                "args": vars(args),
                "mlp_baseline": mlp_baseline,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, default=str)

    render_report(mlp_baseline, results, os.path.join(args.output_dir, "BENCHMARK_REPORT.md"))
    print(f"\nResults saved to {args.output_dir}/")


def render_report(mlp_baseline, results, path):
    b = mlp_baseline
    lines = []
    lines.append("# Spatial K-NN Augmentation Benchmark")
    lines.append("")
    lines.append("Each row trains a fresh MLP head on `concat(self_embedding, mean(K_nearest_neighbor_embeddings))`.")
    lines.append("Neighbors are picked by Euclidean distance in pixel coords, restricted to the same HEST section.")
    lines.append("Self is excluded. K=0 is a pipeline sanity check (no augmentation; fresh head trained on the same 256-d embeddings).")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| K | input dim | epochs | gene PCC | Δ gene | pwy PCC | Δ pwy | gene MSE | gene MAE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| (MLP-only baseline) | 256 | — | **{b['gene']['PCC']:.4f}** | — | "
        f"**{b['pathway']['PCC']:.4f}** | — | {b['gene']['MSE']:.4f} | {b['gene']['MAE']:.4f} |"
    )
    for r in results:
        lines.append(
            f"| {r['K']} | {r['in_dim']} | {r['n_epochs']} | {r['gene']['PCC']:.4f} | "
            f"{r['delta_gene_PCC']:+.4f} | {r['pathway']['PCC']:.4f} | "
            f"{r['delta_pathway_PCC']:+.4f} | {r['gene']['MSE']:.4f} | {r['gene']['MAE']:.4f} |"
        )
    lines.append("")
    lines.append("## Time")
    lines.append("")
    lines.append("| K | aggregate (s) | train (s) | total (s) |")
    lines.append("|---:|---:|---:|---:|")
    for r in results:
        total = r["t_aggregate_s"] + r["t_train_s"]
        lines.append(f"| {r['K']} | {r['t_aggregate_s']:.1f} | {r['t_train_s']:.1f} | {total:.1f} |")
    lines.append("")

    # Summarize
    best = max(results, key=lambda r: r["gene"]["PCC"])
    lines.append(f"**Best gene PCC**: K={best['K']} → {best['gene']['PCC']:.4f} "
                 f"(Δ {best['delta_gene_PCC']:+.4f} vs MLP-only baseline)")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

"""
Per-gene cheap-learner benchmark.

Tests whether a per-dim specialist (LightGBM or kernel ridge with RFF features)
can refine the MLP head's predictions on the hardest dims — same comparison as
the TabPFN benchmark, just with cheap learners instead of a 25M-param tabular
transformer.

Variants:
  L1  LightGBM refinement     — replace MLP on top-k MLP-residual-var dims
  L2  LightGBM residual + α   — predict MLP residual, blend with per-dim α∈[0,1]
                                 calibrated on a 10% holdout (bounded never-worse)
  L3  Kernel ridge RFF        — random Fourier feature ridge regression per dim,
                                 refinement mode (replace MLP on top-k dims)

Top-k: pathway=20, gene=50 — matches the TabPFN C1 reference.

The trained MLP + embeddings + targets are cached to disk after the first run
so future learner ideas can iterate without retraining. The reference rows
(MLP-only baseline, TabPFN C1, TabPFN C5) are read from bench_tabpfn_results
when present.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb

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


CACHE_KEYS = [
    "X_tr", "yp_tr", "yg_tr", "mlp_pp_tr", "mlp_gp_tr",
    "val_embeds", "val_mlp_pp", "val_mlp_gp", "val_yp", "val_yg",
]


# ----------------------------------------------------------------------------
# Cache: train MLP + collect arrays once, save/load .npz
# ----------------------------------------------------------------------------


def collect_arrays(model, loader, device):
    """Run model on loader: return embeddings, targets, MLP predictions."""
    model.eval()
    Xs, yp, yg, ppm, gpm = [], [], [], [], []
    with torch.no_grad():
        for f, pw, g, _c in loader:
            f = f.to(device)
            h = model.forward_vision(f)
            Xs.append(h.cpu().numpy())
            yp.append(pw.numpy())
            yg.append(g.numpy())
            # Bench infrastructure was originally built around TabPFNHead, which
            # exposes the MLP via `.mlp(...)`. Plain Sequential heads (used for
            # mlp-head_type) take h directly.
            if hasattr(model.pathway_head, "mlp"):
                ppm.append(model.pathway_head.mlp(h).cpu().numpy())
                gpm.append(model.gene_head.mlp(h).cpu().numpy())
            else:
                ppm.append(model.pathway_head(h).cpu().numpy())
                gpm.append(model.gene_head(h).cpu().numpy())
    return (
        np.concatenate(Xs),
        np.concatenate(yp),
        np.concatenate(yg),
        np.concatenate(ppm),
        np.concatenate(gpm),
    )


def build_cache(args, device, cache_path: str) -> Dict[str, np.ndarray]:
    print(f"\nNo cache at {cache_path} — building (load data, extract features, train MLP) ...")

    sample_ids = select_breast_section_ids(args.metadata_csv, args.n_sections, seed=args.seed)
    print(f"Selected {len(sample_ids)} sections")

    t0 = time.time()
    patches, genes, pathways, coords, _section_ids = load_hest_multi_sample(
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
    features_cpu = extract_features(encoder, ds, device, args.feat_batch_size)
    del encoder
    torch.cuda.empty_cache()
    print(f"  features {tuple(features_cpu.shape)} in {time.time()-t0:.1f}s")

    pathways_t = torch.from_numpy(pathways)
    genes_t = torch.from_numpy(genes)
    coords_t = torch.from_numpy(coords)

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(iter(kf.split(np.arange(len(patches)))))
    print(f"  fold: train={len(train_idx)}, val={len(val_idx)}")
    train_loader = make_tensor_loader(
        features_cpu, pathways_t, genes_t, coords_t, train_idx, args.batch_size, True
    )
    val_loader = make_tensor_loader(
        features_cpu, pathways_t, genes_t, coords_t, val_idx, args.batch_size, False
    )

    # MLP-only model (head_type="mlp"). The TabPFN bench used a TabPFN head whose
    # internal MLP was trained — same MLP, same training signal. Use plain mlp
    # here to keep this script self-contained and not depend on TabPFNHead's
    # internals.
    model = PEaRLCached(
        feat_dim=features_cpu.shape[1],
        n_pathways=pathways.shape[1],
        n_genes=genes.shape[1],
        embed_dim=cfg.EMBED_DIM,
        pathway_hidden=cfg.PATHWAY_HIDDEN,
        head_type="mlp",
    ).to(device)

    print("\nStage 1 (contrastive) ...")
    t0 = time.time()
    stage1_contrastive(
        model, train_loader, val_loader, device,
        args.epochs_stage1, args.patience, args.lr, args.weight_decay, args.temperature,
    )
    print(f"  stage 1 done in {time.time()-t0:.1f}s")

    print("\nStage 2 (supervised MLP head training) ...")
    t0 = time.time()
    stage2_supervised(
        model, train_loader, val_loader, device,
        args.epochs_stage2, args.patience, args.lr, args.weight_decay, "mlp",
    )
    print(f"  stage 2 done in {time.time()-t0:.1f}s")

    print("\nCollecting arrays ...")
    X_tr, yp_tr, yg_tr, mlp_pp_tr, mlp_gp_tr = collect_arrays(model, train_loader, device)
    val_embeds_np, val_yp, val_yg, val_mlp_pp_np, val_mlp_gp_np = collect_arrays(model, val_loader, device)
    print(f"  train: X {X_tr.shape}, yp {yp_tr.shape}, yg {yg_tr.shape}")
    print(f"  val:   X {val_embeds_np.shape}")

    cache = {
        "X_tr": X_tr.astype(np.float32),
        "yp_tr": yp_tr.astype(np.float32),
        "yg_tr": yg_tr.astype(np.float32),
        "mlp_pp_tr": mlp_pp_tr.astype(np.float32),
        "mlp_gp_tr": mlp_gp_tr.astype(np.float32),
        "val_embeds": val_embeds_np.astype(np.float32),
        "val_mlp_pp": val_mlp_pp_np.astype(np.float32),
        "val_mlp_gp": val_mlp_gp_np.astype(np.float32),
        "val_yp": val_yp.astype(np.float32),
        "val_yg": val_yg.astype(np.float32),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    print(f"  cache saved to {cache_path} ({os.path.getsize(cache_path)/1e6:.1f} MB)")
    return cache


def load_cache(cache_path: str) -> Dict[str, np.ndarray]:
    print(f"\nLoading cache from {cache_path} ...")
    z = np.load(cache_path)
    cache = {k: z[k] for k in CACHE_KEYS}
    print(f"  loaded: X_tr {cache['X_tr'].shape}, val_embeds {cache['val_embeds'].shape}")
    return cache


# ----------------------------------------------------------------------------
# Top-k selection (same as TabPFN refinement: highest MLP-residual variance)
# ----------------------------------------------------------------------------


def top_k_by_residual_variance(y: np.ndarray, mlp_pred: np.ndarray, k: int) -> np.ndarray:
    """Pick top-k dims where the MLP made the largest mistakes."""
    res = y - mlp_pred
    var = res.var(axis=0)
    k = min(k, y.shape[1])
    return np.argsort(var)[-k:][::-1].astype(np.int64).copy()


# ----------------------------------------------------------------------------
# L1: LightGBM refinement
# ----------------------------------------------------------------------------


LGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)


def fit_lgb_refinement(X_tr, y_tr, mlp_pred_tr, top_k) -> Tuple[List, np.ndarray, float]:
    """Train one LightGBM per top-k dim on raw y. Returns regressors + dim idx + fit time."""
    idx = top_k_by_residual_variance(y_tr, mlp_pred_tr, top_k)
    t0 = time.time()
    models = []
    for d in idx:
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=42)
        m.fit(X_tr, y_tr[:, d])
        models.append(m)
    return models, idx, time.time() - t0


def apply_lgb_refinement(models, idx, X_val, mlp_pred_val) -> Tuple[np.ndarray, float]:
    t0 = time.time()
    out = mlp_pred_val.copy()
    for i, d in enumerate(idx):
        out[:, d] = models[i].predict(X_val)
    return out, time.time() - t0


# ----------------------------------------------------------------------------
# L2: LightGBM residual + α-shrinkage
# ----------------------------------------------------------------------------


def fit_lgb_residual_alpha(X_tr, y_tr, mlp_pred_tr, top_k, holdout_frac=0.1, seed=42):
    """Train one LightGBM per top-k dim on (X, residual). Calibrate per-dim α on
    a holdout slice. Returns (models, idx, alpha, fit_time).
    """
    idx = top_k_by_residual_variance(y_tr, mlp_pred_tr, top_k)
    n = X_tr.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = int(round(n * holdout_frac))
    hold = perm[:n_hold]
    inner = perm[n_hold:]

    t0 = time.time()
    models = []
    alpha = np.zeros(len(idx), dtype=np.float32)
    for i, d in enumerate(idx):
        res = y_tr[:, d] - mlp_pred_tr[:, d]
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=42 + i)
        m.fit(X_tr[inner], res[inner])
        models.append(m)
        if n_hold > 0:
            pred_hold = m.predict(X_tr[hold])
            res_hold = res[hold]
            denom = float(np.sum(pred_hold * pred_hold) + 1e-8)
            a_unclip = float(np.sum(pred_hold * res_hold) / denom)
            alpha[i] = max(0.0, min(1.0, a_unclip))
    fit_t = time.time() - t0
    return models, idx, alpha, fit_t


def apply_lgb_residual_alpha(models, idx, alpha, X_val, mlp_pred_val):
    t0 = time.time()
    out = mlp_pred_val.copy()
    for i, d in enumerate(idx):
        out[:, d] = out[:, d] + alpha[i] * models[i].predict(X_val)
    return out, time.time() - t0


# ----------------------------------------------------------------------------
# L3: Kernel ridge with RFF features
# ----------------------------------------------------------------------------


def fit_krr_rff_refinement(X_tr, y_tr, mlp_pred_tr, top_k, n_components=1024, seed=42):
    """Per-dim kernel ridge regression with random Fourier features.

    The RFF projection is shared across dims (it depends only on X, not y), so
    we transform once and reuse. Each dim's ridge solve is then a closed-form
    Linear(d, 1) on the RFF features.
    """
    idx = top_k_by_residual_variance(y_tr, mlp_pred_tr, top_k)
    t0 = time.time()
    # Pick gamma from median pairwise distance on a 1000-sample subsample
    rng = np.random.default_rng(seed)
    sub = rng.choice(X_tr.shape[0], size=min(1000, X_tr.shape[0]), replace=False)
    Xs = X_tr[sub]
    # median pairwise sqeuclidean distance (subsample of pairs for speed)
    pairs = rng.choice(Xs.shape[0], size=(1000, 2), replace=True)
    d2 = np.sum((Xs[pairs[:, 0]] - Xs[pairs[:, 1]]) ** 2, axis=1)
    d2 = d2[d2 > 0]
    sigma2 = float(np.median(d2)) if len(d2) else 1.0
    gamma = 1.0 / max(sigma2, 1e-6)
    rff = RBFSampler(gamma=gamma, n_components=n_components, random_state=seed)
    Z_tr = rff.fit_transform(X_tr).astype(np.float32)
    models = []
    for d in idx:
        m = Ridge(alpha=1.0)
        m.fit(Z_tr, y_tr[:, d])
        models.append(m)
    fit_t = time.time() - t0
    return models, idx, rff, fit_t


def apply_krr_rff_refinement(models, idx, rff, X_val, mlp_pred_val):
    t0 = time.time()
    Z_val = rff.transform(X_val).astype(np.float32)
    out = mlp_pred_val.copy()
    for i, d in enumerate(idx):
        out[:, d] = models[i].predict(Z_val)
    return out, time.time() - t0


# ----------------------------------------------------------------------------
# Run one variant on both heads
# ----------------------------------------------------------------------------


def run_variant(name, fit_fn, apply_fn, cache, top_k_p, top_k_g, alpha=False):
    print(f"\n=== {name} (top_k_p={top_k_p}, top_k_g={top_k_g}) ===")

    # Pathway head
    if alpha:
        p_models, p_idx, p_alpha, p_fit_t = fit_fn(cache["X_tr"], cache["yp_tr"], cache["mlp_pp_tr"], top_k_p)
        p_out, p_apply_t = apply_fn(p_models, p_idx, p_alpha, cache["val_embeds"], cache["val_mlp_pp"])
    else:
        p_extras = fit_fn(cache["X_tr"], cache["yp_tr"], cache["mlp_pp_tr"], top_k_p)
        # Either (models, idx, fit_t) for L1 or (models, idx, rff, fit_t) for L3
        if len(p_extras) == 3:
            p_models, p_idx, p_fit_t = p_extras
            p_out, p_apply_t = apply_fn(p_models, p_idx, cache["val_embeds"], cache["val_mlp_pp"])
        else:
            p_models, p_idx, p_rff, p_fit_t = p_extras
            p_out, p_apply_t = apply_fn(p_models, p_idx, p_rff, cache["val_embeds"], cache["val_mlp_pp"])

    # Gene head
    if alpha:
        g_models, g_idx, g_alpha, g_fit_t = fit_fn(cache["X_tr"], cache["yg_tr"], cache["mlp_gp_tr"], top_k_g)
        g_out, g_apply_t = apply_fn(g_models, g_idx, g_alpha, cache["val_embeds"], cache["val_mlp_gp"])
    else:
        g_extras = fit_fn(cache["X_tr"], cache["yg_tr"], cache["mlp_gp_tr"], top_k_g)
        if len(g_extras) == 3:
            g_models, g_idx, g_fit_t = g_extras
            g_out, g_apply_t = apply_fn(g_models, g_idx, cache["val_embeds"], cache["val_mlp_gp"])
        else:
            g_models, g_idx, g_rff, g_fit_t = g_extras
            g_out, g_apply_t = apply_fn(g_models, g_idx, g_rff, cache["val_embeds"], cache["val_mlp_gp"])

    print(f"  fit:   pathway {p_fit_t:.1f}s, gene {g_fit_t:.1f}s (total {p_fit_t+g_fit_t:.1f}s)")
    print(f"  apply: pathway {p_apply_t:.1f}s, gene {g_apply_t:.1f}s (total {p_apply_t+g_apply_t:.1f}s)")
    pwy = compute_metrics(p_out, cache["val_yp"])
    gn = compute_metrics(g_out, cache["val_yg"])
    print(f"  pathway PCC={pwy['PCC']:.4f}, gene PCC={gn['PCC']:.4f}")

    out = {
        "name": name,
        "t_fit_pathway_s": float(p_fit_t),
        "t_fit_gene_s": float(g_fit_t),
        "t_fit_total_s": float(p_fit_t + g_fit_t),
        "t_apply_pathway_s": float(p_apply_t),
        "t_apply_gene_s": float(g_apply_t),
        "t_apply_total_s": float(p_apply_t + g_apply_t),
        "t_total_s": float(p_fit_t + g_fit_t + p_apply_t + g_apply_t),
        "top_k_p": int(top_k_p),
        "top_k_g": int(top_k_g),
        "pathway": pwy,
        "gene": gn,
    }
    if alpha:
        out["alpha_pathway"] = {"min": float(p_alpha.min()), "max": float(p_alpha.max()),
                                "mean": float(p_alpha.mean()), "n_nonzero": int((p_alpha > 0.05).sum())}
        out["alpha_gene"] = {"min": float(g_alpha.min()), "max": float(g_alpha.max()),
                              "mean": float(g_alpha.mean()), "n_nonzero": int((g_alpha > 0.05).sum())}
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./hest_data")
    p.add_argument("--metadata-csv", default="./hest_data/HEST_v1_1_0.csv")
    p.add_argument("--output-dir", default="./bench_lightgbm_results")
    p.add_argument("--cache-path", default="./bench_tabpfn_results/bench_cache.npz")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Rebuild the cache even if it exists.")
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
    p.add_argument("--top-k-pathways", type=int, default=20)
    p.add_argument("--top-k-genes", type=int, default=50)
    p.add_argument("--rff-components", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load or build cache ---
    if os.path.exists(args.cache_path) and not args.force_rebuild:
        cache = load_cache(args.cache_path)
    else:
        cache = build_cache(args, device, args.cache_path)

    # --- MLP-only baseline (always recompute from cached arrays) ---
    mlp_baseline = {
        "pathway": compute_metrics(cache["val_mlp_pp"], cache["val_yp"]),
        "gene":    compute_metrics(cache["val_mlp_gp"], cache["val_yg"]),
    }
    print(f"\nMLP-only baseline: gene PCC={mlp_baseline['gene']['PCC']:.4f}, "
          f"pathway PCC={mlp_baseline['pathway']['PCC']:.4f}")

    # --- Run variants ---
    results = []
    results.append(run_variant(
        "L1_lgb_refinement",
        fit_lgb_refinement, apply_lgb_refinement,
        cache, args.top_k_pathways, args.top_k_genes, alpha=False,
    ))
    results.append(run_variant(
        "L2_lgb_residual_alpha",
        fit_lgb_residual_alpha, apply_lgb_residual_alpha,
        cache, args.top_k_pathways, args.top_k_genes, alpha=True,
    ))
    results.append(run_variant(
        "L3_krr_rff_refinement",
        lambda X, y, mlp, k: fit_krr_rff_refinement(X, y, mlp, k, n_components=args.rff_components, seed=args.seed),
        apply_krr_rff_refinement,
        cache, args.top_k_pathways, args.top_k_genes, alpha=False,
    ))

    # --- Save raw results ---
    out = {
        "args": vars(args),
        "mlp_baseline": mlp_baseline,
        "variants": results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "bench_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    # --- Render report ---
    render_report(out, os.path.join(args.output_dir, "BENCHMARK_REPORT.md"))
    print(f"\nResults saved to {args.output_dir}/")


def render_report(out: Dict, path: str):
    b = out["mlp_baseline"]
    lines = []
    lines.append("# Per-gene Cheap-Learner Benchmark")
    lines.append("")
    lines.append("Same fold and same trained MLP as the TabPFN benchmark. "
                 "Top-k dims are picked by MLP-residual variance — the dims where the MLP did worst.")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| ID | Method | Mode |")
    lines.append("|---|---|---|")
    lines.append("| L1 | LightGBM (200 trees) | Refinement — LightGBM prediction *replaces* MLP on top-k dims |")
    lines.append("| L2 | LightGBM (200 trees) | Residual + α-shrinkage — LightGBM predicts MLP residual; per-dim α∈[0,1] calibrated on 10% holdout |")
    lines.append("| L3 | Kernel Ridge w/ RFF features | Refinement — closed-form ridge on RFF-projected features |")
    lines.append("")
    lines.append("## Time")
    lines.append("")
    lines.append("| Variant | fit (s) | apply (s) | total (s) |")
    lines.append("|---|---:|---:|---:|")
    for r in out["variants"]:
        lines.append(f"| {r['name']} | {r['t_fit_total_s']:.1f} | {r['t_apply_total_s']:.1f} | {r['t_total_s']:.1f} |")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Variant | gene PCC | gene MSE | gene MAE | pwy PCC | pwy MSE | pwy MAE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| MLP-only baseline | **{b['gene']['PCC']:.4f}** | {b['gene']['MSE']:.4f} | {b['gene']['MAE']:.4f} | "
        f"**{b['pathway']['PCC']:.4f}** | {b['pathway']['MSE']:.4f} | {b['pathway']['MAE']:.4f} |"
    )
    for r in out["variants"]:
        g, p = r["gene"], r["pathway"]
        lines.append(
            f"| {r['name']} | {g['PCC']:.4f} | {g['MSE']:.4f} | {g['MAE']:.4f} | "
            f"{p['PCC']:.4f} | {p['MSE']:.4f} | {p['MAE']:.4f} |"
        )
    lines.append("")
    lines.append("## Δ-from-MLP-baseline")
    lines.append("")
    lines.append("Positive = better than MLP. TabPFN C1 was −0.0023 on genes, −0.0019 on pathways (from prior benchmark) for reference.")
    lines.append("")
    lines.append("| Variant | Δ gene PCC | Δ pathway PCC |")
    lines.append("|---|---:|---:|")
    for r in out["variants"]:
        dg = r["gene"]["PCC"] - b["gene"]["PCC"]
        dp = r["pathway"]["PCC"] - b["pathway"]["PCC"]
        lines.append(f"| {r['name']} | {dg:+.4f} | {dp:+.4f} |")
    lines.append("")

    # α-shrinkage diagnostics
    for r in out["variants"]:
        if "alpha_pathway" in r:
            a_p = r["alpha_pathway"]; a_g = r["alpha_gene"]
            lines.append(f"### {r['name']} α-shrinkage diagnostics")
            lines.append("")
            lines.append(f"- pathway α: min={a_p['min']:.2f}, max={a_p['max']:.2f}, mean={a_p['mean']:.2f}, "
                         f"nonzero={a_p['n_nonzero']}/{r['top_k_p']}")
            lines.append(f"- gene    α: min={a_g['min']:.2f}, max={a_g['max']:.2f}, mean={a_g['mean']:.2f}, "
                         f"nonzero={a_g['n_nonzero']}/{r['top_k_g']}")
            lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

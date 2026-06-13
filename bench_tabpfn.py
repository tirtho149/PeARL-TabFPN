"""
Benchmark 5 TabPFN configurations on the SAME trained MLP head.

Goal: understand where TabPFN spends its time and how the obvious speedup
knobs trade off against accuracy. Reuses the cached-feature training path
from run_paper_reproduction.py so the comparison is at production scale
(36 sections, ~10k training context, UNI features) but only one fold is
run and the MLP is trained exactly once.

The 5 configs (refinement mode throughout):
  C1  baseline           n_est=4, context=full, top_k_p=20, top_k_g=50
  C2  n_estimators=1     n_est=1, context=full, top_k_p=20, top_k_g=50
  C3  context cap 4096   n_est=4, context=4096, top_k_p=20, top_k_g=50
  C4  top_k_genes=20     n_est=4, context=full, top_k_p=20, top_k_g=20
  C5  all combined       n_est=1, context=4096, top_k_p=20, top_k_g=20

For each: fit_time, apply_time, gene/pathway PCC/MSE/MAE. MLP-only baseline
is also reported as a reference row.
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
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from pearl_config import cfg
from pearl_data import HESTDataset, load_hest_multi_sample
from pearl_eval import compute_metrics
from pearl_models import VisionEncoder
from pearl_models_tabpfn import TabPFNHead
from run_paper_reproduction import (
    PEaRLCached,
    extract_features,
    make_tensor_loader,
    select_breast_section_ids,
    stage1_contrastive,
    stage2_supervised,
)


# ----------------------------------------------------------------------------
# The 5 configurations. Each is applied to the same trained model so timings
# and metrics are directly comparable.
# ----------------------------------------------------------------------------

CONFIGS = [
    {"name": "C1_baseline",        "n_estimators": 4, "max_ctx": None, "top_k_p": 20, "top_k_g": 50},
    {"name": "C2_n_est_1",         "n_estimators": 1, "max_ctx": None, "top_k_p": 20, "top_k_g": 50},
    {"name": "C3_ctx_cap_4096",    "n_estimators": 4, "max_ctx": 4096, "top_k_p": 20, "top_k_g": 50},
    {"name": "C4_top_k_genes_20",  "n_estimators": 4, "max_ctx": None, "top_k_p": 20, "top_k_g": 20},
    {"name": "C5_all_combined",    "n_estimators": 1, "max_ctx": 4096, "top_k_p": 20, "top_k_g": 20},
]


def collect_train_arrays(model, train_loader, device):
    """Run the model on the train set once: collect embeddings, targets, MLP preds.
    Reused across all 5 configs so each TabPFN config sees identical inputs.
    """
    model.eval()
    Xs, yp, yg, ppm, gpm = [], [], [], [], []
    with torch.no_grad():
        for f, pw, g, _c in train_loader:
            f = f.to(device)
            h = model.forward_vision(f)
            Xs.append(h.cpu().numpy())
            yp.append(pw.numpy())
            yg.append(g.numpy())
            ppm.append(model.pathway_head.mlp(h).cpu().numpy())
            gpm.append(model.gene_head.mlp(h).cpu().numpy())
    return (
        np.concatenate(Xs),
        np.concatenate(yp),
        np.concatenate(yg),
        np.concatenate(ppm),
        np.concatenate(gpm),
    )


def collect_val_arrays(model, val_loader, device):
    """Same for val: returns (embeds_tensor, mlp_pp_tensor, mlp_gp_tensor, y_p, y_g)."""
    model.eval()
    embeds, mlp_pp, mlp_gp, pt, gt = [], [], [], [], []
    with torch.no_grad():
        for f, pw, g, _c in val_loader:
            f = f.to(device)
            h = model.forward_vision(f)
            embeds.append(h.cpu())
            mlp_pp.append(model.pathway_head.mlp(h).cpu())
            mlp_gp.append(model.gene_head.mlp(h).cpu())
            pt.append(pw.numpy())
            gt.append(g.numpy())
    return (
        torch.cat(embeds, dim=0),
        torch.cat(mlp_pp, dim=0),
        torch.cat(mlp_gp, dim=0),
        np.concatenate(pt),
        np.concatenate(gt),
    )


def reset_head(head: TabPFNHead, top_k: int, n_estimators: int):
    """Reset a TabPFNHead to behave as if it were just constructed with the new
    knobs (without re-creating it, which would discard the trained MLP)."""
    head.tabpfn_top_k = min(top_k, head.output_dim)
    head.n_estimators = n_estimators
    head.is_fitted = False
    head._regressors = None
    head._top_k_indices = torch.zeros(0, dtype=torch.long)
    head._alpha = torch.zeros(0, dtype=torch.float32)
    # ensure use_tabpfn stayed on (could be False if tabpfn import failed earlier)
    if not head.use_tabpfn:
        try:
            from tabpfn import TabPFNRegressor  # noqa: F401
            head.use_tabpfn = True
        except ImportError:
            pass


def run_one_config(
    model: PEaRLCached,
    cfg_spec: Dict,
    X_tr: np.ndarray,
    yp_tr: np.ndarray,
    yg_tr: np.ndarray,
    mlp_pp_tr: np.ndarray,
    mlp_gp_tr: np.ndarray,
    val_embeds: torch.Tensor,
    val_mlp_pp: torch.Tensor,
    val_mlp_gp: torch.Tensor,
    val_yp: np.ndarray,
    val_yg: np.ndarray,
    seed: int = 42,
) -> Dict:
    """Time fit + apply for one config, compute metrics, return a result dict."""
    name = cfg_spec["name"]
    n_est = cfg_spec["n_estimators"]
    max_ctx = cfg_spec["max_ctx"]
    top_k_p = cfg_spec["top_k_p"]
    top_k_g = cfg_spec["top_k_g"]
    print(f"\n=== {name}: n_est={n_est}, max_ctx={max_ctx}, top_k_p={top_k_p}, top_k_g={top_k_g} ===")

    reset_head(model.pathway_head, top_k_p, n_est)
    reset_head(model.gene_head, top_k_g, n_est)

    # Subsample training context if max_ctx is set. Same seed across configs so
    # the actual subsample is deterministic per cap value.
    if max_ctx is not None and X_tr.shape[0] > max_ctx:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X_tr.shape[0], size=max_ctx, replace=False)
        X_use, yp_use, yg_use, ppm_use, gpm_use = (
            X_tr[idx], yp_tr[idx], yg_tr[idx], mlp_pp_tr[idx], mlp_gp_tr[idx]
        )
        actual_ctx = max_ctx
    else:
        X_use, yp_use, yg_use, ppm_use, gpm_use = X_tr, yp_tr, yg_tr, mlp_pp_tr, mlp_gp_tr
        actual_ctx = X_tr.shape[0]

    # ---- Fit ----
    t0 = time.time()
    model.pathway_head.fit(X_use, yp_use, mlp_pred_on_X=ppm_use)
    t_fit_p = time.time() - t0
    t0 = time.time()
    model.gene_head.fit(X_use, yg_use, mlp_pred_on_X=gpm_use)
    t_fit_g = time.time() - t0
    print(f"  fit: pathway {t_fit_p:.1f}s, gene {t_fit_g:.1f}s (total {t_fit_p+t_fit_g:.1f}s)")

    # ---- Apply on val ----
    t0 = time.time()
    refined_pp = model.pathway_head.apply_tabpfn(val_embeds, val_mlp_pp)
    t_apply_p = time.time() - t0
    t0 = time.time()
    refined_gp = model.gene_head.apply_tabpfn(val_embeds, val_mlp_gp)
    t_apply_g = time.time() - t0
    print(f"  apply: pathway {t_apply_p:.1f}s, gene {t_apply_g:.1f}s (total {t_apply_p+t_apply_g:.1f}s)")

    # ---- Metrics ----
    pathway_metrics = compute_metrics(refined_pp.numpy(), val_yp)
    gene_metrics = compute_metrics(refined_gp.numpy(), val_yg)
    print(f"  pathway PCC={pathway_metrics['PCC']:.4f}, gene PCC={gene_metrics['PCC']:.4f}")

    return {
        "name": name,
        "spec": cfg_spec,
        "actual_train_context": int(actual_ctx),
        "t_fit_pathway_s": float(t_fit_p),
        "t_fit_gene_s": float(t_fit_g),
        "t_fit_total_s": float(t_fit_p + t_fit_g),
        "t_apply_pathway_s": float(t_apply_p),
        "t_apply_gene_s": float(t_apply_g),
        "t_apply_total_s": float(t_apply_p + t_apply_g),
        "t_total_s": float(t_fit_p + t_fit_g + t_apply_p + t_apply_g),
        "pathway": pathway_metrics,
        "gene": gene_metrics,
    }


def render_chart(results: List[Dict], baseline: Dict, out_path: str):
    """Plain-text bar chart: fit + apply per config."""
    max_total = max(r["t_total_s"] for r in results)
    bar_width = 50
    lines = []
    lines.append("")
    lines.append("Time per TabPFN config (seconds)")
    lines.append("=" * 88)
    lines.append(f"  {'Config':<22} {'fit':>8} {'apply':>8} {'total':>8}  bar")
    lines.append("  " + "-" * 84)
    for r in results:
        n = max(1, int(round(r["t_total_s"] / max_total * bar_width)))
        bar = "#" * n
        lines.append(
            f"  {r['name']:<22} {r['t_fit_total_s']:>8.1f} {r['t_apply_total_s']:>8.1f} {r['t_total_s']:>8.1f}  {bar}"
        )
    lines.append("")
    lines.append("Metrics per TabPFN config")
    lines.append("=" * 88)
    lines.append(f"  {'Config':<22} {'gene PCC':>10} {'gene MSE':>10} {'pwy PCC':>10} {'pwy MSE':>10}")
    lines.append("  " + "-" * 84)
    lines.append(
        f"  {'MLP-only baseline':<22} {baseline['gene']['PCC']:>10.4f} {baseline['gene']['MSE']:>10.4f} "
        f"{baseline['pathway']['PCC']:>10.4f} {baseline['pathway']['MSE']:>10.4f}"
    )
    for r in results:
        lines.append(
            f"  {r['name']:<22} {r['gene']['PCC']:>10.4f} {r['gene']['MSE']:>10.4f} "
            f"{r['pathway']['PCC']:>10.4f} {r['pathway']['MSE']:>10.4f}"
        )
    lines.append("")
    chart = "\n".join(lines)
    print(chart)
    with open(out_path, "w") as f:
        f.write(chart)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./hest_data")
    p.add_argument("--metadata-csv", default="./hest_data/HEST_v1_1_0.csv")
    p.add_argument("--output-dir", default="./bench_tabpfn_results")
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
    p.add_argument(
        "--normalization", choices=["paper", "paper_log1p_only", "paper_zscore"],
        default="paper_log1p_only",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny scale to verify the pipeline runs end-to-end.")
    args = p.parse_args()

    if args.smoke_test:
        args.n_sections = 5
        args.max_spots_per_section = 100
        args.n_pathways = 200
        args.epochs_stage1 = 5
        args.epochs_stage2 = 5
        args.patience = 3
        # Smaller top-k so each config still distinguishes itself at small scale
        for c in CONFIGS:
            c["top_k_p"] = min(c["top_k_p"], 5)
            c["top_k_g"] = min(c["top_k_g"], 10)
            if c["max_ctx"] is not None:
                c["max_ctx"] = 200

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}, Encoder: {args.encoder}")
    print(f"Sections: {args.n_sections}, max_spots/section: {args.max_spots_per_section}")

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------
    sample_ids = select_breast_section_ids(args.metadata_csv, args.n_sections, seed=args.seed)
    print(f"Selected {len(sample_ids)} sections")
    print("\nLoading data ...")
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
    print(f"Data loaded in {time.time()-t0:.1f}s; patches={patches.shape}")

    # ------------------------------------------------------------------------
    # Extract features once
    # ------------------------------------------------------------------------
    print("\nExtracting backbone features ...")
    t0 = time.time()
    encoder = VisionEncoder(
        embed_dim=cfg.EMBED_DIM, pretrained=True, backbone=args.encoder, freeze_backbone=True
    ).to(device)
    ds = HESTDataset(patches, genes, pathways, coords, sample_id="multi")
    features_cpu = extract_features(encoder, ds, device, args.feat_batch_size)
    del encoder
    torch.cuda.empty_cache()
    print(f"Features {tuple(features_cpu.shape)} in {time.time()-t0:.1f}s")

    pathways_t = torch.from_numpy(pathways)
    genes_t = torch.from_numpy(genes)
    coords_t = torch.from_numpy(coords)

    # ------------------------------------------------------------------------
    # Single 80/20 fold (KFold by spot, like --split spot in the main script)
    # ------------------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(iter(kf.split(np.arange(len(patches)))))
    print(f"\nFold: train={len(train_idx)}, val={len(val_idx)}")
    train_loader = make_tensor_loader(
        features_cpu, pathways_t, genes_t, coords_t, train_idx, args.batch_size, True
    )
    val_loader = make_tensor_loader(
        features_cpu, pathways_t, genes_t, coords_t, val_idx, args.batch_size, False
    )

    # ------------------------------------------------------------------------
    # Build PEaRLCached with TabPFN heads (head_type=tabpfn) — the MLP inside
    # TabPFNHead is what gets trained in stage 1+2. We'll then mutate the heads
    # per config and refit.
    # ------------------------------------------------------------------------
    model = PEaRLCached(
        feat_dim=features_cpu.shape[1],
        n_pathways=pathways.shape[1],
        n_genes=genes.shape[1],
        embed_dim=cfg.EMBED_DIM,
        pathway_hidden=cfg.PATHWAY_HIDDEN,
        head_type="tabpfn",
        # initial knobs don't matter — we override before each fit
        tabpfn_top_k_pathways=CONFIGS[0]["top_k_p"],
        tabpfn_top_k_genes=CONFIGS[0]["top_k_g"],
        tabpfn_mode="refinement",
        tabpfn_n_estimators=CONFIGS[0]["n_estimators"],
    ).to(device)

    print("\nStage 1 (contrastive) ...")
    t0 = time.time()
    stage1_contrastive(
        model, train_loader, val_loader, device,
        args.epochs_stage1, args.patience, args.lr, args.weight_decay, args.temperature,
    )
    t_stage1 = time.time() - t0
    print(f"Stage 1 done in {t_stage1:.1f}s")

    print("\nStage 2 (supervised MLP head training) ...")
    t0 = time.time()
    stage2_supervised(
        model, train_loader, val_loader, device,
        args.epochs_stage2, args.patience, args.lr, args.weight_decay, "tabpfn",
    )
    t_stage2 = time.time() - t0
    print(f"Stage 2 done in {t_stage2:.1f}s")

    # ------------------------------------------------------------------------
    # Collect train/val arrays ONCE — reused across configs
    # ------------------------------------------------------------------------
    print("\nCollecting train + val arrays ...")
    X_tr, yp_tr, yg_tr, mlp_pp_tr, mlp_gp_tr = collect_train_arrays(model, train_loader, device)
    val_embeds, val_mlp_pp, val_mlp_gp, val_yp, val_yg = collect_val_arrays(model, val_loader, device)
    print(f"  train: X {X_tr.shape}, yp {yp_tr.shape}, yg {yg_tr.shape}")
    print(f"  val:   X {tuple(val_embeds.shape)}")

    # MLP-only baseline metrics
    mlp_baseline = {
        "pathway": compute_metrics(val_mlp_pp.numpy(), val_yp),
        "gene":    compute_metrics(val_mlp_gp.numpy(), val_yg),
    }
    print(f"\nMLP-only baseline: gene PCC={mlp_baseline['gene']['PCC']:.4f}, "
          f"pathway PCC={mlp_baseline['pathway']['PCC']:.4f}")

    # ------------------------------------------------------------------------
    # Loop the 5 configs
    # ------------------------------------------------------------------------
    all_results = []
    for cfg_spec in CONFIGS:
        res = run_one_config(
            model, cfg_spec,
            X_tr, yp_tr, yg_tr, mlp_pp_tr, mlp_gp_tr,
            val_embeds, val_mlp_pp, val_mlp_gp, val_yp, val_yg,
            seed=args.seed,
        )
        all_results.append(res)
        # write incrementally so a crash mid-config still preserves earlier results
        with open(os.path.join(args.output_dir, "bench_results.json"), "w") as f:
            json.dump({
                "args": vars(args),
                "n_train": int(X_tr.shape[0]),
                "n_val": int(val_embeds.shape[0]),
                "t_stage1_s": float(t_stage1),
                "t_stage2_s": float(t_stage2),
                "mlp_baseline": mlp_baseline,
                "configs": all_results,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, default=str)

    render_chart(all_results, mlp_baseline, os.path.join(args.output_dir, "chart.txt"))
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

"""
Faithful PEaRL reproduction targeting arXiv:2510.03455.

Key differences from the original code:
  - UNI vision encoder (pathology foundation model) instead of ImageNet ViT-L
  - ImageNet-normalized image inputs (the original code only divided by 255)
  - Multi-section data: pools spots from N HEST-1k Breast cancer sections,
    pathway columns aligned across sections by global top-N variance
  - 5-fold GroupKFold cross-validation, splitting by section so spots from
    the same section never leak between train and val
  - Paper-style normalization: per-gene min-max for genes, z-normed pathways
  - Early stopping (patience 15) on val loss, up to 100 epochs
  - PCC computed with constant-target columns filtered out
  - **Feature caching**: UNI is frozen, so we run it once per fold to extract
    (N, 1024) features and reuse them across all training epochs and both
    variants. Turns a 4-hr fold into ~10 min.

Both the baseline (MLP heads) and the TabPFN follow-up are run per fold; mean
± std over folds is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, TensorDataset

from pearl_config import cfg
from pearl_data import HESTDataset, load_hest_multi_sample
from pearl_eval import compute_metrics
from pearl_models import (
    ContrastiveLoss,
    PathwayEncoder,
    SupervisedLoss,
    VisionEncoder,
)
from pearl_models_tabpfn import SupervisedLossTabPFN, TabPFNHead


# ----------------------------------------------------------------------------
# Cached-feature model. Functionally equivalent to PEaRL with frozen vision
# encoder, but parameterized by precomputed UNI features instead of patches.
# ----------------------------------------------------------------------------


class PEaRLCached(nn.Module):
    """Trainable parts only. Backbone-features are precomputed and passed in.

    Components:
      - feat_proj:        Linear(feat_dim, embed_dim) — the only trainable
                          piece of the vision branch (replaces VisionEncoder
                          since UNI is frozen).
      - pathway_encoder:  same as baseline.
      - pathway_head, gene_head:  MLP for baseline; TabPFNHead (which also
                          contains an MLP) for the TabPFN variant.
    """

    def __init__(
        self,
        feat_dim: int,
        n_pathways: int,
        n_genes: int,
        embed_dim: int = 256,
        pathway_hidden: int = 512,
        head_type: str = "mlp",  # "mlp" or "tabpfn"
        tabpfn_top_k_pathways: int = 20,
        tabpfn_top_k_genes: int = 50,
        tabpfn_mode: str = "refinement",  # "refinement" or "residual"
        tabpfn_n_estimators: int = 4,
    ):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, embed_dim)
        self.pathway_encoder = PathwayEncoder(
            n_pathways=n_pathways,
            embed_dim=embed_dim,
            hidden_dim=pathway_hidden,
        )
        self.head_type = head_type
        self.tabpfn_mode = tabpfn_mode
        if head_type == "tabpfn":
            self.pathway_head = TabPFNHead(
                embed_dim, n_pathways, use_tabpfn=True, tabpfn_top_k=tabpfn_top_k_pathways,
                mode=tabpfn_mode, n_estimators=tabpfn_n_estimators,
            )
            self.gene_head = TabPFNHead(
                embed_dim, n_genes, use_tabpfn=True, tabpfn_top_k=tabpfn_top_k_genes,
                mode=tabpfn_mode, n_estimators=tabpfn_n_estimators,
            )
        else:
            self.pathway_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, n_pathways),
            )
            self.gene_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, n_genes),
            )

    def forward_vision(self, features: torch.Tensor) -> torch.Tensor:
        return self.feat_proj(features)

    def forward_contrastive(self, features, pathways, coords):
        h_image = F.normalize(self.forward_vision(features), p=2, dim=1)
        h_pathway = F.normalize(self.pathway_encoder(pathways, coords), p=2, dim=1)
        return h_image, h_pathway

    def forward_supervised(self, features):
        h = self.forward_vision(features)
        return self.pathway_head(h), self.gene_head(h)

    def fit_tabpfn_heads(self, X_train, y_pathway_train, y_gene_train,
                         mlp_pathway_pred=None, mlp_gene_pred=None):
        if self.head_type != "tabpfn":
            return
        self.pathway_head.fit(X_train, y_pathway_train, mlp_pred_on_X=mlp_pathway_pred)
        self.gene_head.fit(X_train, y_gene_train, mlp_pred_on_X=mlp_gene_pred)


# ----------------------------------------------------------------------------
# Feature extraction (run once per fold)
# ----------------------------------------------------------------------------


def extract_features(
    encoder: VisionEncoder,
    dataset: HESTDataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    encoder.eval()
    feats = []
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            patches = batch["patch"].to(device, non_blocking=True)
            f = encoder.backbone_features(patches)
            feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)


# ----------------------------------------------------------------------------
# Training loops (operating on cached features)
# ----------------------------------------------------------------------------


def make_tensor_loader(
    features: torch.Tensor,
    pathways: torch.Tensor,
    genes: torch.Tensor,
    coords: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    idx = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    ds = TensorDataset(features[idx], pathways[idx], genes[idx], coords[idx])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def stage1_contrastive(model, train_loader, val_loader, device, epochs, patience, lr, wd, temperature):
    loss_fn = ContrastiveLoss(temperature=temperature)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, bad, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        ep, n = 0.0, 0
        for f, pw, _g, c in train_loader:
            f, pw, c = f.to(device), pw.to(device), c.to(device)
            hi, hp = model.forward_contrastive(f, pw, c)
            loss = loss_fn(hi, hp)
            opt.zero_grad(); loss.backward(); opt.step()
            ep += loss.item(); n += 1
        sched.step()
        model.eval()
        v, vn = 0.0, 0
        with torch.no_grad():
            for f, pw, _g, c in val_loader:
                f, pw, c = f.to(device), pw.to(device), c.to(device)
                hi, hp = model.forward_contrastive(f, pw, c)
                v += loss_fn(hi, hp).item(); vn += 1
        v /= max(vn, 1)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if (epoch + 1) % 5 == 0:
            print(f"    [stage1] epoch {epoch+1}/{epochs} train={ep/max(n,1):.4f} val={v:.4f} bad={bad}")
        if bad >= patience:
            print(f"    [stage1] early stop at epoch {epoch+1} (best val {best:.4f})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def stage2_supervised(model, train_loader, val_loader, device, epochs, patience, lr, wd, head_type):
    """Train the supervised heads + the feat_proj on cached features.
    The pathway encoder (used only for stage 1) is frozen here."""
    for p in model.pathway_encoder.parameters():
        p.requires_grad = False
    loss_fn = SupervisedLossTabPFN() if head_type == "tabpfn" else SupervisedLoss()
    trainables = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(trainables, lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, bad, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        ep, n = 0.0, 0
        for f, pw, g, _c in train_loader:
            f, pw, g = f.to(device), pw.to(device), g.to(device)
            pp, gp = model.forward_supervised(f)
            pl, gl = loss_fn(pp, pw, gp, g)
            loss = pl + gl
            opt.zero_grad(); loss.backward(); opt.step()
            ep += loss.item(); n += 1
        sched.step()
        model.eval()
        v, vn = 0.0, 0
        with torch.no_grad():
            for f, pw, g, _c in val_loader:
                f, pw, g = f.to(device), pw.to(device), g.to(device)
                pp, gp = model.forward_supervised(f)
                pl, gl = loss_fn(pp, pw, gp, g)
                v += (pl + gl).item(); vn += 1
        v /= max(vn, 1)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if (epoch + 1) % 5 == 0:
            print(f"    [stage2] epoch {epoch+1}/{epochs} train={ep/max(n,1):.4f} val={v:.4f} bad={bad}")
        if bad >= patience:
            print(f"    [stage2] early stop at epoch {epoch+1} (best val {best:.4f})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def evaluate(model, val_loader, device):
    """Run MLP forward over the full val set, then (if TabPFN heads are fitted)
    do a single per-dim TabPFN predict on all val embeddings at once. Avoids the
    O(n_batches × top_k) per-call scaling that blocked eval at full scale."""
    model.eval()
    embeds, mlp_pp, mlp_gp, pt, gt = [], [], [], [], []
    with torch.no_grad():
        for f, pw, g, _c in val_loader:
            f = f.to(device)
            h = model.forward_vision(f)            # (B, embed_dim)
            ppred = model.pathway_head(h) if not isinstance(model.pathway_head, TabPFNHead) else model.pathway_head.mlp(h)
            gpred = model.gene_head(h) if not isinstance(model.gene_head, TabPFNHead) else model.gene_head.mlp(h)
            embeds.append(h.cpu()); mlp_pp.append(ppred.cpu()); mlp_gp.append(gpred.cpu())
            pt.append(pw.numpy()); gt.append(g.numpy())

    embeds = torch.cat(embeds, dim=0)
    mlp_pp = torch.cat(mlp_pp, dim=0)
    mlp_gp = torch.cat(mlp_gp, dim=0)
    pt = np.concatenate(pt); gt = np.concatenate(gt)

    # TabPFN refinement (no-op for MLP-only models).
    if isinstance(model.pathway_head, TabPFNHead):
        mlp_pp = model.pathway_head.apply_tabpfn(embeds, mlp_pp)
    if isinstance(model.gene_head, TabPFNHead):
        mlp_gp = model.gene_head.apply_tabpfn(embeds, mlp_gp)

    return {
        "pathway": compute_metrics(mlp_pp.numpy(), pt),
        "gene":    compute_metrics(mlp_gp.numpy(), gt),
    }


def fit_tabpfn(model, train_loader, device):
    """Collect train embeddings + targets and fit TabPFN heads.

    Always also collects the MLP's in-sample predictions. Used by:
      - residual mode: TabPFN fit on (X, y - mlp_pred); top-k by residual var
      - refinement mode: TabPFN fit on (X, y); top-k by MLP residual var
        (i.e. where MLP is weakest — gives TabPFN the most room to help)
    """
    model.eval()
    Xs, yp, yg, ppm, gpm = [], [], [], [], []
    with torch.no_grad():
        for f, pw, g, _c in train_loader:
            f = f.to(device)
            h = model.forward_vision(f)
            Xs.append(h.cpu().numpy())
            yp.append(pw.numpy()); yg.append(g.numpy())
            ppm.append(model.pathway_head.mlp(h).cpu().numpy())
            gpm.append(model.gene_head.mlp(h).cpu().numpy())
    X = np.concatenate(Xs); yp = np.concatenate(yp); yg = np.concatenate(yg)
    ppm = np.concatenate(ppm); gpm = np.concatenate(gpm)
    mode = getattr(model, "tabpfn_mode", "refinement")
    print(f"    fitting TabPFN: X {X.shape} (mode={mode})")
    model.fit_tabpfn_heads(X, yp, yg, mlp_pathway_pred=ppm, mlp_gene_pred=gpm)


# ----------------------------------------------------------------------------
# Fold runner
# ----------------------------------------------------------------------------


def run_one_fold(
    features: torch.Tensor,
    pathways: torch.Tensor,
    genes: torch.Tensor,
    coords: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    device,
    fold_idx: int,
) -> Dict[str, Dict]:
    print(f"\n--- Fold {fold_idx+1}/{args.folds} | train={len(train_idx)} val={len(val_idx)} ---")
    feat_dim = features.shape[1]

    train_loader = make_tensor_loader(features, pathways, genes, coords, train_idx, args.batch_size, True)
    val_loader = make_tensor_loader(features, pathways, genes, coords, val_idx, args.batch_size, False)

    out = {}
    for head_type in ("mlp", "tabpfn"):
        label = "Baseline" if head_type == "mlp" else "TabPFN variant"
        print(f"  {label}")
        m = PEaRLCached(
            feat_dim=feat_dim,
            n_pathways=pathways.shape[1],
            n_genes=genes.shape[1],
            embed_dim=cfg.EMBED_DIM,
            pathway_hidden=cfg.PATHWAY_HIDDEN,
            head_type=head_type,
            tabpfn_top_k_pathways=args.tabpfn_top_k_pathways,
            tabpfn_top_k_genes=args.tabpfn_top_k_genes,
            tabpfn_mode=args.tabpfn_mode,
            tabpfn_n_estimators=args.tabpfn_n_estimators,
        ).to(device)

        stage1_contrastive(
            m, train_loader, val_loader, device,
            args.epochs_stage1, args.patience, args.lr, args.weight_decay, args.temperature,
        )
        stage2_supervised(
            m, train_loader, val_loader, device,
            args.epochs_stage2, args.patience, args.lr, args.weight_decay, head_type,
        )
        if head_type == "tabpfn":
            fit_tabpfn(m, train_loader, device)
        out["baseline" if head_type == "mlp" else "tabpfn"] = evaluate(m, val_loader, device)
        del m
        torch.cuda.empty_cache()

    return out


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def select_breast_section_ids(metadata_csv: str, n_sections: int, seed: int = 42) -> List[str]:
    df = pd.read_csv(metadata_csv)
    breast = df[(df.species == "Homo sapiens") & (df.organ == "Breast")].copy()
    breast = breast.sort_values("id").reset_index(drop=True)
    n = min(n_sections, len(breast))
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(breast), size=n, replace=False))
    return breast.iloc[pick]["id"].tolist()


def aggregate_folds(per_fold_results):
    out = {}
    for variant in ("baseline", "tabpfn"):
        out[variant] = {}
        for target in ("pathway", "gene"):
            agg = {}
            for k in ("PCC", "MSE", "MAE"):
                vals = np.array([f[variant][target][k] for f in per_fold_results], dtype=np.float64)
                agg[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
            agg["n_cols_used"] = int(np.median([f[variant][target]["n_cols_used"] for f in per_fold_results]))
            agg["n_cols_dropped"] = int(np.median([f[variant][target]["n_cols_dropped"] for f in per_fold_results]))
            out[variant][target] = agg
    return out


def print_summary(summary, paper):
    print("\n" + "=" * 88)
    print("CROSS-VALIDATED RESULTS (mean ± std)")
    print("=" * 88)
    for target in ("pathway", "gene"):
        print(f"\n{target.upper()} EXPRESSION")
        print(f"  {'Metric':<8} {'Baseline':<22} {'TabPFN':<22} {'Paper PEaRL':<22}")
        print("  " + "-" * 80)
        for k in ("PCC", "MSE", "MAE"):
            bm, bs = summary["baseline"][target][k]
            tm, ts = summary["tabpfn"][target][k]
            pm, ps = paper[target][k]
            print(f"  {k:<8} {bm:.4f}±{bs:.4f}        {tm:.4f}±{ts:.4f}        {pm:.4f}±{ps:.4f}")
        n_used = summary["baseline"][target]["n_cols_used"]
        n_dropped = summary["baseline"][target]["n_cols_dropped"]
        print(f"  cols used: {n_used}, constant cols dropped: {n_dropped}")


PAPER_BASELINE_BREAST = {
    "gene":    {"PCC": (0.5868, 0.0359), "MSE": (0.0732, 0.0033), "MAE": (0.1828, 0.0043)},
    "pathway": {"PCC": (0.5055, 0.0271), "MSE": (0.0017, 0.0001), "MAE": (0.0314, 0.0010)},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./hest_data")
    p.add_argument("--metadata-csv", default="./hest_data/HEST_v1_1_0.csv")
    p.add_argument("--output-dir", default="./reproduction_results")
    p.add_argument("--n-sections", type=int, default=36)
    p.add_argument("--max-spots-per-section", type=int, default=400)
    p.add_argument("--n-genes", type=int, default=1000)
    p.add_argument("--n-pathways", type=int, default=775)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=None,
                   help="Run only the first N folds (still split into --folds).")
    p.add_argument("--epochs-stage1", type=int, default=100)
    p.add_argument("--epochs-stage2", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--feat-batch-size", type=int, default=64,
                   help="Batch size for the one-shot UNI feature extraction.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--encoder", choices=["uni", "vit"], default="uni")
    p.add_argument(
        "--normalization", choices=["paper", "paper_log1p_only", "paper_zscore"],
        default="paper",
        help=(
            "Gene-target normalization. paper=per-gene min-max [0,1] (default). "
            "paper_log1p_only=log1p only, no per-gene scaling. "
            "paper_zscore=log1p + per-gene z-score."
        ),
    )
    p.add_argument("--tabpfn-top-k-pathways", type=int, default=20)
    p.add_argument("--tabpfn-top-k-genes", type=int, default=50)
    p.add_argument(
        "--tabpfn-mode", choices=["refinement", "residual"], default="refinement",
        help=(
            "refinement: TabPFN replaces MLP prediction on top-k highest-target-variance dims. "
            "residual:   TabPFN predicts the MLP's residual on top-k highest-MLP-residual-variance "
            "dims and is added back. Strictly stacks on top of the MLP."
        ),
    )
    p.add_argument("--tabpfn-n-estimators", type=int, default=4,
                   help="TabPFN ensemble size. Higher → better quality, slower.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split", choices=["section", "spot"], default="section",
        help=(
            "section = GroupKFold by section (no leakage; held-out sections "
            "test cross-section generalization). spot = KFold by spot (spots "
            "from same section in both train and val; matches what the PEaRL "
            "paper most likely did, easier task with much higher PCC)."
        ),
    )
    p.add_argument("--smoke-test", action="store_true",
                   help="Quick mode: 5 sections, 2 folds, 5 epochs.")
    args = p.parse_args()

    if args.smoke_test:
        args.n_sections = 5
        args.folds = 2
        args.epochs_stage1 = 5
        args.epochs_stage2 = 5
        args.patience = 3
        args.n_pathways = 200
        args.max_spots_per_section = 100

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}, Encoder: {args.encoder}")
    print(f"Sections: {args.n_sections}, max_spots/section: {args.max_spots_per_section}")
    print(f"Folds: {args.folds}, n_pathways: {args.n_pathways}, batch_size: {args.batch_size}")

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
    print(f"Data loaded in {time.time()-t0:.1f}s")

    # One-shot feature extraction with frozen UNI/ViT.
    print("\nExtracting backbone features (one-shot) ...")
    t0 = time.time()
    encoder = VisionEncoder(
        embed_dim=cfg.EMBED_DIM, pretrained=True, backbone=args.encoder, freeze_backbone=True
    ).to(device)
    ds = HESTDataset(patches, genes, pathways, coords, sample_id="multi")
    features_cpu = extract_features(encoder, ds, device, args.feat_batch_size)
    del encoder
    torch.cuda.empty_cache()
    print(f"Features {tuple(features_cpu.shape)} in {time.time()-t0:.1f}s")

    # Tensorize once. Keep on CPU; loader moves per-batch to GPU.
    pathways_t = torch.from_numpy(pathways)
    genes_t = torch.from_numpy(genes)
    coords_t = torch.from_numpy(coords)

    if args.split == "section":
        kf = GroupKFold(n_splits=args.folds)
        fold_iter = list(kf.split(np.arange(len(patches)), groups=section_ids))
        print("Splitting BY SECTION (held-out sections; rigorous generalization).")
    else:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_iter = list(kf.split(np.arange(len(patches))))
        print("Splitting BY SPOT (mixed sections; easier task, matches paper convention).")
    if args.max_folds is not None:
        fold_iter = fold_iter[: args.max_folds]
        print(f"Limiting to first {len(fold_iter)} fold(s) of {args.folds}")

    fold_results = []
    for fi, (train_idx, val_idx) in enumerate(fold_iter):
        t0 = time.time()
        res = run_one_fold(
            features_cpu, pathways_t, genes_t, coords_t,
            train_idx, val_idx, args, device, fi,
        )
        res["_fold_seconds"] = time.time() - t0
        fold_results.append(res)
        with open(os.path.join(args.output_dir, "fold_results.json"), "w") as f:
            json.dump(fold_results, f, indent=2, default=str)

    summary = aggregate_folds(fold_results)
    out = {
        "args": vars(args),
        "sample_ids": sample_ids,
        "per_fold": fold_results,
        "summary": summary,
        "paper_breast_baseline": PAPER_BASELINE_BREAST,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "reproduction_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    print_summary(summary, PAPER_BASELINE_BREAST)
    print(f"\nResults saved to {args.output_dir}/reproduction_results.json")


if __name__ == "__main__":
    main()

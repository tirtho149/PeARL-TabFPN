"""Stage 3 — train a regression head on frozen LA-3B embeddings; compare to PeARL.

Same data, same 5-fold spot splits, same metric (pearl_tabpfn.eval.compute_metrics)
as the PeARL baseline — the ONLY difference is the image encoder. "Minimum
fine-tune": the LA-3B backbone is frozen (embeddings precomputed) and only this
small MLP head trains, with the paper's optimizer (AdamW lr 1e-4, wd 1e-3, ≤100
epochs, early stopping patience 15). Reports ALL metrics (PCC, MSE, MAE) for gene
and pathway against the cohort's paper reference.

    python la_regression/train_head.py --cohort breast

Run in the PeARL venv (needs scipy). GPU optional.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pearl_tabpfn.eval import compute_metrics  # noqa: E402
from pearl_tabpfn.reproduction import COHORTS  # noqa: E402

HERE = os.path.dirname(__file__)


class Head(nn.Module):
    def __init__(self, d_in, n_gene, n_path, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.gene = nn.Linear(hidden, n_gene)
        self.path = nn.Linear(hidden, n_path)

    def forward(self, x):
        h = self.trunk(x)
        return self.gene(h), self.path(h)


def run_fold(Xtr, ytr_g, ytr_p, Xva, yva_g, yva_p, device, epochs=100, patience=15):
    head = Head(Xtr.shape[1], ytr_g.shape[1], ytr_p.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-3)
    mse = nn.MSELoss()
    Xtr = torch.tensor(Xtr, device=device); Xva = torch.tensor(Xva, device=device)
    ytr_g = torch.tensor(ytr_g, device=device); ytr_p = torch.tensor(ytr_p, device=device)
    yva_g_t = torch.tensor(yva_g, device=device); yva_p_t = torch.tensor(yva_p, device=device)
    n = Xtr.shape[0]; bs = 128
    best = (1e9, None); bad = 0
    for _ in range(epochs):
        head.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pg, pp = head(Xtr[idx])
            loss = mse(pg, ytr_g[idx]) + mse(pp, ytr_p[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vg, vp = head(Xva)
            vl = (mse(vg, yva_g_t) + mse(vp, yva_p_t)).item()
        if vl < best[0] - 1e-6:
            best = (vl, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best[1]); head.eval()
    with torch.no_grad():
        pg, pp = head(Xva)
    return pg.cpu().numpy(), pp.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["breast", "skin", "lymph"], default="breast")
    args = ap.parse_args()
    emb = os.path.join(HERE, f"la_embeddings_{args.cohort}.npz")
    out = os.path.join(HERE, f"la_results_{args.cohort}.json")
    paper = COHORTS[args.cohort]["paper"]

    d = np.load(emb)
    X = d["embeddings"].astype(np.float32)
    g = d["genes"].astype(np.float32); p = d["pathways"].astype(np.float32)
    fold = d["fold"]
    coords = d["coords"] if "coords" in d else np.zeros((len(fold), 2), np.float32)
    section_ids = d["section_ids"] if "section_ids" in d else np.zeros(len(fold), np.int64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.cohort}] LA embeddings {X.shape} | device {device}")

    # Accumulate per-spot validation predictions across folds (each spot is in
    # exactly one val fold) so the head-to-head figures can draw spatial maps and
    # gene/pathway correlation matrices, matching PeARL's predictions/fold_*.npz.
    gene_pred = np.zeros_like(g); path_pred = np.zeros_like(p)
    per_fold = []
    for fi in range(5):
        va = fold == fi; tr = ~va
        pg, pp = run_fold(X[tr], g[tr], p[tr], X[va], g[va], p[va], device)
        gene_pred[va] = pg; path_pred[va] = pp
        gm = compute_metrics(pg, g[va], drop_constant_cols=False)
        pm = compute_metrics(pp, p[va], drop_constant_cols=False)
        per_fold.append({"gene": gm, "pathway": pm})
        print(f"fold {fi}: gene PCC={gm['PCC_per_dim_mean']:.4f} pathway PCC={pm['PCC_per_dim_mean']:.4f}")

    np.savez(os.path.join(HERE, f"la_predictions_{args.cohort}.npz"),
             gene_pred=gene_pred, gene_true=g, path_pred=path_pred, path_true=p,
             coords=coords, section_ids=section_ids, fold=fold)

    keys = ("PCC_per_dim_mean", "PCC", "SCC_per_dim_mean", "R2_per_dim_mean", "MSE", "RMSE", "MAE")
    summary = {}
    for tgt in ("gene", "pathway"):
        agg = {}
        for k in keys:
            v = np.array([f[tgt].get(k, np.nan) for f in per_fold], dtype=np.float64)
            agg[k] = (float(np.nanmean(v)), float(np.nanstd(v)))
        agg["paper"] = paper[tgt]
        summary[tgt] = agg
    json.dump({"cohort": args.cohort, "per_fold": per_fold, "summary": summary},
              open(out, "w"), indent=2, default=str)

    # paper has only PCC/MSE/MAE; map onto our rows, "—" elsewhere.
    paper_row = {"PCC_per_dim_mean": "PCC", "MSE": "MSE", "MAE": "MAE"}
    label = {"PCC_per_dim_mean": "PCC_perdim", "PCC": "PCC_flat",
             "SCC_per_dim_mean": "SCC_perdim", "R2_per_dim_mean": "R2_perdim",
             "MSE": "MSE", "RMSE": "RMSE", "MAE": "MAE"}
    print(f"\n=== LA-3B regressor (frozen tower + MLP head) — cohort={args.cohort}, 5-fold ===")
    for tgt in ("gene", "pathway"):
        s = summary[tgt]; pp = s["paper"]
        print(f"  -- {tgt} --   {'metric':<12}{'LA-3B (ours)':<20}{'PEaRL paper':<18}")
        for k in keys:
            ours = f"{s[k][0]:.4f}±{s[k][1]:.4f}"
            pk = paper_row.get(k)
            ref = f"{pp[pk][0]:.4f}±{pp[pk][1]:.4f}" if pk else "—"
            print(f"  {'':<12}{label[k]:<12}{ours:<20}{ref:<18}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

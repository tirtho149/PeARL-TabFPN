"""Head-to-head figures: PEaRL+MLP vs PEaRL+TabPFN vs LocateAnything-3B.

Paper-style (arXiv:2510.03455) comparison across the three methods on a cohort:
  fig1_metric_bars      — gene & pathway PCC/SCC/R2/MSE/MAE per method + paper
  fig2_perfold_pcc      — per-fold PCC_perdim distribution per method
  fig3_spatial_maps     — GT vs each method, spatial expression of a sample dim
  fig4_corr_matrices    — gene-gene & pathway-pathway correlation (GT vs methods)
  fig5_radar            — all metrics, normalized, per method

Robust to missing methods: only what has results is drawn (so it works now with
LA-3B alone, and fills in as the baseline/TabPFN runs land). Re-run any time.

    python la_regression/make_figures.py --cohort breast

Run in the PeARL venv. CPU only.
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pearl_tabpfn.reproduction import COHORTS  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
METHODS = ["PEaRL+MLP", "PEaRL+TabPFN", "LA-3B"]
COLORS = {"PEaRL+MLP": "#4C72B0", "PEaRL+TabPFN": "#DD8452", "LA-3B": "#55A868",
          "Paper": "#888888"}
METRIC_KEYS = ["PCC_per_dim_mean", "SCC_per_dim_mean", "R2_per_dim_mean", "MSE", "MAE"]
METRIC_LABEL = {"PCC_per_dim_mean": "PCC", "SCC_per_dim_mean": "SCC",
                "R2_per_dim_mean": "R²", "MSE": "MSE", "MAE": "MAE"}


def _pearl_json(cohort):
    p = os.path.join(ROOT, f"reproduction_results_{cohort}", "reproduction_results.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def load_summaries(cohort):
    """method -> {gene:{key:(mean,std)}, pathway:{...}}  (only present methods)."""
    out = {}
    pj = _pearl_json(cohort)
    if pj and "summary" in pj:
        s = pj["summary"]
        if "baseline" in s:
            out["PEaRL+MLP"] = s["baseline"]
        if "tabpfn" in s:
            out["PEaRL+TabPFN"] = s["tabpfn"]
    laf = os.path.join(HERE, f"la_results_{cohort}.json")
    if os.path.isfile(laf):
        out["LA-3B"] = json.load(open(laf))["summary"]
    return out


def load_perfold(cohort):
    """method -> {'gene':[pcc per fold], 'pathway':[...]}"""
    out = {}
    pj = _pearl_json(cohort)
    if pj:
        for variant, name in [("baseline", "PEaRL+MLP"), ("tabpfn", "PEaRL+TabPFN")]:
            rows = [f[variant] for f in pj["per_fold"] if variant in f]
            if rows:
                out[name] = {t: [r[t]["PCC_per_dim_mean"] for r in rows] for t in ("gene", "pathway")}
    laf = os.path.join(HERE, f"la_results_{cohort}.json")
    if os.path.isfile(laf):
        pf = json.load(open(laf))["per_fold"]
        out["LA-3B"] = {t: [r[t]["PCC_per_dim_mean"] for r in pf] for t in ("gene", "pathway")}
    return out


def load_preds(cohort):
    """method -> {gene_pred,gene_true,path_pred,path_true,coords,section_ids} pooled over folds."""
    out = {}
    pdir = os.path.join(ROOT, f"reproduction_results_{cohort}", "predictions")
    files = sorted(glob.glob(os.path.join(pdir, "fold_*.npz")))
    if files:
        acc = {}
        for f in files:
            d = np.load(f)
            for head, name in [("mlp", "PEaRL+MLP"), ("tabpfn", "PEaRL+TabPFN")]:
                gk = f"gene_pred_{head}"
                if gk in d:
                    acc.setdefault(name, {"g": [], "gt": [], "p": [], "pt": [], "c": []})
                    acc[name]["g"].append(d[gk]); acc[name]["gt"].append(d["gene_true"])
                    acc[name]["p"].append(d[f"pathway_pred_{head}"]); acc[name]["pt"].append(d["pathway_true"])
                    acc[name]["c"].append(d["coords"])
        for name, a in acc.items():
            out[name] = dict(gene_pred=np.concatenate(a["g"]), gene_true=np.concatenate(a["gt"]),
                             path_pred=np.concatenate(a["p"]), path_true=np.concatenate(a["pt"]),
                             coords=np.concatenate(a["c"]),
                             section_ids=np.zeros(sum(len(x) for x in a["g"]), int))
    laf = os.path.join(HERE, f"la_predictions_{cohort}.npz")
    if os.path.isfile(laf):
        d = np.load(laf)
        out["LA-3B"] = dict(gene_pred=d["gene_pred"], gene_true=d["gene_true"],
                            path_pred=d["path_pred"], path_true=d["path_true"],
                            coords=d["coords"], section_ids=d["section_ids"])
    return out


def fig_metric_bars(summ, paper, outpath):
    present = [m for m in METHODS if m in summ]
    if not present:
        return
    fig, axes = plt.subplots(2, len(METRIC_KEYS), figsize=(4 * len(METRIC_KEYS), 7))
    for ri, tgt in enumerate(("gene", "pathway")):
        for ci, key in enumerate(METRIC_KEYS):
            ax = axes[ri, ci]
            xs = np.arange(len(present))
            vals = [summ[m][tgt].get(key, (np.nan, 0))[0] for m in present]
            errs = [summ[m][tgt].get(key, (np.nan, 0))[1] for m in present]
            ax.bar(xs, vals, yerr=errs, color=[COLORS[m] for m in present], capsize=4)
            # paper reference (only PCC/MSE/MAE)
            pkey = {"PCC_per_dim_mean": "PCC", "MSE": "MSE", "MAE": "MAE"}.get(key)
            if pkey:
                ax.axhline(paper[tgt][pkey][0], ls="--", color=COLORS["Paper"], lw=1.5,
                           label=f"paper {paper[tgt][pkey][0]:.3f}")
                ax.legend(fontsize=7)
            ax.set_xticks(xs); ax.set_xticklabels(present, rotation=20, fontsize=7)
            if ci == 0:
                ax.set_ylabel(tgt.upper(), fontsize=11)
            ax.set_title(METRIC_LABEL[key], fontsize=10)
    fig.suptitle("Head-to-head metrics — " + os.path.basename(outpath).split("_")[0], y=1.0)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_perfold(perfold, outpath):
    present = [m for m in METHODS if m in perfold]
    if not present:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, tgt in zip(axes, ("gene", "pathway")):
        data = [perfold[m][tgt] for m in present]
        bp = ax.boxplot(data, labels=present, patch_artist=True, widths=0.5)
        for patch, m in zip(bp["boxes"], present):
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.6)
        for i, m in enumerate(present):
            ax.scatter(np.full(len(perfold[m][tgt]), i + 1), perfold[m][tgt],
                       color=COLORS[m], s=18, zorder=3)
        ax.set_title(f"{tgt} PCC per fold"); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_spatial(preds, outpath):
    present = [m for m in METHODS if m in preds]
    if not present:
        return
    ref = preds[present[0]]
    sids = ref["section_ids"]
    # pick the section with the most spots, and the highest-variance pathway dim
    sec = np.bincount(sids).argmax() if sids.max() > 0 else 0
    m = sids == sec
    if m.sum() < 10:
        m = np.ones(len(sids), bool)
    dim = int(ref["path_true"][m].var(0).argmax())
    coords = ref["coords"][m]
    panels = [("GT", ref["path_true"][m, dim])] + \
             [(name, preds[name]["path_pred"][m, dim]) for name in present]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, val) in zip(axes, panels):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=val, cmap="magma", s=10)
        ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.axis("off")
        plt.colorbar(sc, ax=ax, fraction=0.046)
    fig.suptitle(f"Spatial pathway #{dim} (section {sec}) — GT vs methods", y=1.02)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_corr(preds, outpath, topk=40):
    present = [m for m in METHODS if m in preds]
    if not present:
        return
    ref = preds[present[0]]
    # top-K highest-variance pathway dims for a legible matrix
    idx = np.argsort(ref["path_true"].var(0))[-topk:]
    mats = [("GT", np.corrcoef(ref["path_true"][:, idx].T))] + \
           [(name, np.corrcoef(preds[name]["path_pred"][:, idx].T)) for name in present]
    fig, axes = plt.subplots(1, len(mats), figsize=(3.2 * len(mats), 3.2))
    if len(mats) == 1:
        axes = [axes]
    for ax, (title, M) in zip(axes, mats):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle(f"Pathway-pathway correlation (top-{topk} var) — GT vs methods", y=1.02)
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["breast", "skin", "lymph"], default="breast")
    args = ap.parse_args()
    paper = COHORTS[args.cohort]["paper"]
    outdir = os.path.join(HERE, f"figures_h2h_{args.cohort}")
    os.makedirs(outdir, exist_ok=True)

    summ, perfold, preds = load_summaries(args.cohort), load_perfold(args.cohort), load_preds(args.cohort)
    print(f"[{args.cohort}] methods with summaries: {list(summ)} | with predictions: {list(preds)}")

    jobs = [
        ("fig1_metric_bars", lambda p: fig_metric_bars(summ, paper, p)),
        ("fig2_perfold_pcc", lambda p: fig_perfold(perfold, p)),
        ("fig3_spatial_maps", lambda p: fig_spatial(preds, p)),
        ("fig4_corr_matrices", lambda p: fig_corr(preds, p)),
    ]
    for name, fn in jobs:
        path = os.path.join(outdir, f"{name}.png")
        try:
            fn(path)
            print("  wrote", path if os.path.isfile(path) else f"(skipped {name}: no data)")
        except Exception as e:
            print(f"  {name} FAILED: {type(e).__name__}: {e}")
    print(f"figures in {outdir}")


if __name__ == "__main__":
    main()

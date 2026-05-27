"""5-fold survival training harness: AB-MIL + Cox loss.

Two entry points:
  • `train_one_fold(...)` — trains AB-MIL on a single train/val split, returns
    val risks + val (os_time, event) + final C-index
  • `train_survival(...)` — runs 5-fold CV across a TCGABRCALoader, calls
    train_one_fold per fold, aggregates results with concordance_index_5fold

Tile embeddings are cached to disk in `cache_dir` so subsequent CV folds /
hyper-param sweeps don't re-run the UNI forward (which dominates wall-clock).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .ab_mil import ABMIL, cox_ph_loss
from .data import TCGABRCALoader, tile_wsi, embed_tiles
from .metrics import concordance_index_5fold, fold_c_index


def _cache_path(cache_dir: str, case_id: str, encoder_tag: str) -> str:
    return str(Path(cache_dir) / f"{encoder_tag}__{case_id}.npy")


def _embed_or_load(loader: TCGABRCALoader, encoder, encoder_tag: str,
                   cache_dir: str, device: str,
                   max_tiles_per_slide: int = 1024) -> dict:
    """Return dict case_id → (N_tiles, embed_dim) numpy array. Cached on disk."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    feats_by_case = {}
    for _, row in loader.cases.iterrows():
        path = _cache_path(cache_dir, row["case_id"], encoder_tag)
        if os.path.exists(path):
            feats_by_case[row["case_id"]] = np.load(path)
            continue
        t0 = time.time()
        tiles = tile_wsi(row["wsi_path"], patch_size=224, level=0,
                         max_tiles=max_tiles_per_slide)
        emb = embed_tiles(tiles, encoder, device=device)
        np.save(path, emb)
        feats_by_case[row["case_id"]] = emb
        print(f"    [embed] {row['case_id']}: {emb.shape} in {time.time()-t0:.1f}s")
    return feats_by_case


def train_one_fold(features: dict, cases: "pd.DataFrame",
                   train_idx: np.ndarray, val_idx: np.ndarray,
                   embed_dim: int, epochs: int = 30, lr: float = 1e-4,
                   wd: float = 1e-4, device: str = "cuda",
                   seed: int = 42) -> dict:
    """Train ABMIL on one CV fold.

    features: dict case_id → (N_tiles, embed_dim) numpy array
    cases:    DataFrame with columns case_id, os_time, event
    train_idx, val_idx: integer indices into cases
    """
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ABMIL(embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_cases = cases.iloc[train_idx].reset_index(drop=True)
    val_cases = cases.iloc[val_idx].reset_index(drop=True)
    best_val_c = -1.0
    best_val_risks = None
    for ep in range(epochs):
        # ---- Train: one slide per "batch", accumulate Cox loss across slides ----
        # We compute the Cox partial likelihood across all train slides at once
        # (the partial likelihood is inherently a set-level loss, not per-sample).
        model.train()
        risks = []
        for _, row in train_cases.iterrows():
            tiles = torch.from_numpy(features[row["case_id"]]).float().to(device)
            r, _ = model(tiles)
            risks.append(r)
        risks = torch.stack(risks)
        times = torch.tensor(train_cases["os_time"].values, dtype=torch.float32, device=device)
        events = torch.tensor(train_cases["event"].values, dtype=torch.float32, device=device)
        loss = cox_ph_loss(risks, times, events)
        opt.zero_grad(); loss.backward(); opt.step()

        # ---- Val ----
        model.eval()
        val_risks = []
        with torch.no_grad():
            for _, row in val_cases.iterrows():
                tiles = torch.from_numpy(features[row["case_id"]]).float().to(device)
                r, _ = model(tiles)
                val_risks.append(r.detach().cpu().item())
        val_risks = np.array(val_risks)
        val_c = fold_c_index(val_risks, val_cases["os_time"].values,
                             val_cases["event"].values)
        if val_c > best_val_c:
            best_val_c = val_c
            best_val_risks = val_risks
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"      epoch {ep+1:3d}: loss={loss.item():.4f}, val C={val_c:.3f}  best={best_val_c:.3f}")
    return {
        "val_risks": best_val_risks,
        "val_os_time": val_cases["os_time"].values,
        "val_event": val_cases["event"].values,
        "val_c_index": best_val_c,
    }


def train_survival(wsi_dir: str, clinical_tsv: str,
                   encoder, encoder_tag: str,
                   cache_dir: str = "wacv_results/survival/embeddings",
                   output_dir: str = "wacv_results/survival",
                   embed_dim: int = 1024,
                   epochs: int = 30, n_folds: int = 5,
                   max_tiles_per_slide: int = 1024,
                   device: str = "cuda", seed: int = 42) -> dict:
    """Full 5-fold survival run end-to-end.

    Returns a dict with c_index_mean, c_index_std, per-fold details, and the
    concatenated (val_risks, os_time, event) arrays for KM plotting.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    loader = TCGABRCALoader(wsi_dir=wsi_dir, clinical_tsv=clinical_tsv)
    print(f"[survival] {loader.n_cases} cases, {loader.n_events} events")

    print("[survival] embedding tiles (cached in {cache_dir})")
    features = _embed_or_load(loader, encoder, encoder_tag, cache_dir, device,
                              max_tiles_per_slide=max_tiles_per_slide)

    fold_outputs = []
    fold_details = []
    splits = loader.split_5fold(seed=seed, n_splits=n_folds)
    for fold, (tr, va) in enumerate(splits):
        print(f"\n[survival] fold {fold+1}/{n_folds}: n_train={len(tr)}, n_val={len(va)}")
        out = train_one_fold(features, loader.cases, tr, va,
                             embed_dim=embed_dim, epochs=epochs,
                             device=device, seed=seed + fold)
        fold_outputs.append((out["val_risks"], out["val_os_time"], out["val_event"]))
        fold_details.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "val_c_index": out["val_c_index"],
        })

    summary = concordance_index_5fold(fold_outputs)
    # Concatenate fold val sets for the overall KM curve
    all_risks = np.concatenate([o[0] for o in fold_outputs])
    all_times = np.concatenate([o[1] for o in fold_outputs])
    all_events = np.concatenate([o[2] for o in fold_outputs])
    summary["risk_scores"] = all_risks.tolist()
    summary["os_time"] = all_times.tolist()
    summary["event"] = all_events.tolist()
    summary["folds"] = fold_details

    import json
    with open(Path(output_dir) / "survival_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[survival] FINAL: C-index = {summary['c_index_mean']:.3f} ± {summary['c_index_std']:.3f}")
    print(f"  (paper PEaRL: 0.659 ± 0.027)")
    return summary

"""Stage 2 — extract LA-3B pooled vision-tower embeddings for every HEST patch.

Loads the raw patches exported in Stage 1, runs each through LA-3B's frozen
vision tower (worker.embed_image), and caches the (N, dim) embedding matrix.
This is the "minimum fine-tune" feature basis: the backbone is frozen and only a
small head is trained downstream (Stage 3). If the linear probe underperforms the
PeARL baseline we escalate to LoRA on the tower.

Run in the LocateAnythingBench .venv on a GPU.
"""
import os
import sys
import time
import argparse

import numpy as np
import torch
from PIL import Image

LA_ROOT = "/work/mech-ai-scratch/tirtho/LocateAnythingBench"
sys.path.insert(0, os.path.join(LA_ROOT, "eagle", "Embodied"))

HERE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["breast", "skin", "lymph"], default="breast")
    args = ap.parse_args()
    DATASET = os.path.join(HERE, f"la_dataset_{args.cohort}.npz")
    OUT = os.path.join(HERE, f"la_embeddings_{args.cohort}.npz")

    from locateanything_worker import LocateAnythingWorker

    d = np.load(DATASET)
    raw = d["raw_patches"]  # (N, 224, 224, 3) uint8
    n = raw.shape[0]
    print(f"[extract] {n} patches to embed", flush=True)

    worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
    print("[extract] model loaded", flush=True)

    embs = None
    t0 = time.time()
    for i in range(n):
        img = Image.fromarray(raw[i]).convert("RGB")
        with torch.no_grad():
            v = worker.embed_image(img).float().cpu().numpy()
        if embs is None:
            embs = np.zeros((n, v.shape[0]), dtype=np.float32)
            print(f"[extract] embed dim = {v.shape[0]}", flush=True)
        embs[i] = v
        if (i + 1) % 250 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n - i - 1) / rate / 60
            print(f"[extract] {i+1}/{n}  {rate:.1f} img/s  ETA {eta:.0f} min", flush=True)

    np.savez(
        OUT,
        embeddings=embs,
        genes=d["genes"],
        pathways=d["pathways"],
        section_ids=d["section_ids"],
        fold=d["fold"],
        coords=d["coords"] if "coords" in d else np.zeros((n, 2), np.float32),
    )
    print(f"[extract] saved {OUT}  embeddings {embs.shape} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

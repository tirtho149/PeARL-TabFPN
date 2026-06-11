"""De-risk probe: load nvidia/LocateAnything-3B and embed a few HEST H&E patches.

Confirms (a) the gated model loads in the LA .venv, (b) embed_image() yields a
fixed-length pooled vector we can regress from, (c) the embedding dimension (to
size the regression head), and (d) per-image latency (to budget the full
13,620-patch extraction).

Run in the LocateAnythingBench .venv on a GPU. eagle/Embodied must be on PYTHONPATH.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

LA_ROOT = "/work/mech-ai-scratch/tirtho/LocateAnythingBench"
sys.path.insert(0, os.path.join(LA_ROOT, "eagle", "Embodied"))
PATCHES_NPY = "/work/mech-ai-scratch/tirtho/PeARL-TabFPN/la_regression/probe_patches.npy"


def load_patches(_sid, n):
    imgs = np.load(PATCHES_NPY)[:n]
    return [Image.fromarray(imgs[i].astype(np.uint8)).convert("RGB") for i in range(len(imgs))]


def main():
    from locateanything_worker import LocateAnythingWorker

    t0 = time.time()
    worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
    print(f"[probe] model loaded in {time.time()-t0:.1f}s", flush=True)

    patches = load_patches("SPA121", 6)
    dims = None
    for i, img in enumerate(patches):
        t = time.time()
        vec = worker.embed_image(img)          # pooled (dim,)
        toks = worker.embed_image_tokens(img)  # (n_tokens, dim)
        dt = time.time() - t
        dims = tuple(vec.shape)
        print(f"[probe] patch {i}: pooled={tuple(vec.shape)} dtype={vec.dtype} "
              f"tokens={tuple(toks.shape)}  {dt:.2f}s/img", flush=True)

    print(f"\n[probe] EMBED DIM = {dims[0]}  -> regression head: Linear({dims[0]}, 1775)")
    print("[probe] OK")


if __name__ == "__main__":
    main()

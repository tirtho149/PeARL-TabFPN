#!/usr/bin/env python3
"""Train survival arm (Figure/Table 3 of arXiv:2510.03455).

5-fold AB-MIL + Cox partial-likelihood on TCGA-BRCA WSIs. Encoder defaults to
UNI v1 (matches the paper); pass --encoder vit-l-16 for the ImageNet fallback.

Prereqs:
  • TCGA-BRCA WSIs on disk (download via `gdc-client download -m manifest.txt`)
  • Clinical TSV from the GDC /cases endpoint (smoke_survival.py downloads it)
  • CUDA GPU available (training is hours per fold even with cached embeddings)

Usage:
    python scripts/train_survival.py \
        --wsi-dir /scratch/tcga_brca/wsi \
        --clinical-tsv tcga_smoke/brca_survival.tsv \
        --output-dir wacv_results/survival
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make src/ importable without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wsi-dir", required=True, help="Directory containing TCGA-BRCA .svs files.")
    ap.add_argument("--clinical-tsv", required=True,
                    help="GDC /cases TSV with vital_status + days columns.")
    ap.add_argument("--output-dir", default="wacv_results/survival")
    ap.add_argument("--cache-dir", default="wacv_results/survival/embeddings")
    ap.add_argument("--encoder", default="uni", choices=["uni", "vit-l-16"],
                    help="UNI v1 (paper-faithful) or ImageNet ViT-L/16.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-tiles-per-slide", type=int, default=1024,
                    help="Random subsample after tissue masking. 1024 ≈ 0.5 GB/slide cached.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("ERROR: training requires CUDA but torch.cuda.is_available() is False.", file=sys.stderr)
        print("       See scripts/smoke_gpu.py for diagnostic checklist.", file=sys.stderr)
        return 1

    print(f"==> Building encoder: {args.encoder}")
    from pearl_tabpfn.encoders import VisionEncoder
    encoder = VisionEncoder(
        embed_dim=1024, pretrained=True,
        backbone="uni" if args.encoder == "uni" else "vit_l_16",
        freeze_backbone=True,
    )
    encoder.embed_dim = 1024  # for cache tag
    encoder_tag = args.encoder.replace("-", "_")

    from pearl_tabpfn.survival import train_survival
    result = train_survival(
        wsi_dir=args.wsi_dir,
        clinical_tsv=args.clinical_tsv,
        encoder=encoder,
        encoder_tag=encoder_tag,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        embed_dim=1024,
        epochs=args.epochs,
        n_folds=args.n_folds,
        max_tiles_per_slide=args.max_tiles_per_slide,
        device="cuda",
        seed=args.seed,
    )
    print(f"\n==> Results written to {args.output_dir}/survival_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

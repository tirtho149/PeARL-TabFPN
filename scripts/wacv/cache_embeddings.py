#!/usr/bin/env python
"""WACV Section 1 — per-(cohort, fold) PEaRL embedding cache.

Protocol (docs/WACV_PIPELINE.md, Section 1):

    For each cohort, for each of the 5 GroupKFold folds:
      1. Run Stage 1 (contrastive pretraining with UNI's last-4 blocks
         unfrozen) on the fold's train spots.
      2. Freeze the encoder.
      3. Forward-pass over ALL spots in the fold → cache 256-d embeddings.
      4. Save per fold:
           H_train (N_train, 256), H_val (N_val, 256),
           y_gene_{train,val}, y_path_{train,val},
           section_id arrays for both splits.
      5. Verify no section_id appears in both train and val of the same
         fold. If it does, the split is leaking — abort and fix.

After this step the GPU is essentially free for Phases 1–5 (except for
the large-context Lymph runs the Phase-0e probe rules on).

Status: SCAFFOLDED. The current cleanest path is to add a
`--export-embeddings` mode to `pearl_tabpfn.reproduction.main` that
exits after Stage 1 and dumps the embeddings. Until that lands, Phases
1–5 can run *without* the cache by calling `reproduction.main` end-to-
end per cohort — slower but correct.

This script currently:
  - validates the directory structure expected by Phases 1–5,
  - prints the exact protocol it would execute,
  - exits non-zero so a CI/SLURM submission can't silently no-op.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pearl_tabpfn.config import cfg  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohort", choices=["Breast", "Skin", "Lymph"], required=True,
        help="HEST-1k cohort to cache embeddings for.",
    )
    p.add_argument("--output-dir", default="./wacv_results/embeddings_cache")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--n-sections", type=int, default=None,
        help="Number of sections to draw from this cohort. Defaults to "
             "36 for Breast, 24 for Skin/Lymph (Phase 0e may adjust Lymph).",
    )
    args = p.parse_args()

    n_sections = args.n_sections or (36 if args.cohort == "Breast" else 24)
    cohort_dir = os.path.join(args.output_dir, args.cohort)
    os.makedirs(cohort_dir, exist_ok=True)

    print(
        f"\n[cache_embeddings] cohort={args.cohort} "
        f"n_sections={n_sections} folds={args.folds}\n"
        f"  cfg.HEST_IDS[{args.cohort}] = {cfg.HEST_IDS[args.cohort]}\n"
        f"  cfg.DATASET_PATHWAYS[{args.cohort}] = "
        f"{cfg.DATASET_PATHWAYS[args.cohort]}\n"
        f"  output: {cohort_dir}/{{cohort}}_{{fold}}_{{split}}.npz\n"
    )
    print(
        "Protocol:\n"
        "  for fold in 0..N-1:\n"
        "    1. GroupKFold split on section_id\n"
        "    2. Stage-1 contrastive pretrain (UNI last-4 unfrozen) on train\n"
        "    3. Freeze; forward over train+val; save (H, y_gene, y_path, sec_id)\n"
        "    4. Assert disjoint train/val section_id sets\n"
    )

    # TODO(implementation):
    #   Either (a) add `--export-embeddings` to reproduction.main and call
    #   it here, or (b) lift Stage-1 + extract_features into a standalone
    #   helper. (a) is preferred — one trained encoder, one source of truth.
    print(
        "[cache_embeddings] NOT YET IMPLEMENTED — see docstring.\n"
        "  Today: Phases 1–5 should call `scripts/train_tabpfn3.py "
        "--cohort {} --apple-to-apple ...` directly. Caching is an "
        "optimization, not a correctness requirement.".format(args.cohort)
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

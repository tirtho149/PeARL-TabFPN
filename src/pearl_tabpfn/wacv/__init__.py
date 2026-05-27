"""WACV B-track characterization utilities for the PEaRL+TabPFN-3 combination.

Modules:
  - `calibration` — reliability diagram, ECE, selective-prediction curve.
  - `pathway_maps` — predicted vs ssGSEA spatial PCC per pathway + ranking.
  - `stats` — paired Wilcoxon / paired-t across folds (anchor comparisons).

See docs/WACV_PIPELINE.md for the full protocol. Each function operates
on saved fold predictions (the `.npz` artifacts emitted by
`reproduction.py` under `wacv_results/`); none of them depend on TabPFN-3
being installed.
"""

from . import calibration, pathway_maps, stats  # noqa: F401

__all__ = ["calibration", "pathway_maps", "stats"]

"""TCGA-BRCA survival analysis — reproduces Figure/Table 3 of arXiv:2510.03455.

Pipeline:
  1. `data.TCGABRCALoader` — pairs WSIs with OS_time + event from clinical TSV
  2. `data.tile_wsi(...)` — tiles each WSI into 224×224 patches at level 0
  3. `data.embed_tiles(...)` — runs UNI v1 (or PEaRL image encoder) over tiles
  4. `ab_mil.ABMIL` — attention-based MIL pools tile embeddings → slide embedding
  5. `trainer.train_survival(...)` — Cox partial-likelihood loss, 5-fold CV
  6. `metrics.concordance_index_5fold(...)` — C-index with mean ± std

The 5-fold CV is patient-stratified (no patient appears in train + val of the
same fold) and stratified by event (so each fold has a similar event rate).
"""
from .data import TCGABRCALoader, tile_wsi, embed_tiles
from .ab_mil import ABMIL
from .trainer import train_survival, train_one_fold
from .metrics import concordance_index_5fold, fold_c_index

__all__ = [
    "TCGABRCALoader", "tile_wsi", "embed_tiles",
    "ABMIL",
    "train_survival", "train_one_fold",
    "concordance_index_5fold", "fold_c_index",
]

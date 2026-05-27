"""TCGA-BRCA WSI + clinical loader for survival analysis.

Three responsibilities:
  • `TCGABRCALoader` — discover WSIs on disk, pair each with OS_time/event from
    the clinical TSV downloaded via the GDC /cases endpoint.
  • `tile_wsi(...)` — extract 224×224 patches at level 0 from a .svs file,
    optionally with a simple tissue mask (Otsu on grayscale level-2 thumbnail).
  • `embed_tiles(...)` — run an image encoder (UNI v1 by default) over the
    tiles and return the (N_tiles, embed_dim) array.

These are I/O + numerics only — no training loop, no MIL aggregation. See
`ab_mil.py` and `trainer.py` for those.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TCGABRCALoader:
    """Pair TCGA-BRCA WSIs on disk with their OS time + event labels.

    Expected layout:
      wsi_dir/
        TCGA-XX-XXXX-...DX1.<uuid>.svs    # one or more per case
        ...
      clinical_tsv: GDC /cases TSV with columns
        submitter_id, demographic.vital_status,
        demographic.days_to_death, diagnoses.0.days_to_last_follow_up

    Case ID is parsed from filename: the first 12 chars (e.g. 'TCGA-OL-A5RW').
    """
    wsi_dir: str
    clinical_tsv: str

    cases: pd.DataFrame = field(init=False)
    n_cases: int = field(init=False, default=0)
    n_events: int = field(init=False, default=0)

    def __post_init__(self):
        self.cases = self._build_case_table()
        self.n_cases = len(self.cases)
        self.n_events = int(self.cases["event"].sum()) if self.n_cases else 0

    def _parse_case_id(self, filename: str) -> str:
        """TCGA filename → 12-char case ID (TCGA-XX-XXXX)."""
        return os.path.basename(filename)[:12]

    def _build_case_table(self) -> pd.DataFrame:
        """Cross-reference WSIs on disk with clinical OS labels."""
        if not Path(self.clinical_tsv).is_file():
            raise FileNotFoundError(
                f"clinical TSV not found at {self.clinical_tsv}. "
                "Download via the GDC /cases endpoint (see scripts/smoke_survival.py)."
            )
        clin = pd.read_csv(self.clinical_tsv, sep="\t", low_memory=False)
        # Build event + OS_time
        days_death = pd.to_numeric(clin["demographic.days_to_death"], errors="coerce")
        days_lfu = pd.to_numeric(clin["diagnoses.0.days_to_last_follow_up"], errors="coerce")
        clin["event"] = (clin["demographic.vital_status"] == "Dead").astype(int)
        clin["os_time"] = np.where(clin["event"] == 1, days_death, days_lfu)
        clin["case_id"] = clin["submitter_id"]
        clin = clin[(clin["os_time"].notna()) & (clin["os_time"] >= 0)][
            ["case_id", "event", "os_time"]
        ]

        wsi_dir = Path(self.wsi_dir)
        if not wsi_dir.is_dir():
            raise FileNotFoundError(
                f"wsi_dir {wsi_dir} does not exist. "
                "Download TCGA-BRCA via `gdc-client download -m gdc_manifest.txt`."
            )
        wsis = sorted(p for p in wsi_dir.glob("**/*.svs"))
        if not wsis:
            raise FileNotFoundError(
                f"no .svs files under {wsi_dir} — was the manifest empty?"
            )
        rows = []
        for p in wsis:
            cid = self._parse_case_id(p.name)
            row = clin[clin["case_id"] == cid]
            if len(row):
                rows.append({
                    "case_id": cid,
                    "wsi_path": str(p),
                    "event": int(row.iloc[0]["event"]),
                    "os_time": float(row.iloc[0]["os_time"]),
                })
        if not rows:
            raise RuntimeError(
                f"No WSIs matched a clinical row. WSI count: {len(wsis)}, "
                f"clinical rows: {len(clin)}. Likely a case_id parsing mismatch."
            )
        df = pd.DataFrame(rows)
        # One WSI per case (the DX1 diagnostic slide); deduplicate just in case
        df = df.drop_duplicates(subset=["case_id"]).reset_index(drop=True)
        return df

    def split_5fold(self, seed: int = 42, n_splits: int = 5):
        """Patient-stratified, event-stratified 5-fold CV.

        Returns a list of (train_idx, val_idx) tuples into self.cases.
        """
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.arange(self.n_cases), self.cases["event"].values))


def tile_wsi(svs_path: str, patch_size: int = 224, level: int = 0,
             tissue_thresh: int = 220, stride: Optional[int] = None,
             max_tiles: Optional[int] = None) -> np.ndarray:
    """Tile a WSI into (N, patch_size, patch_size, 3) uint8 patches.

    Tissue mask: very simple — drop patches whose mean grayscale > tissue_thresh
    (mostly white = background). For production use, swap in a more robust
    Otsu or HistoQC mask.

    Args:
        svs_path: path to .svs
        patch_size: tile edge in pixels (default 224 = PEaRL convention)
        level: pyramid level (0 = full resolution)
        tissue_thresh: drop tiles with mean grayscale > this (default 220)
        stride: pixels between tile origins (default = patch_size, no overlap)
        max_tiles: if set, randomly subsample to this many tiles after masking

    Returns: (N_tiles, patch_size, patch_size, 3) uint8 array.
    """
    try:
        import openslide
    except ImportError as e:
        raise ImportError(
            "openslide-python not installed. `pip install openslide-bin openslide-python`."
        ) from e
    slide = openslide.OpenSlide(svs_path)
    W, H = slide.level_dimensions[level]
    s = stride or patch_size
    coords = []
    tiles = []
    for y in range(0, H - patch_size, s):
        for x in range(0, W - patch_size, s):
            patch = np.array(
                slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            )
            # Quick tissue check
            if patch.mean() < tissue_thresh:
                tiles.append(patch)
                coords.append((x, y))
    slide.close()
    if not tiles:
        return np.zeros((0, patch_size, patch_size, 3), dtype=np.uint8)
    arr = np.stack(tiles, axis=0)
    if max_tiles is not None and len(arr) > max_tiles:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(arr), size=max_tiles, replace=False)
        arr = arr[idx]
    return arr


def embed_tiles(tiles: np.ndarray, encoder, device: str = "cuda",
                batch_size: int = 64) -> np.ndarray:
    """Run `encoder` over tiles → (N, embed_dim) numpy array.

    `encoder` should be a torch.nn.Module returning a (B, D) tensor for an
    input of shape (B, 3, H, W) (ImageNet-normalized). UNI v1 from
    `pearl_tabpfn.encoders.VisionEncoder` fits this.
    """
    import torch
    import torchvision.transforms as T

    if len(tiles) == 0:
        return np.zeros((0, getattr(encoder, "embed_dim", 256)), dtype=np.float32)

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    encoder = encoder.to(device).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = torch.stack([normalize(t) for t in tiles[i:i+batch_size]]).to(device)
            out = encoder(batch)
            # Accept either (B, D) features directly or a model that exposes
            # `.backbone_features(x)` (matches pearl_tabpfn.encoders.VisionEncoder).
            if hasattr(encoder, "backbone_features") and out.ndim != 2:
                out = encoder.backbone_features(batch)
            feats.append(out.detach().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)

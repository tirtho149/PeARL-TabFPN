"""PEaRL Data Loading & Preprocessing"""
import os
import json
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from scipy.stats import rankdata
import warnings
import anndata

warnings.filterwarnings("ignore")

try:
    import scanpy as sc
except ImportError:
    sc = None


def ssgsea(expr_matrix: np.ndarray, gene_names: List[str], pathways: Dict[str, List[str]]) -> np.ndarray:
    """
    Single-Sample Gene Set Enrichment Analysis (ssGSEA).

    Args:
        expr_matrix: (n_spots, n_genes) gene expression matrix
        gene_names: list of gene names
        pathways: dict mapping pathway_name -> list of genes in pathway

    Returns:
        (n_spots, n_pathways) enrichment scores
    """
    n_spots = expr_matrix.shape[0]
    n_pathways = len(pathways)
    scores = np.zeros((n_spots, n_pathways))

    gene_idx = {g: i for i, g in enumerate(gene_names)}

    for p_idx, (pathway_name, genes) in enumerate(pathways.items()):
        genes_in_dataset = [gene_idx[g] for g in genes if g in gene_idx]

        if len(genes_in_dataset) == 0:
            continue

        for spot_idx in range(n_spots):
            expr = expr_matrix[spot_idx]
            ranked = rankdata(expr)

            signal = np.sum(ranked[genes_in_dataset])
            noise = np.sum(ranked[~np.isin(np.arange(len(expr)), genes_in_dataset)])

            n_genes_in_pathway = len(genes_in_dataset)
            n_genes_not = len(expr) - n_genes_in_pathway

            if n_genes_not > 0:
                es = signal / n_genes_in_pathway - noise / n_genes_not
            else:
                es = signal / n_genes_in_pathway

            scores[spot_idx, p_idx] = es

    return scores


def load_hest_sample(
    hest_dir: str,
    sample_id: str,
    n_genes: int = 1000,
    n_pathways: int = 200,
    patch_size: int = 224,
    max_spots: int = 400,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one HEST-1k sample: patches, gene expression, pathway scores, spatial coords.

    Returns:
        patches: (n_spots, 3, H, W) float32 [0, 1]
        genes: (n_spots, n_genes) float32
        pathways: (n_spots, n_pathways) float32
        coords: (n_spots, 2) float32 normalized [0, 1]
    """
    rng = np.random.default_rng(seed)

    # Load AnnData with gene expression
    st_file = os.path.join(hest_dir, "st", f"{sample_id}.h5ad")
    adata = anndata.read_h5ad(st_file)

    # Load patches & barcodes
    patches_h5 = _find_patch_file(hest_dir, sample_id)
    img_array, bc_list = _read_patches_h5(patches_h5)

    # Match barcodes
    bc_map = _build_barcode_map(adata)
    pairs = _match_barcodes(bc_map, bc_list)

    if not pairs:
        raise RuntimeError(f"No barcodes matched for {sample_id}")

    if len(pairs) > max_spots:
        pick = np.sort(rng.choice(len(pairs), size=max_spots, replace=False))
        pairs = [pairs[j] for j in pick]

    # Slice data
    obs_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    img_idx = [p[1] for p in pairs]
    adata = adata[obs_idx].copy()
    img_array = np.stack([img_array[j] for j in img_idx], axis=0)

    # Normalize & log (numpy-based, no scanpy needed)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X = X / (X.sum(axis=1, keepdims=True) + 1e-10) * 1e4
    X = np.log1p(X)
    adata.X = X

    # HVG selection (numpy-based)
    X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    means = X_dense.mean(axis=0)
    vars = X_dense.var(axis=0)
    disp = vars / (means + 1e-6)
    top_genes = np.argsort(disp)[-n_genes:]
    adata = adata[:, top_genes].copy()

    # Get gene matrix
    X = adata.X
    genes = X.toarray().astype(np.float32) if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)
    gene_names = list(adata.var_names)

    # Pad/truncate genes to n_genes
    n_genes_actual = genes.shape[1]
    if genes.shape[1] < n_genes:
        pad = np.zeros((genes.shape[0], n_genes - genes.shape[1]), dtype=np.float32)
        genes = np.concatenate([genes, pad], axis=1)
    elif genes.shape[1] > n_genes:
        genes = genes[:, :n_genes].copy()
        n_genes_actual = n_genes

    # Pathway scoring (placeholder - would load from Reactome/MSigDB)
    pathways_dict = _load_pathways_from_reactome()  # Would fetch real pathways
    pathway_scores = ssgsea(genes[:, :n_genes_actual], gene_names[:n_genes_actual], pathways_dict)

    if pathway_scores.shape[1] < n_pathways:
        pad = np.zeros((pathway_scores.shape[0], n_pathways - pathway_scores.shape[1]), dtype=np.float32)
        pathway_scores = np.concatenate([pathway_scores, pad], axis=1)
    elif pathway_scores.shape[1] > n_pathways:
        pathway_scores = pathway_scores[:, :n_pathways].copy()

    pathway_scores = ((pathway_scores - pathway_scores.mean(0)) / (pathway_scores.std(0) + 1e-6)).astype(np.float32)

    # Process patches
    patches = np.stack(
        [_process_patch(img_array[i], patch_size) for i in range(len(img_array))],
        axis=0
    ).astype(np.float32)

    # Spatial coords
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    coords = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-6)

    return patches, genes, pathway_scores, coords


def _find_patch_file(hest_dir: str, sample_id: str) -> str:
    """Find patches HDF5 file for sample."""
    pdir = os.path.join(hest_dir, "patches")
    for name in (f"{sample_id}.h5", f"{sample_id}_patches.h5"):
        p = os.path.join(pdir, name)
        if os.path.isfile(p):
            return p
    import glob
    matches = sorted(glob.glob(os.path.join(pdir, f"*{sample_id}*.h5")))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No patches h5 for {sample_id}")


def _read_patches_h5(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load RGB patches and barcodes from h5."""
    with h5py.File(path, "r") as f:
        # Find image array
        imgs = None
        for k in ("img", "images", "patches", "data"):
            if k in f:
                imgs = np.asarray(f[k][:])
                break

        if imgs is None:
            raise RuntimeError(f"No image array in {path}")

        # Find barcodes
        bcs = None
        for k in ("barcodes", "barcode", "spot_id"):
            if k in f:
                bcs = _decode_h5_strings(f[k])
                break

        if bcs is None:
            raise RuntimeError(f"No barcodes in {path}")

    return imgs, bcs


def _decode_h5_strings(dset) -> List[str]:
    """Decode HDF5 string datasets."""
    raw = np.asarray(dset[:])

    # Handle (N, 1) shaped arrays by taking first column
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]

    out = []
    for x in raw:
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf-8", errors="replace").strip())
        elif isinstance(x, np.bytes_):
            out.append(bytes(x).decode("utf-8", errors="replace").strip())
        else:
            out.append(str(x).strip())
    return out


def _barcode_variants(s: str) -> set:
    """Generate barcode string variants."""
    s = str(s).strip()
    v = {s}
    if s.endswith("_1"):
        v.add(s[:-2] + "-1")
    if s.endswith("-1"):
        v.add(s[:-2] + "_1")
    v.add(s.replace("_", "-"))
    v.add(s.replace("-", "_"))
    return {x for x in v if x}


def _build_barcode_map(adata) -> Dict[str, int]:
    """Map barcode variants to AnnData row indices."""
    m = {}
    for i in range(len(adata)):
        candidates = [str(adata.obs_names[i])]
        for c in ("barcode", "Barcode", "spot_id", "id"):
            if c in adata.obs.columns:
                val = adata.obs.iloc[i][c]
                if val is not None and str(val) not in ("nan", "None"):
                    candidates.append(str(val))

        for c in candidates:
            for key in _barcode_variants(c):
                if key and key not in m:
                    m[key] = i
    return m


def _match_barcodes(bc_map: Dict[str, int], bc_list: List[str]) -> List[Tuple[int, int]]:
    """Match barcodes between AnnData and patches."""
    pairs = []
    seen = set()
    for hi, b in enumerate(bc_list):
        for key in _barcode_variants(b):
            if key in bc_map:
                idx = bc_map[key]
                if idx not in seen:
                    seen.add(idx)
                    pairs.append((idx, hi))
                break
    return pairs


def _process_patch(img: np.ndarray, size: int) -> np.ndarray:
    """Convert patch to CHW float32 [0, 1]."""
    # Handle HWC/CHW conversion
    if img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 4:
            img = img[..., :3]

    # Convert to uint8
    if img.dtype != np.uint8:
        if float(np.nanmax(img)) <= 1.0 + 1e-5:
            img = np.clip(img * 255.0, 0, 255)
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Ensure HWC uint8
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB, got {img.shape}")

    # Resize & normalize
    p = PILImage.fromarray(img).resize((size, size), PILImage.BILINEAR)
    arr = np.asarray(p, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return arr


def _load_pathways_from_reactome() -> Dict[str, List[str]]:
    """Load Reactome pathways. TODO: Fetch from actual source."""
    # Placeholder - would load real pathways from Reactome/MSigDB
    return {}


class HESTDataset(Dataset):
    """PyTorch Dataset for HEST samples."""

    def __init__(
        self,
        patches: np.ndarray,
        genes: np.ndarray,
        pathways: np.ndarray,
        coords: np.ndarray,
        sample_id: str = "sample",
    ):
        self.patches = torch.from_numpy(patches)
        self.genes = torch.from_numpy(genes)
        self.pathways = torch.from_numpy(pathways)
        self.coords = torch.from_numpy(coords)
        self.sample_id = sample_id

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return {
            "patch": self.patches[idx],
            "gene": self.genes[idx],
            "pathway": self.pathways[idx],
            "coord": self.coords[idx],
        }


def create_dataloader(
    patches: np.ndarray,
    genes: np.ndarray,
    pathways: np.ndarray,
    coords: np.ndarray,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create PyTorch DataLoader."""
    dataset = HESTDataset(patches, genes, pathways, coords)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

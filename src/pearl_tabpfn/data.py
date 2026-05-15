"""PEaRL Data Loading & Preprocessing"""
import os
import json
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Dict, List, Optional
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


def _select_hvg(adata, X: np.ndarray, n_top: int, method: str = "dispersion") -> np.ndarray:
    """Select top-`n_top` highly variable genes.

    Order of preference:
      1. `method="scanpy"` + scanpy installed → `sc.pp.highly_variable_genes(
         flavor="seurat")` (what the PEaRL paper uses).
      2. `method="seurat"` or scanpy missing → Seurat-flavor numpy fallback:
         bin genes by mean, z-score variance within each bin, pick top.
      3. `method="dispersion"` (legacy) → simple variance/mean ranking.

    Returns:
        ndarray of selected gene column indices (length min(n_top, n_genes)).
    """
    n_total = X.shape[1]
    if n_top >= n_total:
        return np.arange(n_total)

    if method == "scanpy" and sc is not None:
        # Scanpy expects an AnnData; we have one with X already log1p-norm'd.
        try:
            ad = adata.copy()
            ad.X = X
            sc.pp.highly_variable_genes(ad, n_top_genes=n_top, flavor="seurat")
            keep = np.where(ad.var["highly_variable"].values)[0]
            if len(keep) >= n_top:
                return keep[:n_top]
            print(f"  WARN: scanpy HVG returned {len(keep)} < {n_top}; padding.")
            extra = np.setdiff1d(np.arange(n_total), keep)
            return np.concatenate([keep, extra[: n_top - len(keep)]])
        except Exception as e:
            print(f"  scanpy HVG failed ({type(e).__name__}: {e}); using seurat numpy fallback.")
            method = "seurat"

    if method in ("scanpy", "seurat"):
        # Seurat-flavor: bin genes by mean of log-normalized expression,
        # z-score (variance/mean) within each bin, pick top by z.
        means = X.mean(axis=0)
        vars_ = X.var(axis=0)
        valid = means > 1e-12
        log_means = np.log10(means + 1e-12)
        log_disp = np.log10(vars_ / (means + 1e-12) + 1e-12)
        n_bins = 20
        lo, hi = log_means[valid].min(), log_means[valid].max()
        edges = np.linspace(lo - 1e-6, hi + 1e-6, n_bins + 1)
        bin_idx = np.clip(np.digitize(log_means, edges) - 1, 0, n_bins - 1)
        z = np.zeros(n_total, dtype=np.float32)
        for b in range(n_bins):
            m = (bin_idx == b) & valid
            if m.sum() < 3:
                continue
            mu = log_disp[m].mean()
            sd = log_disp[m].std() + 1e-12
            z[m] = (log_disp[m] - mu) / sd
        z[~valid] = -np.inf
        return np.argsort(z)[-n_top:]

    # Legacy: dispersion = var/mean
    means = X.mean(axis=0)
    vars_ = X.var(axis=0)
    disp = vars_ / (means + 1e-6)
    return np.argsort(disp)[-n_top:]


def apply_spatial_smoothing(
    expr: np.ndarray, coords: np.ndarray, k: int = 8
) -> np.ndarray:
    """K-nearest-neighbor (default 8) spatial smoothing of expression.

    Each spot's gene vector is replaced by the unweighted mean of itself and
    its k nearest spatial neighbors. Reduces spot-level dropout noise while
    preserving local tissue structure. The PEaRL paper applies this as part
    of preprocessing — apple-to-apple reproduction needs it.

    Args:
        expr:   (N, G) gene expression
        coords: (N, 2) spatial coordinates in any units (kNN is invariant to scale)
        k:      number of neighbors to average with (excludes self in the count)
    """
    from sklearn.neighbors import NearestNeighbors

    n = expr.shape[0]
    if n <= 1:
        return expr.copy()
    k_use = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_use + 1, algorithm="ball_tree").fit(coords)
    _, idx = nn.kneighbors(coords)  # (N, k+1); col 0 is self
    return expr[idx].mean(axis=1).astype(expr.dtype, copy=False)


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
    n_spots, n_genes = expr_matrix.shape
    n_pathways = len(pathways)
    scores = np.zeros((n_spots, n_pathways), dtype=np.float32)

    if n_pathways == 0 or n_genes == 0:
        return scores

    # Rank once per spot (was being redone per-pathway, an n_pathways× speedup
    # since ranks don't depend on the gene set).
    ranks = np.empty_like(expr_matrix, dtype=np.float32)
    for s in range(n_spots):
        ranks[s] = rankdata(expr_matrix[s]).astype(np.float32)

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    all_idx = np.arange(n_genes)

    for p_idx, (_, genes) in enumerate(pathways.items()):
        in_set = np.array([gene_idx[g] for g in genes if g in gene_idx], dtype=np.int64)
        n_in = len(in_set)
        if n_in == 0:
            continue
        n_out = n_genes - n_in
        if n_out == 0:
            scores[:, p_idx] = ranks[:, in_set].sum(axis=1) / n_in
            continue

        mask = np.ones(n_genes, dtype=bool)
        mask[in_set] = False
        out_idx = all_idx[mask]
        signals = ranks[:, in_set].sum(axis=1)
        noises = ranks[:, out_idx].sum(axis=1)
        scores[:, p_idx] = signals / n_in - noises / n_out

    return scores


def load_hest_sample(
    hest_dir: str,
    sample_id: str,
    n_genes: int = 1000,
    n_pathways: int = 200,
    patch_size: int = 224,
    max_spots: int = 400,
    seed: int = 42,
    normalization: str = "pearl_orig",
    pathway_dict: Dict[str, List[str]] = None,
    pathway_sources: str = "reactome",
    return_full_pathways: bool = False,
    smooth_genes: bool = False,
    smoothing_k: int = 8,
    min_spots_detected: int = 0,
    hvg_method: str = "dispersion",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one HEST-1k sample: patches, gene expression, pathway scores, spatial coords.

    normalization:
        'pearl_orig' — CP10K + log1p genes; pathway scores z-normalized per-column
                       (the original code's behavior; produces std=1 targets).
        'paper'      — CP10K + log1p genes, then per-gene min-max scaled to [0,1];
                       pathway scores left as raw ssGSEA (~0.05 std). This matches
                       the scale at which the published paper reports MSE/MAE.

    pathway_dict: if provided, score these pathways instead of loading Reactome.
                  Used by `load_hest_multi_sample` to keep pathway columns aligned
                  across samples.
    return_full_pathways: if True, return all scored pathways (no top-N selection).
                          Used by the multi-sample loader to do global selection.

    Returns:
        patches: (n_spots, 3, H, W) float32 [0, 1]
        genes:   (n_spots, n_genes) float32
        pathways: (n_spots, n_pathways) float32 (or full set if return_full_pathways)
        coords:  (n_spots, 2) float32 normalized [0, 1]
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

    # Paper-faithful gene filter: drop genes detected (count > 0) in fewer
    # than `min_spots_detected` spots BEFORE CP10K normalization. The PEaRL
    # paper says "filtered out genes detected in fewer than 1,000 spots".
    # Doing this pre-CP10K (rather than post-log1p) matches the Scanpy
    # convention `sc.pp.filter_genes(adata, min_cells=N)`.
    if min_spots_detected > 0 and X.shape[1] > 0:
        n_detected = (X > 0).sum(axis=0)
        keep = n_detected >= min_spots_detected
        if keep.sum() == 0:
            print(
                f"  WARN: min_spots_detected={min_spots_detected} kept 0 genes "
                f"on {sample_id}; relaxing to keep all genes."
            )
        else:
            n_drop = (~keep).sum()
            if n_drop > 0:
                X = X[:, keep]
                adata = adata[:, keep].copy()
                # print(f"  filter: dropped {n_drop}/{n_drop+keep.sum()} genes "
                #       f"(detected in <{min_spots_detected} spots)")

    X = X / (X.sum(axis=1, keepdims=True) + 1e-10) * 1e4
    X = np.log1p(X)

    # Optional 8-neighbor spatial smoothing on the (CP10K + log1p) matrix.
    # Paper applies this before HVG selection; doing it earlier here also lets
    # the variance-based HVG ranking reflect the smoothed signal.
    if smooth_genes:
        xy_for_smooth = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        X = apply_spatial_smoothing(X, xy_for_smooth, k=smoothing_k)
    adata.X = X

    # HVG selection — prefer scanpy when available (matches paper exactly),
    # fall back to a Seurat-flavor numpy implementation otherwise.
    top_genes = _select_hvg(adata, X, n_genes, method=hvg_method)
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

    # Pathway scoring on real pathway definitions.
    pathways_dict = (
        pathway_dict
        if pathway_dict is not None
        else _load_pathways(sources=pathway_sources)
    )
    pathway_scores = ssgsea(genes[:, :n_genes_actual], gene_names[:n_genes_actual], pathways_dict)

    if normalization == "paper":
        # Per-gene min-max [0,1] for genes. Paper's exact gene normalization is
        # undocumented but its tiny MSE values imply a [0,1]-ish range.
        gmin = genes.min(axis=0, keepdims=True)
        gmax = genes.max(axis=0, keepdims=True)
        genes = ((genes - gmin) / (gmax - gmin + 1e-6)).astype(np.float32)
    elif normalization == "paper_log1p_only":
        # Skip per-gene normalization; targets stay in log1p space (~[0, 8]).
        # High-variance genes dominate the global flatten PCC, which usually
        # raises it because high-var genes are also the most predictable.
        pass
    elif normalization == "paper_zscore":
        # Per-gene z-score: mean 0, std 1. PCC is invariant per dim, but the
        # global flatten PCC can shift because all genes contribute equally
        # in std-units (vs unequally in raw log1p).
        gmean = genes.mean(axis=0, keepdims=True)
        gstd = genes.std(axis=0, keepdims=True)
        genes = ((genes - gmean) / (gstd + 1e-6)).astype(np.float32)

    if return_full_pathways:
        # Multi-sample loader needs raw scores so it can do global variance-
        # based pathway selection. It will normalize after selection.
        pathway_scores = pathway_scores.astype(np.float32)
    else:
        # Single-sample path: pick top-n_pathways by variance, pad if smaller,
        # z-normalize. Pathways with no gene-set overlap produce constant-zero
        # columns; the top-N filter drops those.
        col_std = pathway_scores.std(axis=0)
        if pathway_scores.shape[1] > n_pathways:
            keep_idx = np.argsort(col_std)[-n_pathways:][::-1]
            pathway_scores = pathway_scores[:, keep_idx]
        elif pathway_scores.shape[1] < n_pathways:
            pad = np.zeros((pathway_scores.shape[0], n_pathways - pathway_scores.shape[1]), dtype=np.float32)
            pathway_scores = np.concatenate([pathway_scores, pad], axis=1)
        pathway_scores = (
            (pathway_scores - pathway_scores.mean(0)) / (pathway_scores.std(0) + 1e-6)
        ).astype(np.float32)

    # Process patches
    patches = np.stack(
        [_process_patch(img_array[i], patch_size) for i in range(len(img_array))],
        axis=0
    ).astype(np.float32)

    # Spatial coords
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    coords = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-6)

    return patches, genes, pathway_scores, coords


def load_hest_multi_sample(
    hest_dir: str,
    sample_ids: List[str],
    n_genes: int = 1000,
    n_pathways: int = 200,
    patch_size: int = 224,
    max_spots_per_section: int = 400,
    normalization: str = "paper",
    seed: int = 42,
    verbose: bool = True,
    pathway_sources: str = "reactome",
    pathway_normalization: str = "zscore",
    smooth_genes: bool = False,
    smoothing_k: int = 8,
    min_spots_detected: int = 0,
    hvg_method: str = "dispersion",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and concatenate multiple HEST-1k sections.

    Pathway columns are aligned across sections by scoring against a shared
    Reactome dict and selecting the top-`n_pathways` columns by *pooled*
    variance after concatenation. Without this, top-N-by-variance picks
    different pathways per section, breaking cross-section training.

    Returns:
        patches:     (total_spots, 3, H, W) float32
        genes:       (total_spots, n_genes) float32
        pathways:    (total_spots, n_pathways) float32
        coords:      (total_spots, 2) float32 (per-section [0,1] normalized)
        section_ids: (total_spots,) int64 — same index for spots from same section
    """
    pathway_dict = _load_pathways(sources=pathway_sources)
    if verbose:
        print(f"Pathways loaded ({pathway_sources}): {len(pathway_dict)}")

    parts = []
    for i, sid in enumerate(sample_ids):
        try:
            p, g, pw_full, c = load_hest_sample(
                hest_dir=hest_dir,
                sample_id=sid,
                n_genes=n_genes,
                n_pathways=n_pathways,
                patch_size=patch_size,
                max_spots=max_spots_per_section,
                seed=seed + i,
                normalization=normalization,
                pathway_dict=pathway_dict,
                return_full_pathways=True,
                smooth_genes=smooth_genes,
                smoothing_k=smoothing_k,
                min_spots_detected=min_spots_detected,
                hvg_method=hvg_method,
            )
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{len(sample_ids)}] {sid}: SKIP ({type(e).__name__}: {e})")
            continue
        parts.append((p, g, pw_full, c, i))
        if verbose:
            print(f"  [{i+1}/{len(sample_ids)}] {sid}: {p.shape[0]} spots")

    if not parts:
        raise RuntimeError("No samples loaded successfully")

    patches = np.concatenate([t[0] for t in parts], axis=0)
    genes = np.concatenate([t[1] for t in parts], axis=0)
    pw_full = np.concatenate([t[2] for t in parts], axis=0)
    coords = np.concatenate([t[3] for t in parts], axis=0)
    section_ids = np.concatenate(
        [np.full(t[0].shape[0], t[4], dtype=np.int64) for t in parts]
    )

    # Global top-N pathway selection by pooled variance. The PEaRL paper's
    # reported pathway MSE (~0.0017 on Breast) implies raw-ssGSEA scaling
    # (std ≈ 0.05). z-normalization (std=1) inflates MSE by ~400× — PCC is
    # scale-invariant per-dim but not in the global flatten metric. Use 'raw'
    # for apple-to-apple parity with the paper.
    col_std = pw_full.std(axis=0)
    keep = np.argsort(col_std)[-n_pathways:][::-1]
    pathways = pw_full[:, keep]
    if pathway_normalization == "zscore":
        pathways = (
            (pathways - pathways.mean(0)) / (pathways.std(0) + 1e-6)
        ).astype(np.float32)
    elif pathway_normalization == "raw":
        pathways = pathways.astype(np.float32)
    else:
        raise ValueError(
            f"pathway_normalization must be 'raw' or 'zscore', got {pathway_normalization!r}"
        )

    if verbose:
        print(
            f"Pooled: {patches.shape[0]} spots, {len(parts)} sections, "
            f"genes {genes.shape}, pathways {pathways.shape} "
            f"(picked top-{n_pathways} of {pw_full.shape[1]}, scale={pathway_normalization}, "
            f"smoothing={'on' if smooth_genes else 'off'})"
        )

    return patches, genes, pathways, coords, section_ids


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


# UNI / standard ViT input statistics. Without this normalization, frozen
# pretrained features degrade significantly — the model sees raw [0,1] tensors
# instead of mean/std-shifted ones.
_IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _process_patch(img: np.ndarray, size: int) -> np.ndarray:
    """Convert patch to CHW float32, ImageNet-normalized."""
    if img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 4:
            img = img[..., :3]

    if img.dtype != np.uint8:
        if float(np.nanmax(img)) <= 1.0 + 1e-5:
            img = np.clip(img * 255.0, 0, 255)
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB, got {img.shape}")

    p = PILImage.fromarray(img).resize((size, size), PILImage.BILINEAR)
    arr = np.asarray(p, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = (arr - _IMNET_MEAN) / _IMNET_STD
    return arr


_REACTOME_GMT_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
# MSigDB Hallmark gene sets (50 well-curated cancer pathways). The PEaRL paper
# uses Reactome + MSigDB; without Hallmark, ~50 high-signal cancer pathways are
# missing from the pool that ssGSEA + variance ranking can pick from.
_MSIGDB_HALLMARK_URLS = (
    # Primary: GitHub-hosted GMT (no auth required).
    "https://raw.githubusercontent.com/igordot/msigdb/main/data/h.all.v2023.1.Hs.symbols.gmt",
    # Mirror.
    "https://raw.githubusercontent.com/RasmussenLab/msigdb-mirror/master/h.all.v7.5.1.symbols.gmt",
)
_PATHWAY_CACHE_DIR = os.environ.get("PEARL_PATHWAY_CACHE", "./pathway_data")


def _load_pathways_from_reactome(
    cache_dir: str = None,
    min_genes: int = 5,
    max_genes: int = 500,
) -> Dict[str, List[str]]:
    """
    Load Reactome canonical pathways as {pathway_name: [HGNC_gene_symbol, ...]}.

    Downloads the GMT once on first call (~5 MB) into `cache_dir` and reuses it.
    Filters out pathways with too few or too many genes, since they're either
    too noisy (small sets) or too generic (e.g., "Metabolism", >500 genes) for
    ssGSEA to be informative.
    """
    import zipfile
    import urllib.request

    cache_dir = cache_dir or _PATHWAY_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    gmt_path = os.path.join(cache_dir, "ReactomePathways.gmt")

    if not os.path.isfile(gmt_path):
        zip_path = os.path.join(cache_dir, "ReactomePathways.gmt.zip")
        print(f"Downloading Reactome pathways from {_REACTOME_GMT_URL} ...")
        urllib.request.urlretrieve(_REACTOME_GMT_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                if member.endswith("ReactomePathways.gmt"):
                    with z.open(member) as src, open(gmt_path, "wb") as dst:
                        dst.write(src.read())
                    break
            else:
                z.extractall(cache_dir)
        os.remove(zip_path)

    pathways: Dict[str, List[str]] = {}
    with open(gmt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            # Reactome's GMT bundles all species in one file, distinguished by
            # the second column (description) being suffixed with the species
            # name. Filter to Homo sapiens, since HEST gene symbols are human.
            description = parts[1] if len(parts) > 1 else ""
            if description and "Homo sapiens" not in description:
                continue
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if min_genes <= len(genes) <= max_genes:
                pathways[name] = genes

    if not pathways:
        # Fall back to no species filter if the description-based filter
        # excluded everything (different Reactome versions vary in formatting).
        with open(gmt_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                name = parts[0]
                genes = [g.strip() for g in parts[2:] if g.strip()]
                if min_genes <= len(genes) <= max_genes:
                    pathways[name] = genes

    return pathways


def _load_pathways_msigdb_hallmark(
    cache_dir: str = None,
    min_genes: int = 5,
    max_genes: int = 500,
) -> Dict[str, List[str]]:
    """Load MSigDB Hallmark gene sets (50 cancer-relevant pathways).

    Tries each URL in `_MSIGDB_HALLMARK_URLS` in order; on persistent failure
    returns an empty dict and prints a warning, so callers can fall back to
    Reactome-only. The Hallmark collection is small (~50 sets) but very
    high-signal for oncology spatial transcriptomics — paper uses it.
    """
    import urllib.request

    cache_dir = cache_dir or _PATHWAY_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    gmt_path = os.path.join(cache_dir, "msigdb_hallmark.gmt")

    if not os.path.isfile(gmt_path):
        last_err = None
        for url in _MSIGDB_HALLMARK_URLS:
            try:
                print(f"Downloading MSigDB Hallmark from {url} ...")
                urllib.request.urlretrieve(url, gmt_path)
                break
            except Exception as e:
                last_err = e
                continue
        else:
            print(
                f"WARNING: failed to download MSigDB Hallmark "
                f"({type(last_err).__name__}: {last_err}); using Reactome only."
            )
            return {}

    pathways: Dict[str, List[str]] = {}
    with open(gmt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if min_genes <= len(genes) <= max_genes:
                pathways[name] = genes
    return pathways


def _load_pathways(
    sources: str = "reactome",
    cache_dir: str = None,
) -> Dict[str, List[str]]:
    """Load pathway gene sets from the configured sources.

    sources='reactome'           — Reactome only (legacy default).
    sources='reactome_msigdb'    — Reactome + MSigDB Hallmark (paper-faithful).
                                   MSigDB names are prefixed with 'HALLMARK_' so
                                   they don't collide with Reactome names.
    """
    pathways = _load_pathways_from_reactome(cache_dir=cache_dir)
    if sources == "reactome_msigdb":
        hall = _load_pathways_msigdb_hallmark(cache_dir=cache_dir)
        for name, genes in hall.items():
            key = name if name.upper().startswith("HALLMARK") else f"HALLMARK_{name}"
            pathways[key] = genes
        print(f"  Reactome + MSigDB Hallmark combined: {len(pathways)} pathways")
    return pathways


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

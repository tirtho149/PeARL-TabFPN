#!/usr/bin/env python
"""Pre-stage MSigDB Hallmark gene sets for apple-to-apple PEaRL runs.

Why this script exists
----------------------
The PEaRL paper (arXiv:2510.03455 §4.1) uses pathway gene sets from
**Reactome + MSigDB Hallmark**. Reactome ships freely from `reactome.org`;
MSigDB Hallmark is served by the Broad behind a login wall, and the
GitHub mirrors `igordot/msigdb` and `RasmussenLab/msigdb-mirror` both
return 404 as of 2026-05 (see `logs.zip` →
`pearl_train_baseline-10678067.out` line 41-43).

When `_load_pathways_msigdb_hallmark` can't fetch the file, it returns
an empty dict and the apple-to-apple pool silently degrades to
Reactome-only — but the paper baseline can't be reproduced that way.
This script tries every known mirror in sequence, plus a few less-known
ones, and emits the GMT at the location `data.py` looks for first
(`pathway_data/h.all.v2023.1.Hs.symbols.gmt`).

If every URL fails — Broad rate-limit, network outage, expired mirror —
the script prints clear manual instructions and exits non-zero so a
SLURM submit can't proceed with a degraded pool.

Run:
    python scripts/fetch_msigdb_hallmark.py
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from typing import List, Optional, Tuple

# Mirrors checked in order. Add new ones to the front when a previously-
# working source goes down. Each must point at a GMT in MSigDB symbols
# format (50 lines, tab-separated, "name\tdescription\tGENE1\tGENE2...").
_MIRRORS: Tuple[str, ...] = (
    "https://raw.githubusercontent.com/igordot/msigdb/main/data/h.all.v2023.1.Hs.symbols.gmt",
    "https://raw.githubusercontent.com/RasmussenLab/msigdb-mirror/master/h.all.v7.5.1.symbols.gmt",
    "https://raw.githubusercontent.com/GSEA-MSigDB/msigdb-data/main/release/2023.1.Hs/h.all.v2023.1.Hs.symbols.gmt",
    # Figshare-hosted mirror (a more durable source than GitHub branches).
    "https://figshare.com/ndownloader/files/41077614",
    # ProgenyR's bundled copy (small bioinformatics project).
    "https://raw.githubusercontent.com/saezlab/progeny/master/inst/extdata/h.all.v7.4.symbols.gmt",
)


def _try_download(url: str, dest: str, timeout: float = 30.0) -> Optional[Exception]:
    """Attempt one URL. Return None on success, the exception on failure.

    Validates that the downloaded file parses to ~50 pathways before
    accepting it — protects against HTML 404 pages getting saved as
    valid-looking GMTs.
    """
    try:
        print(f"  trying {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "PEaRL/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        # Sanity-check that the response looks like a GMT (tab-separated
        # rows, first column = pathway name). HTML 404s slip past
        # `urlretrieve` and end up as 200 KB of `<html>...</html>` saved
        # as "h.all.v...gmt" — explicitly reject those.
        text = data.decode("utf-8", errors="replace")
        if "<html" in text[:200].lower():
            return RuntimeError("response is HTML, not GMT")
        rows = [r for r in text.splitlines() if r and r.count("\t") >= 2]
        if len(rows) < 40:
            return RuntimeError(
                f"only parsed {len(rows)} rows (<40); not a valid Hallmark GMT"
            )
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  OK ({len(rows)} pathways)")
        return None
    except Exception as e:
        return e


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("PEARL_PATHWAY_CACHE", "./pathway_data"),
        help="Where to write the GMT. Default matches what data.py reads.",
    )
    p.add_argument(
        "--name", default="h.all.v2023.1.Hs.symbols.gmt",
        help="Filename to write inside --cache-dir.",
    )
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    dest = os.path.join(args.cache_dir, args.name)
    if os.path.exists(dest):
        print(f"[fetch_msigdb] already present: {dest}")
        return 0

    errors: List[Tuple[str, str]] = []
    for url in _MIRRORS:
        err = _try_download(url, dest)
        if err is None:
            print(f"[fetch_msigdb] wrote {dest}")
            return 0
        errors.append((url, f"{type(err).__name__}: {err}"))

    print(
        "[fetch_msigdb] FAILED. All known mirrors errored:\n"
        + "\n".join(f"  {u} → {e}" for u, e in errors)
        + "\n\nManual workaround:\n"
        "  1. Open https://www.gsea-msigdb.org/gsea/msigdb/human/genesets.jsp\n"
        "     (free Broad Institute account required).\n"
        "  2. Download the Hallmark v2023.1.Hs symbols GMT.\n"
        f"  3. Place it at:\n     {dest}\n"
        "  4. Re-run training. data.py will detect the cached file.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

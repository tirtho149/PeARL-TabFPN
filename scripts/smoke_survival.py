#!/usr/bin/env python3
"""TCGA-BRCA survival pipeline smoke test (Figure/Table 3 of the PEaRL paper).

Verifies that every layer of the survival pipeline is reachable on this
machine, WITHOUT needing the full 1.08 TB TCGA-BRCA WSI download:

  [1] GDC REST API is reachable and reports the expected file inventory
      (1,133 TCGA-BRCA diagnostic slides as of Oct 2025).
  [2] Clinical / OS-time loading works: pulls all 1,098 cases via the
      /cases endpoint, builds the (OS_time, event) arrays Cox loss
      needs, reports censoring stats.
  [3] C-index implementation is mathematically correct: random risk →
      0.5, perfect oracle → 1.0, anti-oracle → 0.0, moderate predictor
      in the realistic 0.6-0.9 band.
  [4] Real .svs WSI loading works via openslide: opens the smallest
      TCGA-BRCA slide (~24 MB), reads its level pyramid, serves
      224×224 patches at the magnification PEaRL uses.

What this does NOT do:
  - Train an ABMIL model (no PyTorch survival code in the repo yet).
  - Download all 1,133 slides (~1.08 TB; impossible on a laptop).
  - Produce a real C-index against PEaRL embeddings.

Prerequisites:
  - venv with: numpy, pandas, scipy, scikit-learn, lifelines, openslide-bin,
    openslide-python, tifffile, requests
  - Network access to api.gdc.cancer.gov (no auth needed — WSI metadata
    + clinical TSVs + open-access diagnostic slides are all unauthenticated).
  - ~50 MB free disk for the one-slide WSI smoke download.

Usage:
    python scripts/smoke_survival.py            # full smoke
    python scripts/smoke_survival.py --no-wsi   # skip the 24 MB WSI download
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


GDC_API = "https://api.gdc.cancer.gov"
DATA_DIR = Path(os.environ.get("PEARL_SMOKE_DIR", "./tcga_smoke")).resolve()
WSI_FILE = DATA_DIR / "smallest_brca.svs"
CLINICAL_FILE = DATA_DIR / "brca_survival.tsv"


GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def section(title: str):
    print(f"\n{YELLOW}{title}{RESET}")


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    suffix = f" {DIM}— {detail}{RESET}" if detail else ""
    print(f"    {mark} {name}{suffix}")
    return ok


def gdc_brca_slide_filters() -> dict:
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
        ],
    }


def smoke_gdc_inventory() -> list[bool]:
    section("[1/4] GDC API — TCGA-BRCA inventory")
    results = []
    r = requests.post(
        f"{GDC_API}/files",
        json={"filters": gdc_brca_slide_filters(), "size": 0},
        timeout=30,
    )
    results.append(check("GDC API reachable", r.status_code == 200, f"HTTP {r.status_code}"))
    if r.status_code != 200:
        return results
    total = r.json()["data"]["pagination"]["total"]
    results.append(check(
        f"TCGA-BRCA has the expected ~1,133 diagnostic slides",
        1100 <= total <= 1200,
        f"found {total}",
    ))
    return results


def smoke_clinical() -> tuple[list[bool], dict | None]:
    section("[2/4] Clinical / OS-time loading")
    results = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = requests.get(
        f"{GDC_API}/cases",
        params={
            "filters": '{"op":"in","content":{"field":"project.project_id","value":["TCGA-BRCA"]}}',
            "fields": "submitter_id,demographic.vital_status,demographic.days_to_death,"
                      "diagnoses.days_to_last_follow_up",
            "format": "TSV",
            "size": 2000,
        },
        timeout=60,
    )
    elapsed = time.time() - t0
    results.append(check("/cases endpoint reachable", r.status_code == 200, f"HTTP {r.status_code}"))
    if r.status_code != 200:
        return results, None
    CLINICAL_FILE.write_bytes(r.content)
    df = pd.read_csv(CLINICAL_FILE, sep="\t")
    results.append(check(
        "1098 TCGA-BRCA cases retrieved",
        1050 <= len(df) <= 1150,
        f"{len(df)} cases in {elapsed*1000:.0f}ms",
    ))

    days_death = pd.to_numeric(df["demographic.days_to_death"], errors="coerce")
    days_lfu = pd.to_numeric(df["diagnoses.0.days_to_last_follow_up"], errors="coerce")
    event = (df["demographic.vital_status"] == "Dead").astype(int).values
    os_time = np.where(event == 1, days_death, days_lfu)
    valid = ~pd.isna(os_time) & (os_time >= 0)
    os_time = os_time[valid].astype(float)
    event = event[valid].astype(int)
    n_events = int(event.sum())
    n_censored = int((1 - event).sum())

    results.append(check(
        f"OS_time + event arrays buildable for Cox loss",
        len(os_time) > 1000 and 100 < n_events < 300,
        f"{len(os_time)} usable cases ({n_events} events, {n_censored} censored)",
    ))
    return results, {"os_time": os_time, "event": event}


def concordance_via_lifelines(os_time, risk, event) -> float:
    from lifelines.utils import concordance_index
    return float(concordance_index(os_time, -risk, event))


def smoke_cindex(survival: dict | None) -> list[bool]:
    section("[3/4] C-index math (against real TCGA-BRCA outcomes)")
    results = []
    if survival is None:
        results.append(check("skipped — survival data unavailable", False))
        return results
    os_time, event = survival["os_time"], survival["event"]
    rng = np.random.default_rng(42)

    # Random risk over many trials → mean ~0.5
    cs = [concordance_via_lifelines(os_time, rng.standard_normal(len(os_time)), event) for _ in range(20)]
    mean, std = float(np.mean(cs)), float(np.std(cs))
    results.append(check(
        "random risk → C ≈ 0.5",
        0.45 < mean < 0.55,
        f"C = {mean:.3f} ± {std:.3f} over 20 trials",
    ))

    # Perfect oracle: risk = -OS_time (concordant on every comparable pair)
    c = concordance_via_lifelines(os_time, -os_time, event)
    results.append(check(
        "perfect oracle (-OS_time) → C ≈ 1.0",
        c > 0.99,
        f"C = {c:.3f}",
    ))

    # Anti-oracle: risk = +OS_time → C ≈ 0
    c = concordance_via_lifelines(os_time, os_time, event)
    results.append(check(
        "anti-oracle (+OS_time) → C ≈ 0.0",
        c < 0.02,
        f"C = {c:.3f}",
    ))

    # Realistic noisy predictor — should land in plausible 0.6-0.9 band
    noisy = -os_time + os_time.std() * rng.standard_normal(len(os_time))
    c = concordance_via_lifelines(os_time, noisy, event)
    results.append(check(
        "moderate noisy predictor → C in 0.6-0.9",
        0.6 < c < 0.9,
        f"C = {c:.3f}",
    ))

    # 5-fold CV reporting pipeline (matches paper's mean ± std format)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_c = []
    for _, val in kf.split(os_time):
        val_risk = -os_time[val] + 1.5 * os_time.std() * rng.standard_normal(len(val))
        fold_c.append(concordance_via_lifelines(os_time[val], val_risk, event[val]))
    results.append(check(
        "5-fold CV reporting works end-to-end",
        len(fold_c) == 5 and all(0 < c < 1 for c in fold_c),
        f"folds: {[f'{c:.3f}' for c in fold_c]} → {np.mean(fold_c):.3f} ± {np.std(fold_c):.3f} "
        f"(paper PEaRL: 0.659 ± 0.027)",
    ))
    return results


def smoke_wsi(skip: bool) -> list[bool]:
    section("[4/4] WSI loading (real TCGA-BRCA slide via openslide)")
    if skip:
        print(f"    {YELLOW}skipped (--no-wsi){RESET}")
        return []
    results = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Look up the smallest BRCA diagnostic slide via GDC
    r = requests.post(
        f"{GDC_API}/files",
        json={
            "filters": gdc_brca_slide_filters(),
            "fields": "file_id,file_name,file_size,cases.submitter_id",
            "size": 1,
            "sort": "file_size:asc",
        },
        timeout=30,
    )
    results.append(check("GDC file metadata lookup", r.status_code == 200))
    if r.status_code != 200:
        return results
    hit = r.json()["data"]["hits"][0]
    file_id, file_name, file_size = hit["file_id"], hit["file_name"], hit["file_size"]
    case_id = hit.get("cases", [{}])[0].get("submitter_id", "?")
    print(f"    {DIM}smallest slide: {case_id} ({file_size//(1024*1024)} MB){RESET}")

    if not WSI_FILE.exists() or WSI_FILE.stat().st_size != file_size:
        print(f"    {DIM}downloading {file_size//(1024*1024)} MB ...{RESET}")
        t0 = time.time()
        with requests.get(f"{GDC_API}/data/{file_id}", stream=True, timeout=600) as rr:
            with WSI_FILE.open("wb") as f:
                for chunk in rr.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"    {DIM}downloaded in {time.time()-t0:.1f}s{RESET}")
    else:
        print(f"    {DIM}using cached {WSI_FILE.name}{RESET}")

    try:
        import openslide
    except ImportError:
        results.append(check("openslide importable", False, "install with `pip install openslide-bin openslide-python`"))
        return results

    slide = openslide.OpenSlide(str(WSI_FILE))
    results.append(check(
        ".svs opens via openslide",
        slide.level_count > 0,
        f"levels={slide.level_count}, dims={slide.level_dimensions[0]}, vendor={slide.properties.get('openslide.vendor', '?')}",
    ))

    # Read a 224x224 patch at level 0 (the PEaRL paper's patch size)
    cx, cy = slide.dimensions[0] // 2, slide.dimensions[1] // 2
    t0 = time.time()
    patch = np.array(slide.read_region((cx, cy), 0, (224, 224)).convert("RGB"))
    elapsed = (time.time() - t0) * 1000
    results.append(check(
        "224×224 patch served at level 0",
        patch.shape == (224, 224, 3) and patch.dtype == np.uint8,
        f"shape={patch.shape}, {elapsed:.0f}ms, pixel mean={patch.mean():.0f}",
    ))

    # Test every pyramid level reads
    n_levels_ok = 0
    for lvl in range(slide.level_count):
        try:
            slide.read_region((0, 0), lvl, (256, 256)).convert("RGB")
            n_levels_ok += 1
        except Exception:
            pass
    results.append(check(
        "all pyramid levels readable",
        n_levels_ok == slide.level_count,
        f"{n_levels_ok}/{slide.level_count} levels",
    ))
    slide.close()
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-wsi", action="store_true", help="Skip the ~24 MB WSI download.")
    args = ap.parse_args()

    print("=" * 70)
    print("TCGA-BRCA SURVIVAL PIPELINE SMOKE TEST")
    print("=" * 70)
    print(f"Data dir: {DATA_DIR}")

    all_results = []
    all_results.extend(smoke_gdc_inventory())
    clinical_results, survival = smoke_clinical()
    all_results.extend(clinical_results)
    all_results.extend(smoke_cindex(survival))
    all_results.extend(smoke_wsi(skip=args.no_wsi))

    passed = sum(all_results)
    total = len(all_results)
    print("\n" + "=" * 70)
    if passed == total:
        print(f"{GREEN}SMOKE PASSED — {passed}/{total} assertions{RESET}")
        print("  Every layer of the survival pipeline works on this machine.")
        print("  Missing for a real C-index reproduction: WSI bulk download (~1.08 TB)")
        print("  + ABMIL training code (see docs WACV refactor plan).")
    else:
        print(f"{RED}SMOKE FAILED — {passed}/{total} assertions{RESET}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

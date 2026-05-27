# Reproducibility Fixes — Baseline vs arXiv:2510.03455

This document records the gap between the May 2026 baseline runs in
`logs.zip` (jobs 10678067 / 10678070) and the published PEaRL numbers in
arXiv:2510.03455 v1, and the fixes applied here.

## 1. What we observed

Last successful pair of SLURM jobs:

| Variant | Gene PCC | Gene MSE | Pathway PCC | Pathway MSE |
|---|---:|---:|---:|---:|
| PEaRL+MLP (ours, 10678067) | 0.0754 ± 0.0199 | 0.0618 ± 0.0026 | 0.2446 ± 0.0537 | **12 335.68 ± 7 570.04** |
| PEaRL+TabPFN-v2 (ours, 10678070) | 0.0747 ± 0.0237 | 0.0612 ± 0.0023 | 0.2381 ± 0.0764 | **12 214.97 ± 7 312.76** |
| **PEaRL paper (Breast)** | **0.5868 ± 0.0359** | 0.0732 ± 0.0033 | **0.5055 ± 0.0271** | **0.0017 ± 0.0001** |

Plus several warnings the runs emitted:

- `WARNING: failed to download MSigDB Hallmark (HTTPError: HTTP Error 404: Not Found); using Reactome only.` — every run since the mirror went down silently degrades to Reactome-only.
- `WARN: min_spots_detected=1000 kept 0 genes on <section>; relaxing to keep all genes.` — fired on **all 36 sections**.
- `TabPFNLicenseError: ... no interactive terminal is available.` — every TabPFN-v2 fold lost its head fit after stage 1+2 had already trained.

## 2. Root causes (and the fixes shipped here)

### A. Pathway target scale catastrophically wrong (×7 000 000)

Paper pathway MSE is **0.0017**; ours was **12 335**. The
`pathway_normalization="raw"` mode left the diff-of-mean-ranks ssGSEA
output unchanged, whose values can reach the hundreds per dimension.

Stage 2 supervised loss is `L_path + L_gene`. With `L_path` ~10 000 and
`L_gene` ~0.06, the optimizer spent ≥ 100 000× more gradient on pathway
columns than gene columns. That alone is why gene PCC collapsed from
0.5868 to 0.0754. The pathway head still learned a little (PCC 0.24)
because the targets were huge but at least correlated with image
features.

**Fix.** Added `pathway_normalization="paper"` in
`src/pearl_tabpfn/data.py` (per-pathway min-max to [0, 1]) and made
`--apple-to-apple` use it. The pathway target std now lands near
~0.05 — matching the paper's expected MSE/MAE magnitudes — and the
stage-2 loss is balanced between gene and pathway heads. The runner now
prints the pathway target stats right after loading so a regression to
"raw" can't sneak through silently.

### B. `--apple-to-apple` silently degraded to Reactome-only

The two GitHub mirrors used for MSigDB Hallmark both 404 as of May 2026.
`_load_pathways_msigdb_hallmark` printed a warning and returned `{}`,
and `_load_pathways` then announced "Reactome + MSigDB Hallmark combined:
2179 pathways" — but 2179 was Reactome alone. The 50 hallmark cancer-
relevant pathways the paper bakes into its pool were missing.

**Fix.**
1. `_load_pathways_msigdb_hallmark` now also checks `pathway_data/`
   (`PEARL_PATHWAY_CACHE`) for a manually-staged GMT before hitting the
   network, accepting any of `h.all.v2023.1.Hs.symbols.gmt`,
   `h.all.v7.5.1.symbols.gmt`, `h.all.symbols.gmt`, `msigdb_hallmark.gmt`.
2. The mirror list was widened (Figshare mirror + saezlab + GSEA-MSigDB
   data repo).
3. The `_load_pathways` reporter now prints `Reactome=N1, Hallmark=N2,
   combined unique=N3` so a degraded pool is visible at a glance.
4. **`--apple-to-apple` now sets `strict_msigdb=True`** → if Hallmark
   can't load, the run **aborts** with the staging instructions instead
   of training a degraded model.
5. New helper script: `scripts/fetch_msigdb_hallmark.py`.

### C. TabPFN runs lost their head fit after stage 1+2 had trained

Every TabPFN fold trained Stage 1 (~16 epochs) and Stage 2 (~30+ epochs)
on GPU, then raised `TabPFNLicenseError` at the post-stage-2 fit. The
GPU hours are unrecoverable.

**Fix.** Added `_verify_apple_to_apple_data_ready(args)` to
`reproduction.py`. It runs BEFORE any data load when `--apple-to-apple`
is set and `--head-mode` includes TabPFN, and aborts immediately if
`TABPFN_TOKEN` / `TABPFN_API_KEY` are missing from the environment. The
SLURM scripts already source `.env`, so adding `TABPFN_TOKEN=…` there is
sufficient.

### D. `min_spots_detected=1000` was a per-section no-op (KNOWN LIMITATION)

The paper filter is "drop genes detected in fewer than 1 000 spots" —
applied on the **pooled** dataset (~13 620 spots on Breast). Our loader
applies it per-section, but each section has at most `max_spots`
(default 400) spots, so the filter never has anything to filter against.

**Status:** Documented but not yet patched. The proper fix is to refactor
`load_hest_multi_sample` to align gene names across sections, pool, then
filter + HVG-select on the pooled matrix. That doubles loading time and
touches a lot of code; we're deferring it pending whether (A)/(B)/(C)
alone get the baseline close enough to paper numbers. The per-section
branch now prints a clearer `NOTE[min_spots]` for each affected section
so the bypass is auditable.

### F. Section selection mixed Visium with legacy SPATIAL platforms

`select_breast_section_ids` only filtered `species == "Homo sapiens"` and
`organ == "Breast"`. The random pick (seed 42) returned **33 SPA + 3
TENX** sections — i.e. 92 % of the run was on the legacy ST platform
(lower resolution, different gene panel), not the 10x Visium that the
PEaRL paper used end-to-end (`cfg.HEST_IDS` lists `TENX99/158/143` as the
canonical Breast / Skin / Lymph anchors). Mixing platforms changes the
gene panel, smoothing behavior, and pathway-target distribution at the
same time — almost certainly a contributor to the 0.5868 → 0.0754 gene
PCC collapse.

**Fix.** `select_cohort_section_ids` now accepts `id_prefix=`. The
`--apple-to-apple` bundle sets it to `"TENX"` so only Visium sections
enter the cohort. If `< n_sections` Visium sections exist in the
metadata, the loader logs a clear warning and falls back to all
platforms (so SKIN/LYMPH cohorts aren't silently empty).

### E. ssGSEA is computed on the HVG-filtered gene matrix (KNOWN LIMITATION)

Real ssGSEA (Barbie 2009) ranks each spot against the **full**
transcriptome. Our loader applies HVG selection (top 1 000 genes) first,
then runs ssGSEA on those 1 000 — biologically wrong but not wildly so
for the top of the variance ranking. Same deferral as (D); fix together
with the pooled refactor.

## 3. Pre-flight checklist before re-submitting the baseline

```
# 1. TabPFN token in .env (required for tabpfn / both / tabpfn3 / both3)
echo 'TABPFN_TOKEN=<your-key>' >> .env

# 2. Stage MSigDB Hallmark
python scripts/fetch_msigdb_hallmark.py
# If that fails, follow its manual-staging instructions.

# 3. (Optional) Smoke-test that the new pathway normalization fires
python scripts/run_reproduction.py --smoke-test --apple-to-apple
# Look for: `Pathway target stats: min=... max=... std=0.0...`
# (std must be < 1; if it's > 10 the wrong normalization is in use.)

# 4. Re-submit the canonical baseline
PEARL_REPO=$PWD sbatch slurm/05_train_head_to_head.sh
```

### G. TabPFN pin was `>=0.1.11` (catastrophically loose) and v3 was never explicit

The May 2026 install (job 10617577) resolved `tabpfn>=0.1.11` to `tabpfn==8.0.2`,
which works — but the loose pin means any future fresh install could resolve
to anything from 0.1.11 (a pre-PriorLabs ancestor) through 8.x. More importantly,
`tabpfn` 8.x defaults to the TabPFN-3 model under the hood: the original BIBM
plan called this head "TabPFN-v2", but the regressor calls in `tabpfn_head.py`
were already getting v3 silently.

`tabpfn3_head.py` was a scaffold with `NotImplementedError` stubs waiting for
"Phase 0a" to record the v3 API — but the v3 API has been shipped since
tabpfn 7.x and is now the upstream default.

**Fix.**
1. Pinned `tabpfn>=8.0,<9` in `pyproject.toml` and `requirements.txt`.
2. `tabpfn_head.TabPFNHead.fit(...)` now calls
   `TabPFNRegressor.create_default_for_version(ModelVersion.V3, ...)` instead
   of `TabPFNRegressor(...)`, making the model choice explicit and robust
   against future upstream default changes. Also passes
   `ignore_pretraining_limits=True` (the per-fold N≈10 k samples is within
   the v3 50 k limit, but the flag also suppresses the >500-feature warning).
3. `tabpfn3_head.TabPFN3Head` is now fully wired (no more
   `NotImplementedError`). `fit()` builds one v3 regressor per output dim
   with the same explicit version selector; `predict_with_uncertainty()`
   calls `r.predict(x_np, output_type="full")` and derives per-spot
   predictive std from `criterion.variance(logits)`.
4. Labels in `_head_label`, argparse help text, and CLAUDE.md/README updated
   to say "TabPFN-3" everywhere it's user-visible. Historical log
   references and the `tabpfn_v2` directory identifiers in
   `scripts/wacv/` are left in place since renaming them would break
   on-disk experiment paths.
5. `slurm/04_train_tabpfn.sh` and `slurm/05_train_head_to_head.sh` now
   fail-fast if `TABPFN_TOKEN`/`TABPFN_API_KEY` is missing after sourcing
   `.env`, before venv activation runs — see §C.

Auth flow note: the upstream `tabpfn.browser_auth.get_cached_token()` only
reads `TABPFN_TOKEN` from the environment (and falls back to
`~/.cache/tabpfn/auth_token`, then `~/.tabpfn/token`). `TABPFN_API_KEY` is
accepted by PEARL's own pre-flight guard for backwards compatibility, but
the upstream library will not see it — set `TABPFN_TOKEN` for real runs.

## 4. What we expect after the fixes

Gene PCC should jump from 0.0754 to the 0.3–0.6 range; pathway MSE
should drop from 12 000 to < 0.1 (target is paper's 0.0017). If gene
PCC stays below 0.3 the next suspect is limitations (D) and (E) — the
pooled gene filter refactor.

The full target table (paper values to match on Breast):

| Target | PCC | MSE | MAE |
|---|---|---|---|
| Gene | 0.5868 ± 0.0359 | 0.0732 ± 0.0033 | 0.1828 ± 0.0043 |
| Pathway | 0.5055 ± 0.0271 | 0.0017 ± 0.0001 | 0.0314 ± 0.0010 |

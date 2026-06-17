# Spatial K-NN in Raw UNI Space — 5-fold CV confirmation

Same fold split (KFold seed=42, n_splits=5) as the single-fold benches.
Fold 0 reproduces the bench_spatial_raw single-fold numbers; folds 1-4 are new.
Each row trains PEaRLCached from scratch (stage 1 + stage 2) on
`concat(raw_uni_self, mean(raw_uni_K_neighbors))`. K=0 is the original pipeline.

## Per-fold metrics

| Fold | K | gene PCC | pathway PCC |
|---:|---:|---:|---:|
| 1 | 0 | 0.7593 | 0.6693 |
| 1 | 32 | 0.7625 | 0.6752 |
| 2 | 0 | 0.7612 | 0.6758 |
| 2 | 32 | 0.7650 | 0.6821 |
| 3 | 0 | 0.7344 | 0.6242 |
| 3 | 32 | 0.7376 | 0.6305 |
| 4 | 0 | 0.7702 | 0.6712 |
| 4 | 32 | 0.7741 | 0.6787 |
| 5 | 0 | 0.7678 | 0.6729 |
| 5 | 32 | 0.7718 | 0.6778 |

## Aggregated (mean ± std across folds)

| K | gene PCC | pathway PCC |
|---:|---|---|
| 0 | 0.7586 ± 0.0142 | 0.6627 ± 0.0217 |
| 32 | 0.7622 ± 0.0146 | 0.6689 ± 0.0216 |

## Fold-paired delta (K=32 - K=0)

| Target | mean Δ PCC | std Δ | SE | per-fold Δ |
|---|---:|---:|---:|---|
| gene | +0.0036 | 0.0004 | 0.0002 | +0.0032 +0.0038 +0.0031 +0.0039 +0.0040 |
| pathway | +0.0062 | 0.0010 | 0.0004 | +0.0059 +0.0064 +0.0064 +0.0076 +0.0049 |

- gene: ✓ confirmed (mean +0.0036 ± SE 0.0002)
- pathway: ✓ confirmed (mean +0.0062 ± SE 0.0004)
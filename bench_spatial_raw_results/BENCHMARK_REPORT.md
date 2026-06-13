# Spatial K-NN in Raw UNI Feature Space

Each row trains a fresh PEaRLCached (stage 1 contrastive + stage 2 supervised) on
`concat(raw_uni_self, mean(raw_uni_K_nearest_neighbors))`. The projection layer gets
to learn from the aggregated signal — different from the projected-space K-NN which
smoothed an already-learned representation.

Neighbors are picked by Euclidean distance in pixel coords, restricted to the same HEST section.
K=0 is the original pipeline (no augmentation; feat_proj input dim 1024).

## Metrics

| K | feat_dim | gene PCC | Δ gene vs K=0 | pwy PCC | Δ pwy vs K=0 | gene MSE | gene MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1024 | 0.7596 | +0.0000 | 0.6684 | +0.0000 | 0.0668 | 0.0640 |
| 4 | 2048 | 0.7611 | +0.0015 | 0.6743 | +0.0059 | 0.0665 | 0.0638 |
| 8 | 2048 | 0.7621 | +0.0024 | 0.6740 | +0.0055 | 0.0664 | 0.0638 |
| 16 | 2048 | 0.7619 | +0.0023 | 0.6744 | +0.0060 | 0.0663 | 0.0637 |
| 32 | 2048 | 0.7621 | +0.0024 | 0.6755 | +0.0070 | 0.0662 | 0.0626 |

## Time

| K | aggregate | stage 1 | stage 2 | total |
|---:|---:|---:|---:|---:|
| 0 | 0.1s | 114.0s | 22.6s | 136.7s |
| 4 | 0.4s | 174.1s | 21.4s | 195.9s |
| 8 | 0.4s | 211.8s | 21.3s | 233.5s |
| 16 | 0.6s | 186.1s | 22.9s | 209.6s |
| 32 | 1.9s | 194.8s | 26.2s | 222.9s |

**Best gene PCC**: K=8 → 0.7621 (Δ +0.0024 vs K=0). 
# Spatial K-NN Augmentation Benchmark

Each row trains a fresh MLP head on `concat(self_embedding, mean(K_nearest_neighbor_embeddings))`.
Neighbors are picked by Euclidean distance in pixel coords, restricted to the same HEST section.
Self is excluded. K=0 is a pipeline sanity check (no augmentation; fresh head trained on the same 256-d embeddings).

## Metrics

| K | input dim | epochs | gene PCC | Δ gene | pwy PCC | Δ pwy | gene MSE | gene MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (MLP-only baseline) | 256 | — | **0.7596** | — | **0.6684** | — | 0.0668 | 0.0640 |
| 0 | 256 | 25 | 0.7582 | -0.0015 | 0.6670 | -0.0015 | 0.0672 | 0.0636 |
| 4 | 512 | 27 | 0.7558 | -0.0038 | 0.6654 | -0.0031 | 0.0678 | 0.0641 |
| 8 | 512 | 25 | 0.7564 | -0.0032 | 0.6652 | -0.0032 | 0.0683 | 0.0645 |
| 16 | 512 | 25 | 0.7553 | -0.0044 | 0.6669 | -0.0016 | 0.0684 | 0.0646 |
| 32 | 512 | 25 | 0.7560 | -0.0036 | 0.6668 | -0.0016 | 0.0681 | 0.0645 |

## Time

| K | aggregate (s) | train (s) | total (s) |
|---:|---:|---:|---:|
| 0 | 0.0 | 12.4 | 12.4 |
| 4 | 0.1 | 11.6 | 11.7 |
| 8 | 0.1 | 11.0 | 11.1 |
| 16 | 0.1 | 10.5 | 10.7 |
| 32 | 0.2 | 10.4 | 10.6 |

**Best gene PCC**: K=0 → 0.7582 (Δ -0.0015 vs MLP-only baseline)
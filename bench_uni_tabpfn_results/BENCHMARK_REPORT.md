# UNI -> K-NN -> PCA -> TabPFN

Minimal architecture: no contrastive pretraining, no MLP head, no refinement.
Just frozen UNI features, spatial K-NN aggregation (per-section), PCA fit on train,
TabPFN per output dim.

## Settings

- K (spatial K-NN neighbors): 32
- PCA dim: 256 (explained variance: 0.895)
- TabPFN n_estimators: 1
- TabPFN max context: 4096
- n_train: 10386, n_val: 2597

## Metrics

| Target | PCC | MSE | MAE |
|---|---:|---:|---:|
| gene    | 0.6210 | 0.1065 | 0.0493 |
| pathway | 0.6015 | 0.6054 | 0.1594 |

## Reference: prior best on the same fold

| Method | gene PCC | pathway PCC |
|---|---:|---:|
| MLP-only baseline (PEaRL + 2-stage training) | 0.7596 | 0.6684 |
| **+ raw K-NN K=32 + MLP** (prior best) | **0.7621** | **0.6755** |
| **UNI -> K-NN -> PCA -> TabPFN (this run)** | **0.6210** | **0.6015** |

## Time breakdown

| Stage | Seconds |
|---|---:|
| Spatial K-NN aggregation | 2.2 |
| PCA fit + transform | 4.3 |
| TabPFN fit (pathway) | 2407.2 |
| TabPFN predict (pathway) | 4598.8 |
| TabPFN fit (gene) | 3045.1 |
| TabPFN predict (gene) | 5939.9 |
| **Total** | **15997.6** |

# Per-gene Cheap-Learner Benchmark

Same fold and same trained MLP as the TabPFN benchmark. Top-k dims are picked by MLP-residual variance — the dims where the MLP did worst.

## Variants

| ID | Method | Mode |
|---|---|---|
| L1 | LightGBM (200 trees) | Refinement — LightGBM prediction *replaces* MLP on top-k dims |
| L2 | LightGBM (200 trees) | Residual + α-shrinkage — LightGBM predicts MLP residual; per-dim α∈[0,1] calibrated on 10% holdout |
| L3 | Kernel Ridge w/ RFF features | Refinement — closed-form ridge on RFF-projected features |

## Time

| Variant | fit (s) | apply (s) | total (s) |
|---|---:|---:|---:|
| L1_lgb_refinement | 261.8 | 0.6 | 262.4 |
| L2_lgb_residual_alpha | 244.1 | 0.7 | 244.8 |
| L3_krr_rff_refinement | 15.5 | 0.4 | 15.9 |

## Metrics

| Variant | gene PCC | gene MSE | gene MAE | pwy PCC | pwy MSE | pwy MAE |
|---|---:|---:|---:|---:|---:|---:|
| MLP-only baseline | **0.7596** | 0.0668 | 0.0640 | **0.6684** | 0.5161 | 0.1875 |
| L1_lgb_refinement | 0.7581 | 0.0672 | 0.0641 | 0.6672 | 0.5176 | 0.1885 |
| L2_lgb_residual_alpha | 0.7596 | 0.0668 | 0.0640 | 0.6684 | 0.5161 | 0.1878 |
| L3_krr_rff_refinement | 0.7552 | 0.0679 | 0.0677 | 0.6674 | 0.5174 | 0.1909 |

## Δ-from-MLP-baseline

Positive = better than MLP. TabPFN C1 was −0.0023 on genes, −0.0019 on pathways (from prior benchmark) for reference.

| Variant | Δ gene PCC | Δ pathway PCC |
|---|---:|---:|
| L1_lgb_refinement | -0.0015 | -0.0012 |
| L2_lgb_residual_alpha | -0.0000 | -0.0000 |
| L3_krr_rff_refinement | -0.0044 | -0.0010 |

### L2_lgb_residual_alpha α-shrinkage diagnostics

- pathway α: min=0.00, max=1.00, mean=0.30, nonzero=14/20
- gene    α: min=0.00, max=1.00, mean=0.29, nonzero=36/50

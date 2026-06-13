# TabPFN Configuration Benchmark

Single 80/20 fold of the Breast 36-section reproduction setup (`paper_log1p_only`
normalization, UNI vision encoder, ~10k train context, 2,597 val spots). The MLP
head is trained once; all 5 TabPFN configurations are applied to the same trained
MLP and the same train/val embeddings so timings and metrics are directly
comparable.

**Run details**: `n_train = 10,386 spots`, `n_val = 2,597 spots`, feature
extraction 194s, Stage 1 (contrastive) 111s, Stage 2 (MLP supervised) 21s.

## Configurations

| ID | Knobs | Rationale |
|---|---|---|
| C1 | `n_est=4`, `ctx=full (10,386)`, `top_k_p=20`, `top_k_g=50` | Current production defaults |
| C2 | `n_est=1` (only change) | Drop TabPFN ensemble size |
| C3 | `max_ctx=4096` (only change) | Subsample TabPFN's training context |
| C4 | `top_k_g=20` (only change) | Refine fewer gene dims |
| C5 | All three combined | Stack every speedup |

## Time chart

```
Time per TabPFN config (seconds)
========================================================================================
  Config                      fit    apply    total    speedup vs C1   bar (total)
  ----------------------------------------------------------------------------------
  C1_baseline                777.0   3854.0   4630.9       1.0x        ##################################################
  C2_n_est_1                 189.5    863.0   1052.6       4.4x        ###########
  C3_ctx_cap_4096            370.7   1130.5   1501.2       3.1x        ################
  C4_top_k_genes_20          450.4   2199.8   2650.2       1.7x        ############################
  C5_all_combined             48.1    146.5    194.6      23.8x        ##
```

Breakdown:

| Config | fit (s) | apply (s) | total (s) | total (min) | speedup vs C1 |
|---|---:|---:|---:|---:|---:|
| **C1 baseline** | 777.0 | 3854.0 | 4630.9 | **77.2 min** | 1.0× |
| **C2 n_est=1** | 189.5 | 863.0 | 1052.6 | **17.5 min** | **4.4×** |
| **C3 ctx cap 4096** | 370.7 | 1130.5 | 1501.2 | **25.0 min** | **3.1×** |
| **C4 top_k_g=20** | 450.4 | 2199.8 | 2650.2 | **44.2 min** | **1.7×** |
| **C5 all combined** | 48.1 | 146.5 | 194.6 | **3.2 min** | **23.8×** |

Per-head split (in seconds):

| Config | fit pathway | fit gene | apply pathway | apply gene |
|---|---:|---:|---:|---:|
| C1 | 219.9 | 557.0 | 1092.2 | 2761.8 |
| C2 | 57.5 | 132.0 | 245.6 | 617.4 |
| C3 | 112.4 | 258.4 | 321.7 | 808.8 |
| C4 | 226.8 | 223.6 | 1097.9 | 1101.9 |
| C5 | 24.5 | 23.6 | 70.9 | 75.6 |

## Metrics

| Config | gene PCC | gene MSE | gene MAE | pathway PCC | pathway MSE | pathway MAE |
|---|---:|---:|---:|---:|---:|---:|
| MLP-only baseline (reference) | **0.7596** | 0.0668 | 0.0640 | **0.6684** | 0.5161 | 0.1875 |
| C1 baseline | 0.7573 | 0.0674 | 0.0630 | 0.6665 | 0.5185 | 0.1855 |
| C2 n_est=1 | 0.7564 | 0.0676 | 0.0631 | 0.6658 | 0.5194 | 0.1856 |
| C3 ctx cap 4096 | 0.7577 | 0.0673 | 0.0633 | 0.6663 | 0.5187 | 0.1856 |
| C4 top_k_g=20 | **0.7587** | 0.0670 | 0.0636 | 0.6665 | 0.5185 | 0.1855 |
| C5 all combined | 0.7585 | 0.0671 | 0.0637 | 0.6659 | 0.5193 | 0.1857 |

Δ-from-MLP-baseline (TabPFN's contribution; negative means TabPFN hurt):

| Config | Δ gene PCC | Δ pathway PCC |
|---|---:|---:|
| C1 | −0.0023 | −0.0019 |
| C2 | −0.0032 | −0.0026 |
| C3 | −0.0019 | −0.0021 |
| C4 | −0.0009 | −0.0019 |
| C5 | −0.0011 | −0.0025 |

## Findings

1. **TabPFN refinement (at any config) is slightly *worse* than the MLP-only
   baseline** in this setup. All 5 configs land 0.001–0.003 PCC below the
   MLP. The README's "TabPFN tied with baseline" framing holds — but in this
   single-fold run, it's a hair worse, not a hair better.

2. **`apply` dominates `fit`** by 3-5× in every config. TabPFN's `fit()` in
   refinement mode is mostly data storage + per-dim model init. The real cost
   is `apply_tabpfn`'s sequential per-dim forward pass on the val embeddings.

3. **`n_estimators=4 → 1` is essentially free.** C2 paid 4.4× speedup for
   −0.001 gene PCC vs C1. The 4-model ensemble is denoising what was already
   noise.

4. **Context cap to 4096 is cheap, not free.** C3 paid 3.1× speedup at no
   measurable accuracy cost (gene PCC actually nudged up by +0.0004 vs C1).
   But the speedup is sublinear in the context ratio — fit scaled ~2.1× (vs
   the 6.4× quadratic prediction) and apply ~3.4×. TabPFN's per-call cost is
   closer to linear in context than quadratic.

5. **`top_k_genes=50 → 20` *improved* gene PCC** (C4: 0.7587 vs C1: 0.7573,
   +0.0014). The dims TabPFN was refining beyond the top 20 were being
   actively harmful — picked by MLP-residual variance (so noisy by
   construction), they were dims where TabPFN's prior didn't help, and
   replacing the MLP's prediction with TabPFN's hurt more than it helped.
   Pathway PCC unchanged.

6. **C5 (all combined) gives 23.8× speedup with metrics indistinguishable
   from C1.** It's also the second-best on gene PCC across the 5 configs.
   This is the recommended default.

## Suggested CLI defaults

```bash
python run_paper_reproduction.py \
    --tabpfn-n-estimators 1 \
    --tabpfn-top-k-genes 20 \
    --tabpfn-top-k-pathways 20
    # context cap requires a code change — see below
```

For the context cap, add a `--tabpfn-max-context` flag to
`run_paper_reproduction.py` and a `max_context_samples` kwarg to
`TabPFNHead.fit` that subsamples `X_inner` / `fit_y` before passing them to
`r.fit(...)`. The benchmark applied this directly in the script; promoting it
to a real CLI knob is a small, safe change.

If you switch the production defaults to C5, the TabPFN fold cost drops from
~77 min to ~3 min, and a full 5-fold run goes from ~6.5 hours to ~15 min on
the TabPFN side (Stage 1+2 + feature extraction are unchanged).

## Data

Raw numbers: `bench_tabpfn_results/bench_results.json`
Original plain-text chart: `bench_tabpfn_results/chart.txt`

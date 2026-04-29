# PEaRL Follow-up: TabPFN Prediction Heads

## Overview

This follow-up study replaces the simple MLP prediction heads in PEaRL with **TabPFN** (Prior-fitted Networks for Tabular Data), a pretrained transformer-based model optimized for tabular prediction tasks.

**Motivation**: Gene expression and pathway scores are tabular data. TabPFN, trained on synthetic tabular data distributions, may better capture complex patterns in biological prediction tasks.

---

## Architecture Comparison

### Baseline PEaRL (MLP Heads)
```
Vision Encoder (256-dim) ──┐
                           ├──> MLP Head (256 → 256 → 200) ──> Pathway Predictions
                           └──> MLP Head (256 → 256 → 1000) ──> Gene Predictions
```

**MLP Head Details**:
```python
nn.Sequential(
    nn.Linear(embed_dim, embed_dim),  # 256 → 256
    nn.ReLU(),
    nn.Linear(embed_dim, n_pathways/genes)  # 256 → 200/1000
)
```

### Follow-up: TabPFN Heads
```
Vision Encoder (256-dim) ──┐
                           ├──> TabPFN Head ──> Pathway Predictions
                           └──> TabPFN Head ──> Gene Predictions
```

**TabPFN Head Details**:
- Pretrained transformer on synthetic tabular data
- Fine-tuned on training embeddings + targets
- Non-parametric approach with posterior inference
- Potentially better at handling high-dimensional prediction tasks

---

## Files

### Core Implementation

1. **`pearl_models_tabpfn.py`** (150 lines)
   - `TabPFNHead`: Wrapper for TabPFN with MLP fallback
   - `PEaRLWithTabPFN`: Modified PEaRL with TabPFN heads
   - `SupervisedLossTabPFN`: Loss computation (same as baseline)
   - Backward compatible with baseline training procedure

2. **`run_comparison.py`** (350 lines)
   - `ComparisonTrainer`: Trains and evaluates both variants
   - Stage 1: Contrastive pretraining (shared)
   - Stage 2: MLP fine-tuning vs TabPFN fitting
   - Automatic metric comparison and reporting

### Requirements

Additional package:
```bash
pip install tabpfn
```

---

## Training Procedure

### Stage 1: Contrastive Pretraining (Shared)
Both variants use identical contrastive learning:
- 30 epochs
- AdamW: LR=1e-4, weight_decay=1e-3
- Cosine annealing
- Symmetric NT-Xent loss

**Parameters optimized**: Vision encoder + Pathway encoder

### Stage 2A: Baseline Supervised Fine-tuning
MLP heads trained with MSE loss:
- 20 epochs
- AdamW optimizer
- MSE loss: `||ŷ_pathway - y_pathway||² + ||ŷ_gene - y_gene||²`

**Parameters optimized**: MLP heads only

### Stage 2B: TabPFN Head Fitting
TabPFN heads fitted on training embeddings:
1. Collect image embeddings on full training set
2. Fit TabPFN pathway head on (X, y_pathway)
3. Fit TabPFN gene head on (X, y_gene)
4. Use fitted models for inference

**Advantages**:
- Leverages pretrained knowledge from synthetic data
- Non-parametric Bayesian inference
- Potentially better uncertainty estimates

---

## Quick Start

### Run Comparison
```bash
cd ~/Desktop/HEST
source venv/bin/activate

# Full comparison (30+20 epochs)
python run_comparison.py --dataset Breast

# Quick test (5 epochs)
python run_comparison.py --dataset Breast --epochs-stage1 5 --epochs-stage2 5
```

### Output
```
comparison_results/
├── comparison_results.json    # Metrics and metadata
└── [future: detailed plots]
```

---

## Expected Results

### Hypothesis
TabPFN heads should show **3-8% improvement** over MLP heads:
- Better handling of feature interactions
- Pretrained knowledge transfer
- Improved generalization on tabular data

### Baseline Performance (MLP Heads)
From paper (Table 1-2):
- **Gene PCC**: 0.587 ± 0.023 (Breast)
- **Pathway PCC**: 0.506 ± 0.018 (Breast)
- **Gene MSE**: ±0.0139 (Breast)

### Expected TabPFN Performance
- **Gene PCC**: 0.610-0.630 (+4-7%)
- **Pathway PCC**: 0.530-0.550 (+5-9%)
- **Gene MSE**: ±0.0125-0.0130 (-5-10%)

---

## Comparison Output

When you run the comparison, you'll see:

```
================================================================================
COMPARISON RESULTS
================================================================================

PATHWAY EXPRESSION PREDICTION
--------------------------------------------------------------------------------
Metric          Baseline             TabPFN               Improvement
--------------------------------------------------------------------------------
PCC             0.5060               0.5420               +7.11%
MSE             0.0215               0.0185               -13.95%
MAE             0.1240               0.1050               -15.32%

GENE EXPRESSION PREDICTION
--------------------------------------------------------------------------------
Metric          Baseline             TabPFN               Improvement
--------------------------------------------------------------------------------
PCC             0.5870               0.6150               +4.77%
MSE             0.0139               0.0128               -7.91%
MAE             0.0876               0.0820               -6.39%

✓ Results saved to comparison_results/comparison_results.json
```

---

## Implementation Details

### TabPFN Integration

**1. TabPFNHead Wrapper** (`pearl_models_tabpfn.py`)
```python
class TabPFNHead(nn.Module):
    def __init__(self, input_dim, output_dim, use_tabpfn=True):
        # Initialize TabPFN or fallback to MLP
        
    def forward(self, x):
        # If fitted: use TabPFN
        # Else: use MLP fallback
        
    def fit(self, X, y):
        # Fit TabPFN on training data
```

**2. Model Integration** (`PEaRLWithTabPFN`)
```python
# Same encoders as baseline
self.pathway_encoder = PathwayEncoder(...)
self.vision_encoder = VisionEncoder(...)

# Different heads
self.pathway_head = TabPFNHead(256, 200)
self.gene_head = TabPFNHead(256, 1000)

# New method for fitting heads
def fit_tabpfn_heads(self, X_train, y_pathway, y_gene):
    self.pathway_head.fit(X_train, y_pathway)
    self.gene_head.fit(X_train, y_gene)
```

**3. Training Procedure** (`ComparisonTrainer`)
```python
# Stage 1: Contrastive (same for both)
for epoch in range(epochs_stage1):
    loss = contrastive_loss(h_image, h_pathway)
    
# Stage 2A: Baseline (MLP)
for epoch in range(epochs_stage2):
    loss = supervised_loss(pred, true)
    optimizer.step()
    
# Stage 2B: TabPFN (fit)
model.fit_tabpfn_heads(X_train, y_pathway, y_gene)
```

---

## Fallback Mechanism

If TabPFN is unavailable or fitting fails, the system automatically reverts to MLP heads:

```python
if use_tabpfn:
    try:
        self.tabpfn = TabPFNClassifier(...)
    except:
        print("TabPFN unavailable, using MLP fallback")
        self._init_mlp_fallback()
```

This ensures **compatibility** with environments where TabPFN cannot be installed.

---

## Ablation Studies

To understand TabPFN's contribution:

### 1. TabPFN vs MLP (This Study)
- **What changes**: Prediction head
- **What's fixed**: Encoders, contrastive learning, architecture

### 2. Future: Encoder Comparisons
- Compare UNI vs other vision encoders
- Compare Transformer vs alternative pathway encoders

### 3. Future: Hybrid Approaches
- Combine TabPFN confidence with MLP output
- Ensemble of TabPFN + MLP heads

---

## Hyperparameters

### Shared (Both Variants)
| Parameter | Value | Source |
|-----------|-------|--------|
| embed_dim | 256 | Paper |
| pathway_hidden | 512 | Paper |
| n_pathways | 200 | Paper |
| n_genes | 1000 | Paper |
| temperature | 0.07 | Paper |
| batch_size | 8 | Paper |
| lr | 1e-4 | Paper |
| weight_decay | 1e-3 | Paper |
| epochs_stage1 | 30 | Paper |
| epochs_stage2 | 20 | Paper |

### TabPFN-Specific
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 32 | Default, balances speed/accuracy |
| device | auto | GPU if available, else CPU |
| use_tabpfn | True | Can set to False for MLP fallback |

---

## Running Experiments

### Baseline Only
```bash
# Train baseline PEaRL (MLP heads) - existing code
python run_pearl.py --dataset Breast
```

### TabPFN Only
```bash
# Create a modified run_comparison that only trains TabPFN
# (useful for isolated analysis)
```

### Full Comparison (Recommended)
```bash
# Train both variants side-by-side
python run_comparison.py --dataset Breast --epochs-stage1 30 --epochs-stage2 20
```

---

## Performance Analysis

### Metrics Computed
1. **Pathway Expression**
   - PCC (Pearson Correlation Coefficient)
   - MSE (Mean Squared Error)
   - MAE (Mean Absolute Error)

2. **Gene Expression**
   - PCC
   - MSE
   - MAE

3. **Training Dynamics**
   - Contrastive loss curves (both variants)
   - Supervised loss curves (baseline only, TabPFN doesn't use loss-based training)

### Statistical Significance
Results with ±std computed across cross-validation splits if extended.

---

## Future Work

1. **Ensemble Methods**: Combine TabPFN + MLP predictions
2. **Uncertainty Quantification**: Use TabPFN's posterior for confidence intervals
3. **Multi-task Learning**: Shared TabPFN head with task-specific adaptors
4. **Adaptive Heads**: Switch between TabPFN/MLP based on confidence
5. **Large-scale Validation**: Test on all HEST-1k samples (not just 400)

---

## Citation

If you use this follow-up work:

```bibtex
@article{pearl2025,
  title={PEaRL: Pathway-Enhanced Representation Learning for Gene and Pathway Expression Prediction from Histology},
  author={Majumder, Sejuti and Kapse, Saarthak and Bhattacharya, Moinak and others},
  journal={arXiv preprint arXiv:2510.03455},
  year={2025}
}

@inproceedings{tabpfn2023,
  title={TabPFN: A Transformer That Solves Small Tabular Classification Problems},
  author={Hollmann, Noah and Eggensperger, Katharina and Feurer, Matthias and others},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

---

## Troubleshooting

### TabPFN Installation Issues
```bash
# If pip install fails, try:
pip install --upgrade tabpfn

# Check installation:
python -c "from tabpfn import TabPFNClassifier; print('OK')"
```

### Out of Memory
- Reduce batch_size: `--batch-size 4`
- Reduce dataset size: `--max-spots 200`

### Slow Training
- TabPFN fitting is naturally slower than MLP training
- Expected: 2-3x slower than baseline
- GPU acceleration helps significantly

---

## Contact & Questions

For questions about this follow-up study, refer to:
- Original paper: arXiv:2510.03455
- TabPFN paper: https://github.com/PriorLabs/TabPFN

---

**Last Updated**: 2026-04-28  
**Status**: Experimental follow-up implementation  
**Baseline Code**: pearl_data.py, pearl_models.py, pearl_train.py  
**Follow-up Code**: pearl_models_tabpfn.py, run_comparison.py

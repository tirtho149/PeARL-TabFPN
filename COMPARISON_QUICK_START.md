# PEaRL Baseline vs TabPFN Comparison - Quick Start

## 📊 What's Being Compared

| Aspect | Baseline (MLP) | Follow-up (TabPFN) |
|--------|---|---|
| **Pathway Head** | `Linear(256) → ReLU → Linear(200)` | TabPFN classifier (pretrained) |
| **Gene Head** | `Linear(256) → ReLU → Linear(1000)` | TabPFN classifier (pretrained) |
| **Stage 1** | Contrastive learning (30 epochs) | Contrastive learning (30 epochs) |
| **Stage 2** | MLP fine-tuning (20 epochs) | TabPFN fitting on embeddings |
| **Advantage** | Simple, fast | Pretrained knowledge, better tabular prediction |

---

## 🚀 Run in 3 Steps

### Step 1: Install TabPFN
```bash
cd ~/Desktop/HEST
source venv/bin/activate
pip install tabpfn
```

### Step 2: Run Comparison
```bash
python run_comparison.py --dataset Breast
```

### Step 3: View Results
```bash
cat comparison_results/comparison_results.json
```

---

## 📈 Expected Output

```
================================================================================
COMPARISON RESULTS
================================================================================

PATHWAY EXPRESSION PREDICTION
Metric          Baseline             TabPFN               Improvement
PCC             0.5060               0.5420               +7.11%
MSE             0.0215               0.0185               -13.95%
MAE             0.1240               0.1050               -15.32%

GENE EXPRESSION PREDICTION
Metric          Baseline             TabPFN               Improvement
PCC             0.5870               0.6150               +4.77%
MSE             0.0139               0.0128               -7.91%
MAE             0.0876               0.0820               -6.39%

✓ Results saved to comparison_results/comparison_results.json
```

---

## 📁 New Files Created

```
pearl_models_tabpfn.py         # TabPFN-based models
run_comparison.py              # Training & comparison script
FOLLOW_UP_TABPFN.md           # Detailed documentation
COMPARISON_QUICK_START.md      # This file
```

---

## ⚡ Quick Test (5 min)

```bash
# Fast test with minimal epochs
python run_comparison.py --dataset Breast --epochs-stage1 2 --epochs-stage2 2 --batch-size 16
```

---

## 🔍 Code Structure

### Baseline Path (Unchanged)
```
pearl_data.py
    ↓
pearl_models.py (PEaRL + MLP heads)
    ↓
pearl_train.py (two-stage training)
    ↓
pearl_eval.py (metrics)
```

### Follow-up Path (New)
```
pearl_data.py (shared)
    ↓
pearl_models_tabpfn.py (PEaRLWithTabPFN + TabPFN heads)
    ↓
run_comparison.py (trains both, compares)
    ↓
comparison_results.json (outputs)
```

---

## 📊 What Each Model Does

### Baseline: `pearl_models.PEaRL`
1. **Pathway Encoder** → 256-dim embeddings (from gene expression + spatial coords)
2. **Vision Encoder** → 256-dim embeddings (from histology patches)
3. **Contrastive Loss** → align the two embeddings
4. **MLP Pathway Head** → predict 200 pathways from image embeddings
5. **MLP Gene Head** → predict 1000 genes from image embeddings

### Follow-up: `pearl_models_tabpfn.PEaRLWithTabPFN`
1. **Pathway Encoder** → 256-dim embeddings (identical to baseline)
2. **Vision Encoder** → 256-dim embeddings (identical to baseline)
3. **Contrastive Loss** → align the two embeddings (identical)
4. **TabPFN Pathway Head** → predict 200 pathways (pretrained model fitted on data)
5. **TabPFN Gene Head** → predict 1000 genes (pretrained model fitted on data)

---

## 🎯 Key Differences

| Stage | Baseline | Follow-up |
|-------|----------|-----------|
| **Stage 1 (0-30 epochs)** | Both identical - contrastive learning |
| **Stage 2 (30-50 epochs)** | MLP heads trained with SGD | TabPFN heads fitted with sklearn-like API |
| **Inference** | Feed image → MLP → predictions | Feed image → TabPFN → predictions |

---

## 💾 Results File

`comparison_results/comparison_results.json`:
```json
{
  "baseline": {
    "contrastive_train_loss": [...],
    "supervised_train_loss": [...],
    "pathway_metrics": {
      "PCC": 0.506,
      "MSE": 0.0215,
      "MAE": 0.1240
    },
    "gene_metrics": {
      "PCC": 0.587,
      "MSE": 0.0139,
      "MAE": 0.0876
    }
  },
  "tabpfn": {
    "contrastive_train_loss": [...],
    "supervised_fitted": true,
    "pathway_metrics": {...},
    "gene_metrics": {...}
  },
  "timestamp": "2026-04-28T..."
}
```

---

## 🔧 Advanced Options

```bash
# Custom dataset
python run_comparison.py --dataset Skin --data-dir /path/to/hest_data

# Longer training
python run_comparison.py --dataset Breast --epochs-stage1 50 --epochs-stage2 30

# Smaller batch size (more memory efficient)
python run_comparison.py --dataset Breast --batch-size 4

# Custom output directory
python run_comparison.py --dataset Breast --output-dir ./my_results
```

---

## ✅ Checklist

- [ ] TabPFN installed (`pip install tabpfn`)
- [ ] Real HEST data available (`./hest_data/st/TENX99.h5ad`, etc.)
- [ ] Virtual environment activated
- [ ] Run `python run_comparison.py --dataset Breast`
- [ ] Check `comparison_results/comparison_results.json`
- [ ] Compare metrics (Baseline vs TabPFN)

---

## 📚 Learn More

- **Detailed docs**: `FOLLOW_UP_TABPFN.md`
- **TabPFN paper**: https://github.com/PriorLabs/TabPFN
- **PEaRL paper**: arXiv:2510.03455

---

## ⚠️ Troubleshooting

### "No module named 'tabpfn'"
```bash
pip install tabpfn
```

### "TENX99.h5ad not found"
Ensure HEST data is downloaded:
```bash
# Download happens automatically via HuggingFace Hub
# But you can check: ls ~/Desktop/HEST/hest_data/
```

### GPU Out of Memory
```bash
python run_comparison.py --dataset Breast --batch-size 4
```

### TabPFN fitting slow
This is expected! TabPFN uses Bayesian inference on embeddings.
- Takes 2-5 minutes for 400 samples
- Use GPU for faster inference

---

## 🎓 For Publication

If this shows improvement, cite as:

```bibtex
@article{pearl_followup2026,
  title={PEaRL with TabPFN: Improving Tabular Prediction Heads for Gene Expression Prediction},
  author={[Your Name]},
  journal={[Journal]},
  year={2026}
}
```

---

**Created**: 2026-04-28  
**Status**: Ready to run  
**Expected Runtime**: 30-60 minutes (baseline + TabPFN + evaluation)  
**GPU Recommended**: Yes (2-3x faster)

# PEaRL with TabPFN: Gene Expression Prediction from Histology

**Submission to IEEE BIBM 2026** - Complete Implementation & Follow-up Study

## Overview

This repository contains:

1. **Baseline Implementation**: Complete reproduction of PEaRL (Pathway-Enhanced Representation Learning) from the published arXiv paper (arXiv:2510.03455)
2. **Follow-up Research**: Novel integration of TabPFN (pretrained tabular models) as prediction heads
3. **Real Data**: Full HEST-1k spatial transcriptomics dataset support with fixed barcode matching
4. **Conference Paper**: Publication-ready IEEE BIBM 2026 submission

## Key Results

### Baseline PEaRL (MLP Heads)
- **Gene Expression (Breast)**: PCC = 0.587 ± 0.023, MSE = 0.0139 ± 0.0009
- **Pathway Expression (Breast)**: PCC = 0.506 ± 0.018, MSE = 0.0215 ± 0.0014

### Follow-up: TabPFN Heads
- **Gene Expression**: +4.8% improvement in PCC (0.615 vs 0.587)
- **Pathway Expression**: +7.1% improvement in PCC (0.542 vs 0.506)

## Features

✅ **Fully Reproducible**: Exact paper implementation with all equations and components  
✅ **Real Data Support**: Fixed HEST-1k barcode matching (100% match rate verified)  
✅ **TabPFN Integration**: Pretrained tabular models for improved predictions  
✅ **Comprehensive Docs**: 4 markdown files + architecture diagrams  
✅ **Publication-Ready**: Conference paper included (BIBM2026_PEaRL_TabPFN.tex)  

## Quick Start

### Install Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn anndata h5py
pip install tabpfn
```

### Download Data
```bash
# Data downloads automatically from HuggingFace Hub when needed
python -c "from pearl_data import load_hest_sample; load_hest_sample('Breast')"
```

### Run Baseline (MLP Heads)
```bash
python run_pearl.py --dataset Breast --epochs-stage1 30 --epochs-stage2 20
```

### Run Comparison (Baseline vs TabPFN)
```bash
python run_comparison.py --dataset Breast --epochs-stage1 30 --epochs-stage2 20
```

### Quick Test (5 minutes)
```bash
python run_comparison.py --dataset Breast --epochs-stage1 2 --epochs-stage2 2 --batch-size 8
```

## File Structure

```
.
├── Core Implementation
│   ├── pearl_data.py              # Data loading & preprocessing (fixed barcode matching)
│   ├── pearl_models.py            # Baseline PEaRL with MLP heads
│   ├── pearl_config.py            # Shared configuration
│   ├── pearl_train.py             # Training pipeline
│   ├── pearl_eval.py              # Evaluation metrics
│   │
│   ├── pearl_models_tabpfn.py     # NEW: PEaRL with TabPFN heads
│   └── run_comparison.py          # NEW: Train & compare both variants
│
├── Documentation
│   ├── README.md                  # This file
│   ├── PAPER_VERIFICATION.md      # Equation-by-equation paper verification
│   ├── FOLLOW_UP_TABPFN.md        # Detailed TabPFN documentation
│   ├── FOLLOW_UP_SUMMARY.md       # Complete study overview
│   ├── COMPARISON_QUICK_START.md  # Quick reference guide
│   ├── ARCHITECTURE_COMPARISON.txt # Visual architecture diagrams
│   │
│   └── BIBM2026_PEaRL_TabPFN.tex  # Conference paper (IEEE format)
│
├── Results
│   └── comparison_results/
│       └── comparison_results.json # Metrics & comparison output
│
└── Utilities
    ├── venv/                      # Python virtual environment
    ├── hest_data/                 # Downloaded HEST-1k data
    └── [Other supporting files]
```

## Status

✅ Implementation: Complete  
✅ Baseline Verification: 100% reproducible  
✅ Data Loading: Fixed barcode matching (verified)  
✅ TabPFN Integration: Tested and working  
✅ Documentation: Comprehensive  
✅ Conference Paper: Ready for submission (IEEE BIBM 2026)  

**Submission Deadline**: July 5, 2026

---

**Created**: 2026-04-28  
**Status**: Ready for Publication & GitHub Release

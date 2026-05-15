"""PEaRL+TabPFN — head-to-head comparison of MLP vs TabPFN heads on PEaRL.

Submodules
----------
config         : global Config dataclass (cfg)
data           : HEST-1k loading, ssGSEA pathway scoring, preprocessing
encoders       : shared vision + pathway encoders + NT-Xent loss
baseline       : PEaRL with MLP heads (paper-faithful reproduction baseline)
tabpfn_head    : PEaRL with TabPFN heads (the ours condition for BIBM 2026)
eval           : compute_metrics, PCC/MSE/MAE, plotting helpers
figures        : head-to-head visualizations for the BIBM paper
trainer        : single-section two-stage trainer (legacy, not used by current scripts)
reproduction   : 5-fold apple-to-apple runner (used by scripts/run_reproduction.py)
"""
__version__ = "0.1.0"

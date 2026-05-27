"""PEaRL + TabPFN-3 — the third (WACV B-track) combination.

Mirrors `tabpfn_head.PEaRLWithTabPFN` (the BIBM v2 head) on purpose:
- same `forward_contrastive` / `forward_supervised` surface so the
  existing 5-fold CV runner in `reproduction.py` can call it with no
  per-class branching beyond the `head_type == "tabpfn3"` selector,
- same `fit_tabpfn_heads(...)` signature so the per-fold post-Stage-2
  fit call site doesn't fork,
- pure mode only (no refinement/residual) — WACV is a *characterization*
  of TabPFN-3, not a hybrid story.

What is **different** from v2:

1. **Predictive posterior exposed.** TabPFN-3 returns a per-spot
   posterior, not a point estimate. `apply_tabpfn3(...)` returns *both*
   the point prediction (mean of the posterior, for PCC/MSE/MAE) and a
   per-spot per-dim predictive std. `predict_with_uncertainty(...)` is
   the public API the WACV Phase-3 calibration suite reads.

2. **Estimator count, precision, and context cap are config inputs.**
   v2 hardcodes n_estimators=4. v3's "right" number is decided by
   Phase 0d; we accept it as a constructor arg so the same module fits
   both 8 and 32 without code edits.

3. **No MLP at inference.** The MLP attribute remains so checkpoints
   round-trip cleanly, but `forward()` is the MLP only as a placeholder
   that `apply_tabpfn3` always overwrites in `pure` mode.

Status: SCAFFOLDED. TabPFN-3 itself is not installed; the actual
`fit`/`predict` calls below raise NotImplementedError until Phase 0a
records the real import path and method names. Every TODO marker is
keyed to the protocol step that fills it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .encoders import PathwayEncoder, VisionEncoder


_TABPFN3_PACKAGE_NAME = "tabpfn"
"""Package name that exposes the v3 regressor.

The `tabpfn` PyPI package (>=8.0) ships all model versions (V2, V2.5,
V2.6, V3) in one wheel. V3 is selected explicitly via
`TabPFNRegressor.create_default_for_version(ModelVersion.V3, ...)`;
this is independent of the upstream default, which can change between
releases.
"""


def _import_tabpfn3_regressor():
    """Lazy import for the v3 regressor and `ModelVersion.V3` enum.

    Kept as a function (not a top-of-module import) so the package is
    importable on a machine without TabPFN installed — useful for
    smoke tests and doc builds.
    """
    try:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        import tabpfn as _tp

        version = getattr(_tp, "__version__", "unknown")
        return TabPFNRegressor, ModelVersion, version
    except ImportError as e:
        raise ImportError(
            "TabPFN is not installed. Install with `pip install 'tabpfn>=8.0,<9'` "
            "and set TABPFN_TOKEN (see README.md) before running the WACV pipeline."
        ) from e


class TabPFN3Head(nn.Module):
    """TabPFN-3 head — pure-mode only, exposes predictive uncertainty.

    One `TabPFNRegressor` per output dim. The MLP submodule is retained
    for checkpoint compatibility with v2 and as the fallback that
    `forward()` returns before `fit(...)` runs.

    Public API mirrors `tabpfn_head.TabPFNHead` so the runner doesn't
    branch on head class:

      - `forward(x)`                  -> MLP point estimate (placeholder)
      - `fit(X, y)`                   -> fits n_regressors = output_dim
      - `apply_tabpfn3(x, mlp_out)`   -> point predictions (replaces all dims)
      - `predict_with_uncertainty(x)` -> (point, std) tuple
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_estimators: int = 8,
        precision: str = "fp32",
        context_cap: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_estimators = n_estimators
        self.precision = precision
        self.context_cap = context_cap
        self.mode = "pure"  # v3 head is always pure; kept for symmetry with v2

        # MLP fallback used only by `forward()` before fit, and to keep
        # state_dict shape stable across v2/v3.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

        self.is_fitted = False
        self._regressors: Optional[list] = None
        self._tabpfn_version: Optional[str] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def apply_tabpfn3(self, x: torch.Tensor, mlp_out: torch.Tensor) -> torch.Tensor:
        """Replace every dim of `mlp_out` with the TabPFN-3 point estimate.

        Called at evaluation time. Returns a tensor on the same device
        as `mlp_out`.
        """
        point, _std = self.predict_with_uncertainty(x)
        return torch.from_numpy(point).to(mlp_out.device).to(mlp_out.dtype)

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-spot, per-dim (point, std).

        Returned as numpy arrays of shape (N, output_dim). The point
        estimate is the posterior mean; the std is computed from
        `criterion.variance(logits)` on the FullSupportBarDistribution
        returned by `predict(..., output_type="full")`. Phase 3
        (calibration) consumes this directly.
        """
        if not self.is_fitted or not self._regressors:
            raise RuntimeError(
                "TabPFN3Head.predict_with_uncertainty called before fit(); "
                "fit_tabpfn3_heads must run at the end of Stage 2."
            )

        x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
        n = x_np.shape[0]
        point = np.empty((n, self.output_dim), dtype=np.float32)
        std = np.empty((n, self.output_dim), dtype=np.float32)

        for i, r in enumerate(self._regressors):
            full = r.predict(x_np, output_type="full")
            point[:, i] = np.asarray(full["mean"], dtype=np.float32)
            variance = full["criterion"].variance(full["logits"])
            std[:, i] = (
                variance.clamp_min(0.0)
                .sqrt()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        return point, std

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit one TabPFN-3 regressor per output dim.

        Pure mode is the *only* mode for v3. The v2 refinement /
        residual logic does not apply here.
        """
        TabPFNRegressor, ModelVersion, version = _import_tabpfn3_regressor()
        self._tabpfn_version = version

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2:
            raise ValueError(f"expected 2-D y, got shape {y.shape}")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TabPFN3Head.fit() requires CUDA but torch.cuda.is_available() is False. "
                "Check CUDA driver, CUDA_VISIBLE_DEVICES, and that torch was installed "
                "with GPU support (not the CPU-only wheel)."
            )
        device = "cuda"
        print(
            f"  [TabPFN-3] loading regressors on GPU: "
            f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
        )
        k = y.shape[1]
        regressors = []
        print(
            f"  [TabPFN-3 v{version}] fitting {k} per-dim regressors "
            f"(n_estimators={self.n_estimators}, precision={self.precision}, "
            f"context_cap={self.context_cap}) on X{X.shape}"
        )
        for i in range(k):
            kwargs = dict(
                device=device,
                n_estimators=self.n_estimators,
                random_state=42 + i,
                ignore_pretraining_limits=True,
            )
            if self.precision in ("fp32", "fp16", "bf16"):
                import torch as _torch
                kwargs["inference_precision"] = {
                    "fp32": _torch.float32,
                    "fp16": _torch.float16,
                    "bf16": _torch.bfloat16,
                }[self.precision]
            r = TabPFNRegressor.create_default_for_version(
                ModelVersion.V3, **kwargs,
            )
            if self.context_cap is not None and X.shape[0] > self.context_cap:
                rng = np.random.default_rng(42 + i)
                idx = rng.choice(X.shape[0], size=self.context_cap, replace=False)
                r.fit(X[idx], y[idx, i])
            else:
                r.fit(X, y[:, i])
            regressors.append(r)
            if (i + 1) % 50 == 0:
                print(f"    [TabPFN-3] fit: {i+1}/{k} dims")

        self._regressors = regressors
        self.is_fitted = True


class PEaRLWithTabPFN3(nn.Module):
    """PEaRL stack with the TabPFN-3 head.

    Shares encoders with `baseline.PEaRL` and `tabpfn_head.PEaRLWithTabPFN`
    so all three combinations are byte-identical up to and including
    Stage-2 supervised training; only the post-stage-2 head fit and the
    eval-time application differ.
    """

    def __init__(
        self,
        n_pathways: int,
        n_genes: int,
        embed_dim: int = 256,
        pathway_hidden: int = 512,
        use_imagenet_pretrain: bool = True,
        n_estimators: int = 8,
        precision: str = "fp32",
        context_cap: Optional[int] = None,
    ):
        super().__init__()
        self.n_pathways = n_pathways
        self.n_genes = n_genes
        self.embed_dim = embed_dim

        self.pathway_encoder = PathwayEncoder(
            n_pathways=n_pathways,
            embed_dim=embed_dim,
            hidden_dim=pathway_hidden,
        )
        self.vision_encoder = VisionEncoder(
            embed_dim=embed_dim, pretrained=use_imagenet_pretrain
        )

        self.pathway_head = TabPFN3Head(
            embed_dim, n_pathways,
            n_estimators=n_estimators, precision=precision, context_cap=context_cap,
        )
        self.gene_head = TabPFN3Head(
            embed_dim, n_genes,
            n_estimators=n_estimators, precision=precision, context_cap=context_cap,
        )

    def forward_pathway_encoder(self, pathways: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.pathway_encoder(pathways, coords)

    def forward_vision_encoder(self, patches: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(patches)

    def forward_contrastive(
        self, patches: torch.Tensor, pathways: torch.Tensor, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_image = self.forward_vision_encoder(patches)
        h_pathway = self.forward_pathway_encoder(pathways, coords)
        h_image = F.normalize(h_image, p=2, dim=1)
        h_pathway = F.normalize(h_pathway, p=2, dim=1)
        return h_image, h_pathway

    def forward_supervised(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_image = self.forward_vision_encoder(patches)
        pathway_pred = self.pathway_head(h_image)
        gene_pred = self.gene_head(h_image)
        return pathway_pred, gene_pred

    def fit_tabpfn_heads(
        self, X_train: np.ndarray, y_pathway_train: np.ndarray, y_gene_train: np.ndarray
    ) -> None:
        """Same signature as `PEaRLWithTabPFN.fit_tabpfn_heads` so the runner
        can call either head class through the same fit helper."""
        self.pathway_head.fit(X_train, y_pathway_train)
        self.gene_head.fit(X_train, y_gene_train)


class SupervisedLossTabPFN3(nn.Module):
    """Identical to `SupervisedLossTabPFN` — kept as a separate symbol so
    Phase 2 (precision sweep) can swap in a precision-aware variant
    without touching the v2 loss."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pathway_pred: torch.Tensor,
        pathway_true: torch.Tensor,
        gene_pred: torch.Tensor,
        gene_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pathway_loss = self.mse(pathway_pred, pathway_true)
        gene_loss = self.mse(gene_pred, gene_true)
        return pathway_loss, gene_loss

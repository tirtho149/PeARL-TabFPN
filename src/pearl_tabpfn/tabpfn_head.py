"""PEaRL+TabPFN — the head-to-head condition of the BIBM 2026 paper.

Replaces PEaRL's MLP prediction heads with a bank of `TabPFNRegressor`
instances (one per output dim in `mode="pure"`). Shares encoders and the
contrastive loss with the MLP baseline (`baseline.py`), so the only
variable between the two conditions is the head architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from .encoders import PathwayEncoder, VisionEncoder, VitFeatureExtractor, ContrastiveLoss


class TabPFNHead(nn.Module):
    """
    Hybrid head with two prediction paths:

    - **MLP** (always present): gradient-trainable, predicts the full
      `output_dim`. Trained end-to-end during stage 2.
    - **TabPFN** (optional, fitted post-training): one `TabPFNRegressor` per
      selected output dim. The selected dims are the `tabpfn_top_k` highest
      training-target-variance columns, picked once at `fit()` time. Used
      only at inference (eval mode) — replaces the MLP prediction on those
      columns. Other columns keep the MLP value.

    Why this design: TabPFN is a 1-D-target in-context regressor; it can't be
    a multi-output gradient-trained replacement for an MLP. Treating it as a
    drop-in for the MLP head (the prior implementation) caused either silent
    fallbacks (`predict_proba` for regression, multi-output `fit` failures)
    or untrained-MLP outputs. Keeping the MLP and adding TabPFN as a per-dim
    refinement gives a fair comparison: both heads see the same training
    signal, and TabPFN gets credit on the dims it can actually model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_tabpfn: bool = True,
        tabpfn_top_k: Optional[int] = None,
        mode: str = "refinement",  # "refinement" | "residual" | "pure"
        n_estimators: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # mode='pure' replaces the MLP entirely — fit TabPFN on every output
        # dim regardless of tabpfn_top_k.
        if mode == "pure":
            self.tabpfn_top_k = output_dim
        else:
            self.tabpfn_top_k = (
                min(tabpfn_top_k, output_dim) if tabpfn_top_k else output_dim
            )
        self.mode = mode
        self.n_estimators = n_estimators

        # MLP must be created at __init__ — building it lazily inside fit()
        # left it on CPU after the parent module had been moved to CUDA.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

        self.use_tabpfn = use_tabpfn
        self.is_fitted = False
        self._regressors = None  # list[TabPFNRegressor], one per top-k dim
        self.register_buffer("_top_k_indices", torch.zeros(0, dtype=torch.long))
        # Per-dim shrinkage weight α ∈ [0,1] for residual mode. Set on a
        # holdout slice of train: α_d = 1 if TabPFN's residual on dim d
        # reduces MSE on the holdout, 0 if it makes it worse. Bounded blend
        # never makes the head worse than pure MLP on the holdout.
        self.register_buffer("_alpha", torch.zeros(0, dtype=torch.float32))

        if use_tabpfn:
            try:
                from tabpfn import TabPFNRegressor  # noqa: F401
            except ImportError:
                print("TabPFN not available, MLP-only path will be used")
                self.use_tabpfn = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always return MLP output. TabPFN refinement now happens out-of-band
        # via `apply_tabpfn(...)` (single bulk predict per dim, called once at
        # eval), not per-batch-per-dim inside forward — that scaled as
        # O(n_batches * top_k) and made eval 4 orders of magnitude slower than
        # training at full scale.
        return self.mlp(x)

    def apply_tabpfn(self, x: torch.Tensor, mlp_out: torch.Tensor) -> torch.Tensor:
        """Apply TabPFN to MLP output on the configured dims.

        - mode='refinement': overwrite MLP prediction with TabPFN prediction
          on top-k dims; other dims keep MLP value.
        - mode='residual': add α_d * TabPFN-predicted residual to MLP on top-k
          dims, where α_d ∈ [0,1] was set on a holdout slice of train. Bounded
          blend guarantees the eval prediction is no worse than the MLP on the
          holdout (alpha=0 reduces to MLP), and stacks any genuine signal
          TabPFN found (alpha=1).
        - mode='pure': overwrite *all* output dims with TabPFN predictions —
          MLP is not used at inference. This is the apple-to-apple head-to-head
          configuration: PEaRL MLP head replaced 1:1 with a TabPFN bank.
        """
        if (
            not self.use_tabpfn
            or not self.is_fitted
            or not self._regressors
        ):
            return mlp_out
        x_np = x.detach().cpu().numpy()
        out_np = mlp_out.detach().cpu().numpy().copy()
        alpha = self._alpha.detach().cpu().numpy() if self._alpha.numel() > 0 else None
        n_dims = len(self._top_k_indices.tolist())
        for i, dim_idx in enumerate(self._top_k_indices.tolist()):
            try:
                pred = self._regressors[i].predict(x_np)
                if self.mode == "residual":
                    a = float(alpha[i]) if alpha is not None else 1.0
                    out_np[:, dim_idx] = out_np[:, dim_idx] + a * pred
                else:
                    # 'refinement' and 'pure' both overwrite.
                    out_np[:, dim_idx] = pred
            except Exception as e:
                print(f"  TabPFN predict failed for dim {dim_idx}: {e}")
            if self.mode == "pure" and (i + 1) % 100 == 0:
                print(f"    TabPFN predict: {i+1}/{n_dims} dims")
        return torch.from_numpy(out_np).to(mlp_out.device)

    def fit(self, X: np.ndarray, y: np.ndarray, mlp_pred_on_X: Optional[np.ndarray] = None):
        """Fit one TabPFNRegressor per top-k output dim.

        Top-k selection criterion:
          - mode='refinement': dims with highest target variance (current behavior)
          - mode='residual':   dims with highest **MLP residual** variance — i.e.,
            where the MLP made the largest mistakes. `mlp_pred_on_X` (in-sample
            MLP predictions, shape (N, output_dim)) must be provided.

        For mode='residual', TabPFN is fit on the residual y - mlp_pred (the
        part the MLP couldn't model), not on y directly.
        """
        if not self.use_tabpfn:
            return
        try:
            from tabpfn import TabPFNRegressor

            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            if y.ndim != 2:
                raise ValueError(f"expected 2-D y, got shape {y.shape}")

            if self.mode == "pure":
                # Pure replacement: every output dim gets its own TabPFNRegressor.
                # No ranking, no MLP-residual logic.
                k = y.shape[1]
                top_k_idx = np.arange(k, dtype=np.int64)
                fit_y = y
                rank_label = "pure-all-dims"
            elif self.mode == "residual":
                if mlp_pred_on_X is None:
                    raise ValueError("mode='residual' requires mlp_pred_on_X")
                mlp_pred_on_X = np.asarray(mlp_pred_on_X, dtype=np.float32)
                residual = y - mlp_pred_on_X
                # Pick dims where MLP is *worst* (largest residual variance).
                rank_score = residual.var(axis=0)
                fit_y = residual
                k = min(self.tabpfn_top_k, y.shape[1])
                top_k_idx = np.argsort(rank_score)[-k:][::-1].astype(np.int64).copy()
                rank_label = "residual-var-ranked"
            elif mlp_pred_on_X is not None:
                # Refinement mode + MLP preds available: rank by MLP residual
                # variance (weakest MLP dims), not target variance. TabPFN
                # gets the chance to improve where the MLP struggles. Fit on
                # raw target since this is replacement, not residual.
                mlp_pred_on_X = np.asarray(mlp_pred_on_X, dtype=np.float32)
                rank_score = (y - mlp_pred_on_X).var(axis=0)
                fit_y = y
                k = min(self.tabpfn_top_k, y.shape[1])
                top_k_idx = np.argsort(rank_score)[-k:][::-1].astype(np.int64).copy()
                rank_label = "mlp-residual-var-ranked"
            else:
                rank_score = y.var(axis=0)
                fit_y = y
                k = min(self.tabpfn_top_k, y.shape[1])
                top_k_idx = np.argsort(rank_score)[-k:][::-1].astype(np.int64).copy()
                rank_label = "var-ranked"

            self._top_k_indices = torch.from_numpy(top_k_idx)

            # In residual mode, hold out a slice of train to fit α_d.
            # TabPFN is fit on the rest (inner_train); α_d is set per dim
            # by comparing holdout MSE of (mlp + tabpfn_residual) vs mlp.
            holdout_frac = 0.1 if self.mode == "residual" else 0.0
            n = X.shape[0]
            rng = np.random.default_rng(42)
            perm = rng.permutation(n)
            n_holdout = int(round(n * holdout_frac))
            holdout = perm[:n_holdout]
            inner = perm[n_holdout:]
            X_inner, X_hold = X[inner], X[holdout]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._regressors = []
            alpha = np.ones(k, dtype=np.float32)
            n_pos_alpha = 0
            print(
                f"  Fitting TabPFN ({self.mode}, n_est={self.n_estimators}) "
                f"on {k} dims (out of {y.shape[1]}; {rank_label})"
            )
            for i, dim in enumerate(top_k_idx):
                r = TabPFNRegressor(
                    device=device,
                    n_estimators=self.n_estimators,
                    random_state=42 + i,
                )
                r.fit(X_inner, fit_y[inner, dim])
                self._regressors.append(r)
                if self.mode == "residual" and n_holdout > 0:
                    # On holdout: α_d minimizes MSE of (mlp + α * pfn - y)
                    # over α ∈ ℝ via closed form, then clipped to [0,1].
                    pred_hold = r.predict(X_hold)               # (n_holdout,)
                    res_hold = y[holdout, dim] - mlp_pred_on_X[holdout, dim]
                    denom = float(np.sum(pred_hold * pred_hold) + 1e-8)
                    a_unclip = float(np.sum(pred_hold * res_hold) / denom)
                    a = max(0.0, min(1.0, a_unclip))
                    alpha[i] = a
                    if a > 0.05:
                        n_pos_alpha += 1
                # Pure mode fits 1775 regressors per fold — surface progress so
                # a long run doesn't look hung.
                if self.mode == "pure" and (i + 1) % 50 == 0:
                    print(f"    TabPFN fit: {i+1}/{k} dims")
            self._alpha = torch.from_numpy(alpha)
            self.is_fitted = True
            extra = ""
            if self.mode == "residual":
                extra = (
                    f", α∈[{alpha.min():.2f},{alpha.max():.2f}], "
                    f"mean α={alpha.mean():.2f}, {n_pos_alpha}/{k} dims kept"
                )
            print(
                f"  Fitted TabPFN ({self.mode}, n_est={self.n_estimators}) "
                f"on {k} dims (out of {y.shape[1]}; {rank_label}){extra}"
            )
        except Exception as e:
            print(f"TabPFN fitting failed ({type(e).__name__}: {e}); using MLP only")
            self.use_tabpfn = False
            self._regressors = None


class PEaRLWithTabPFN(nn.Module):
    """
    PEaRL with TabPFN Prediction Heads (Follow-up Variant)

    Replaces MLP heads with TabPFN for improved tabular prediction.
    """

    def __init__(
        self,
        n_pathways: int,
        n_genes: int,
        embed_dim: int = 256,
        pathway_hidden: int = 512,
        use_imagenet_pretrain: bool = True,
        use_tabpfn: bool = True,
        tabpfn_top_k_pathways: int = 20,
        tabpfn_top_k_genes: int = 50,
    ):
        super().__init__()
        self.n_pathways = n_pathways
        self.n_genes = n_genes
        self.embed_dim = embed_dim
        self.use_tabpfn = use_tabpfn

        # Encoders (same as baseline)
        self.pathway_encoder = PathwayEncoder(
            n_pathways=n_pathways,
            embed_dim=embed_dim,
            hidden_dim=pathway_hidden,
        )
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim, pretrained=use_imagenet_pretrain)

        # Hybrid heads (MLP for gradient training + TabPFN refinement on top-k dims).
        # When use_tabpfn=False, TabPFN is disabled at the head level and only the
        # MLP path runs.
        self.pathway_head = TabPFNHead(
            embed_dim, n_pathways, use_tabpfn=use_tabpfn, tabpfn_top_k=tabpfn_top_k_pathways
        )
        self.gene_head = TabPFNHead(
            embed_dim, n_genes, use_tabpfn=use_tabpfn, tabpfn_top_k=tabpfn_top_k_genes
        )

    def forward_pathway_encoder(self, pathways: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.pathway_encoder(pathways, coords)

    def forward_vision_encoder(self, patches: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(patches)

    def forward_contrastive(
        self, patches: torch.Tensor, pathways: torch.Tensor, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Contrastive pretraining: return normalized embeddings."""
        h_image = self.forward_vision_encoder(patches)
        h_pathway = self.forward_pathway_encoder(pathways, coords)

        # L2 normalize
        h_image = F.normalize(h_image, p=2, dim=1)
        h_pathway = F.normalize(h_pathway, p=2, dim=1)

        return h_image, h_pathway

    def forward_supervised(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supervised prediction heads."""
        h_image = self.forward_vision_encoder(patches)
        pathway_pred = self.pathway_head(h_image)
        gene_pred = self.gene_head(h_image)
        return pathway_pred, gene_pred

    def fit_tabpfn_heads(self, X_train: np.ndarray, y_pathway_train: np.ndarray, y_gene_train: np.ndarray):
        """
        Fit TabPFN heads on training data.

        Args:
            X_train: (N, embed_dim) training embeddings
            y_pathway_train: (N, n_pathways) target pathway scores
            y_gene_train: (N, n_genes) target gene expression
        """
        if self.use_tabpfn:
            if isinstance(self.pathway_head, TabPFNHead):
                self.pathway_head.fit(X_train, y_pathway_train)
            if isinstance(self.gene_head, TabPFNHead):
                self.gene_head.fit(X_train, y_gene_train)


class SupervisedLossTabPFN(nn.Module):
    """MSE loss for TabPFN heads (same as baseline)."""

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
        """
        Returns:
            pathway_loss, gene_loss
        """
        pathway_loss = self.mse(pathway_pred, pathway_true)
        gene_loss = self.mse(gene_pred, gene_true)
        return pathway_loss, gene_loss

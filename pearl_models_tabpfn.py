"""PEaRL Models with TabPFN Prediction Heads (Follow-up Variant)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from pearl_models import PathwayEncoder, VisionEncoder, VitFeatureExtractor, ContrastiveLoss


class TabPFNHead(nn.Module):
    """
    TabPFN-based prediction head for tabular data.

    Uses pretrained TabPFN model to predict from embeddings.
    """

    def __init__(self, input_dim: int, output_dim: int, use_tabpfn: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_tabpfn = use_tabpfn

        if use_tabpfn:
            try:
                from tabpfn import TabPFNClassifier
                # Initialize TabPFN for regression (will use as-is)
                self.tabpfn = TabPFNClassifier(
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    n_estimators=32,
                )
                self.is_fitted = False
            except ImportError:
                print("TabPFN not available, falling back to MLP")
                self.use_tabpfn = False
                self._init_mlp_fallback()
        else:
            self._init_mlp_fallback()

    def _init_mlp_fallback(self):
        """Fallback to simple MLP if TabPFN not available."""
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) embeddings

        Returns:
            predictions: (batch, output_dim)
        """
        if not self.use_tabpfn:
            return self.mlp(x)

        # For TabPFN, we need numpy arrays
        x_np = x.detach().cpu().numpy()

        # If TabPFN not fitted, use MLP fallback
        if not self.is_fitted:
            return self.mlp(x)

        # Use TabPFN for prediction
        try:
            predictions = self.tabpfn.predict_proba(x_np)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            return torch.from_numpy(predictions).float().to(x.device)
        except:
            # Fallback to MLP if prediction fails
            return self.mlp(x)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit TabPFN on training data."""
        if self.use_tabpfn:
            try:
                self.tabpfn.fit(X, y)
                self.is_fitted = True
            except Exception as e:
                print(f"TabPFN fitting failed: {e}, using MLP")
                self.use_tabpfn = False
                self._init_mlp_fallback()


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

        # TabPFN prediction heads (or MLP fallback)
        if use_tabpfn:
            self.pathway_head = TabPFNHead(embed_dim, n_pathways, use_tabpfn=True)
            self.gene_head = TabPFNHead(embed_dim, n_genes, use_tabpfn=True)
        else:
            # Fallback to MLP heads
            self.pathway_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, n_pathways),
            )
            self.gene_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, n_genes),
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

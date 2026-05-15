"""PEaRL+MLP baseline — the ML baseline reproduced from arXiv:2510.03455.

This is the "MLP head" condition in the BIBM 2026 head-to-head. Both heads
share `PathwayEncoder`, `VisionEncoder`, and `ContrastiveLoss` from
`encoders.py`; only the prediction heads change between this module and
`tabpfn_head.py`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .encoders import PathwayEncoder, VisionEncoder


class PEaRL(nn.Module):
    """
    PEaRL: Pathway-Enhanced Representation Learning, MLP-head variant.

    Combines pathway encoder + vision encoder + contrastive learning + two MLP
    heads. This is the baseline reproduced from arXiv:2510.03455.
    """

    def __init__(
        self,
        n_pathways: int,
        n_genes: int,
        embed_dim: int = 256,
        pathway_hidden: int = 512,
        use_imagenet_pretrain: bool = True,
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
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim, pretrained=use_imagenet_pretrain)

        # Stage-2 prediction heads — the two MLPs the paper trains.
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
        """Stage 1 — return L2-normalized image and pathway embeddings."""
        h_image = self.forward_vision_encoder(patches)
        h_pathway = self.forward_pathway_encoder(pathways, coords)
        h_image = F.normalize(h_image, p=2, dim=1)
        h_pathway = F.normalize(h_pathway, p=2, dim=1)
        return h_image, h_pathway

    def forward_supervised(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 2 — image-only inference through MLP heads."""
        h_image = self.forward_vision_encoder(patches)
        pathway_pred = self.pathway_head(h_image)
        gene_pred = self.gene_head(h_image)
        return pathway_pred, gene_pred


class SupervisedLoss(nn.Module):
    """MSE loss for gene and pathway prediction (paper's stage-2 objective)."""

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
        return self.mse(pathway_pred, pathway_true), self.mse(gene_pred, gene_true)

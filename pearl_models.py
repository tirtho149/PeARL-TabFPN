"""PEaRL Models: Pathway Encoder, Vision Encoder, Full Framework"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class VitFeatureExtractor(nn.Module):
    """Wraps torchvision ViT to extract encoder output (before classification head)."""
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        self._encoder_output = None
        self._register_hook()

    def _register_hook(self):
        """Hook into encoder output, before heads are applied."""
        def hook_fn(module, input, output):
            self._encoder_output = output
        self.vit.encoder.register_forward_hook(hook_fn)

    def forward(self, x):
        _ = self.vit(x)
        if self._encoder_output is not None:
            return self._encoder_output[:, 0]
        return None


class PathwayEncoder(nn.Module):
    """
    Transformer-based pathway encoder.

    Input: pathway scores (N, P) + spatial coordinates (N, 2)
    Output: pathway embeddings (N, embed_dim)
    """

    def __init__(
        self,
        n_pathways: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_pathways = n_pathways

        # Positional encoder for spatial coordinates
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Pathway projection
        self.pathway_proj = nn.Linear(n_pathways, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, pathways: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pathways: (batch, n_pathways)
            coords: (batch, 2) normalized [0, 1]

        Returns:
            embeddings: (batch, embed_dim)
        """
        # Encode spatial position
        pos_emb = self.pos_encoder(coords)  # (batch, hidden_dim)

        # Project pathways
        pathway_emb = self.pathway_proj(pathways)  # (batch, hidden_dim)

        # Combine with positional embedding
        combined = pathway_emb + pos_emb  # (batch, hidden_dim)

        # Transformer (add sequence dimension for transformer)
        combined = combined.unsqueeze(1)  # (batch, 1, hidden_dim)
        transformed = self.transformer(combined)  # (batch, 1, hidden_dim)
        transformed = transformed.squeeze(1)  # (batch, hidden_dim)

        # Output projection
        embeddings = self.output_proj(transformed)  # (batch, embed_dim)
        return embeddings


class VisionEncoder(nn.Module):
    """
    Vision Transformer-based image encoder using ViT-L (pretrained).

    Input: histology patches (N, 3, H, W)
    Output: image embeddings (N, embed_dim)
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_timm = False

        try:
            from timm import create_model
            self.backbone = create_model("vit_large_patch16_224", pretrained=pretrained)
            self.use_timm = True
        except:
            try:
                from torchvision.models import vit_l_16
                vit_model = vit_l_16(weights="IMAGENET1K_V1" if pretrained else None)
                self.backbone = VitFeatureExtractor(vit_model)
                self.use_timm = False
            except:
                raise ImportError("Cannot load ViT model from timm or torchvision")

        # Freeze early layers, fine-tune last 4
        for param in list(self.backbone.parameters())[:-4]:
            param.requires_grad = False

        # Projection head
        self.proj_head = nn.Linear(1024, embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch, 3, H, W)

        Returns:
            embeddings: (batch, embed_dim)
        """
        if self.use_timm:
            x = self.backbone.forward_features(patches)
            cls_token = x[:, 0]
        else:
            cls_token = self.backbone(patches)

        embeddings = self.proj_head(cls_token)
        return embeddings


class PEaRL(nn.Module):
    """
    PEaRL: Pathway-Enhanced Representation Learning

    Combines pathway encoder + vision encoder + contrastive learning + supervised heads
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

        # Encoders
        self.pathway_encoder = PathwayEncoder(
            n_pathways=n_pathways,
            embed_dim=embed_dim,
            hidden_dim=pathway_hidden,
        )
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim, pretrained=use_imagenet_pretrain)

        # Supervised prediction heads (Stage 2)
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


class ContrastiveLoss(nn.Module):
    """Symmetric contrastive loss (NT-Xent)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_i: (batch, embed_dim) normalized embeddings (e.g., image)
            h_j: (batch, embed_dim) normalized embeddings (e.g., pathway)

        Returns:
            loss: scalar
        """
        batch_size = h_i.shape[0]

        # Similarity matrix: (batch, batch)
        sim_matrix = torch.matmul(h_i, h_j.t()) / self.temperature  # (batch, batch)

        # Row-wise CE: image -> pathway
        labels = torch.arange(batch_size, device=h_i.device)
        ce_row = F.cross_entropy(sim_matrix, labels)

        # Column-wise CE: pathway -> image
        sim_matrix_t = sim_matrix.t()
        ce_col = F.cross_entropy(sim_matrix_t, labels)

        # Symmetric loss
        loss = (ce_row + ce_col) / 2
        return loss


class SupervisedLoss(nn.Module):
    """MSE loss for gene and pathway prediction."""

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

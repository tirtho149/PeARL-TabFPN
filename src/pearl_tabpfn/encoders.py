"""Shared encoders + contrastive loss for both PEaRL+MLP (baseline) and
PEaRL+TabPFN (ours).

This module is the only place encoder architectures live. Both heads
(`baseline.py` and `tabpfn_head.py`) import from here so that any change
to the encoders is automatically reflected in both head variants. This
is what makes the head-to-head comparison apples-to-apples: image and
pathway encoders are bit-for-bit identical between the two conditions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


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
        pos_emb = self.pos_encoder(coords)
        pathway_emb = self.pathway_proj(pathways)
        combined = pathway_emb + pos_emb
        combined = combined.unsqueeze(1)
        transformed = self.transformer(combined)
        transformed = transformed.squeeze(1)
        embeddings = self.output_proj(transformed)
        return embeddings


class VisionEncoder(nn.Module):
    """
    Vision Transformer image encoder.

    backbone='uni'        → MahmoodLab/UNI, the pathology foundation model the
                            paper uses (DINOv2 ViT-L/16 trained on 100k WSIs).
                            Gated on HuggingFace.
    backbone='vit_l_16'   → timm 'vit_large_patch16_224' (ImageNet pretrained)
                            with torchvision fallback.

    For apple-to-apple parity with arXiv:2510.03455, pass
    `unfreeze_last_n_blocks=4` so the last 4 transformer blocks update during
    stage-1 contrastive pretraining ("we additionally fine-tune the last 4
    layers of UNI").
    """

    def __init__(
        self,
        embed_dim: int = 256,
        pretrained: bool = True,
        backbone: str = "vit_l_16",
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone_name = backbone
        self.use_timm = False
        self.is_uni = False

        if backbone == "uni":
            from timm import create_model
            self.backbone = create_model(
                "hf-hub:MahmoodLab/UNI",
                pretrained=pretrained,
                init_values=1e-5,
                dynamic_img_size=True,
            )
            self.use_timm = True
            self.is_uni = True
        else:
            try:
                from timm import create_model
                self.backbone = create_model("vit_large_patch16_224", pretrained=pretrained)
                self.use_timm = True
            except Exception:
                from torchvision.models import vit_l_16
                vit_model = vit_l_16(weights="IMAGENET1K_V1" if pretrained else None)
                self.backbone = VitFeatureExtractor(vit_model)
                self.use_timm = False

        for p in self.backbone.parameters():
            p.requires_grad = False
        if not freeze_backbone or unfreeze_last_n_blocks > 0:
            n = unfreeze_last_n_blocks or 4
            self._unfreeze_last_n_blocks(n)
            for attr in ("norm", "norm_pre", "fc_norm"):
                mod = getattr(self.backbone, attr, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True

        self.proj_head = nn.Linear(1024, embed_dim)

    def _unfreeze_last_n_blocks(self, n: int):
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            inner = getattr(self.backbone, "vit", None)
            if inner is not None:
                enc = getattr(inner, "encoder", None)
                if enc is not None:
                    blocks = getattr(enc, "layers", None)
        if blocks is None or len(blocks) == 0:
            print(
                f"WARNING: could not locate transformer blocks on backbone "
                f"{self.backbone_name!r}; partial unfreeze skipped."
            )
            return
        k = min(n, len(blocks))
        for blk in list(blocks)[-k:]:
            for p in blk.parameters():
                p.requires_grad = True
        print(
            f"VisionEncoder: unfroze last {k} of {len(blocks)} transformer blocks "
            f"({self.backbone_name})"
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj_head(self.backbone_features(patches))

    def backbone_features(self, patches: torch.Tensor) -> torch.Tensor:
        """Return raw 1024-d backbone features (CLS token), without proj_head."""
        if self.is_uni:
            return self.backbone(patches)
        if self.use_timm:
            x = self.backbone.forward_features(patches)
            return x[:, 0]
        return self.backbone(patches)

    def head_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project precomputed (N, 1024) backbone features to embed_dim."""
        return self.proj_head(features)


class ContrastiveLoss(nn.Module):
    """Symmetric NT-Xent contrastive loss for image-pathway alignment.

    Paper (arXiv:2510.03455) says "τ > 0 is a learnable temperature". Set
    `learnable=True` to parameterize log(τ) as a learnable scalar bounded
    away from zero (matches the CLIP convention). Default keeps the legacy
    fixed-τ behavior so non-apple-to-apple runs don't change.
    """

    def __init__(self, temperature: float = 0.07, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.log_temperature = nn.Parameter(
                torch.tensor(float(np.log(temperature)), dtype=torch.float32)
            )
            self.register_buffer("_temp_min", torch.tensor(float(np.log(1e-2))))
            self.register_buffer("_temp_max", torch.tensor(float(np.log(1.0))))
        else:
            self.temperature = temperature

    def current_temperature(self) -> torch.Tensor:
        if self.learnable:
            return torch.exp(self.log_temperature.clamp(self._temp_min, self._temp_max))
        return torch.tensor(self.temperature)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        batch_size = h_i.shape[0]
        if self.learnable:
            tau = torch.exp(self.log_temperature.clamp(self._temp_min, self._temp_max))
        else:
            tau = self.temperature
        sim_matrix = torch.matmul(h_i, h_j.t()) / tau
        labels = torch.arange(batch_size, device=h_i.device)
        ce_row = F.cross_entropy(sim_matrix, labels)
        ce_col = F.cross_entropy(sim_matrix.t(), labels)
        return (ce_row + ce_col) / 2

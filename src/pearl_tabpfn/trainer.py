"""PEaRL Training Pipeline: Contrastive Pretraining + Supervised Fine-tuning"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
import os

from .baseline import PEaRL, SupervisedLoss
from .encoders import ContrastiveLoss
from .config import cfg


class Trainer:
    """PEaRL Trainer with two-stage training."""

    def __init__(
        self,
        model: PEaRL,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        self.contrastive_loss = ContrastiveLoss(temperature=cfg.TEMPERATURE)
        self.supervised_loss = SupervisedLoss()

    def stage1_contrastive_pretraining(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        output_dir: str = "./checkpoints",
    ) -> Dict[str, float]:
        """
        Stage 1: Contrastive pretraining on image-pathway pairs.

        Objective: Align image and pathway embeddings in shared latent space.
        """
        os.makedirs(output_dir, exist_ok=True)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Train
            train_loss = self._train_contrastive_epoch(train_loader, optimizer)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = self._validate_contrastive_epoch(val_loader)
            history["val_loss"].append(val_loss)

            scheduler.step()

            # Save best checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, "best_contrastive.pt"),
                )

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

        return history

    def _train_contrastive_epoch(self, loader: DataLoader, optimizer) -> float:
        """Train one epoch of contrastive pretraining."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="Training (Contrastive)", leave=False):
            patches = batch["patch"].to(self.device)
            pathways = batch["pathway"].to(self.device)
            coords = batch["coord"].to(self.device)

            optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                h_image, h_pathway = self.model.forward_contrastive(patches, pathways, coords)
                loss = self.contrastive_loss(h_image, h_pathway)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_contrastive_epoch(self, loader: DataLoader) -> float:
        """Validate one epoch of contrastive pretraining."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                patches = batch["patch"].to(self.device)
                pathways = batch["pathway"].to(self.device)
                coords = batch["coord"].to(self.device)

                h_image, h_pathway = self.model.forward_contrastive(patches, pathways, coords)
                loss = self.contrastive_loss(h_image, h_pathway)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def stage2_supervised_finetuning(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        output_dir: str = "./checkpoints",
    ) -> Dict[str, float]:
        """
        Stage 2: Supervised fine-tuning for gene and pathway prediction.

        Backbone frozen, only train prediction heads.
        """
        # Freeze backbone encoders
        for param in self.model.pathway_encoder.parameters():
            param.requires_grad = False
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            train_loss = self._train_supervised_epoch(train_loader, optimizer)
            history["train_loss"].append(train_loss)

            val_loss = self._validate_supervised_epoch(val_loader)
            history["val_loss"].append(val_loss)

            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, "best_supervised.pt"),
                )

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

        return history

    def _train_supervised_epoch(self, loader: DataLoader, optimizer) -> float:
        """Train one epoch of supervised fine-tuning."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="Training (Supervised)", leave=False):
            patches = batch["patch"].to(self.device)
            genes = batch["gene"].to(self.device)
            pathways = batch["pathway"].to(self.device)

            optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                pathway_pred, gene_pred = self.model.forward_supervised(patches)
                pathway_loss, gene_loss = self.supervised_loss(
                    pathway_pred, pathways, gene_pred, genes
                )
                loss = pathway_loss + gene_loss

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_supervised_epoch(self, loader: DataLoader) -> float:
        """Validate one epoch of supervised fine-tuning."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                patches = batch["patch"].to(self.device)
                genes = batch["gene"].to(self.device)
                pathways = batch["pathway"].to(self.device)

                pathway_pred, gene_pred = self.model.forward_supervised(patches)
                pathway_loss, gene_loss = self.supervised_loss(
                    pathway_pred, pathways, gene_pred, genes
                )
                loss = pathway_loss + gene_loss

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches


def train_pearl(
    model: PEaRL,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str = "./checkpoints",
) -> Tuple[Dict, Dict]:
    """Full two-stage training pipeline."""

    trainer = Trainer(
        model=model,
        device=device,
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        use_amp=cfg.USE_AMP,
    )

    print("=" * 80)
    print("STAGE 1: Contrastive Pretraining")
    print("=" * 80)
    history_stage1 = trainer.stage1_contrastive_pretraining(
        train_loader, val_loader, epochs=cfg.PRETRAIN_EPOCHS, output_dir=output_dir
    )

    print("\n" + "=" * 80)
    print("STAGE 2: Supervised Fine-tuning")
    print("=" * 80)
    history_stage2 = trainer.stage2_supervised_finetuning(
        train_loader, val_loader, epochs=cfg.FINETUNE_EPOCHS, output_dir=output_dir
    )

    return history_stage1, history_stage2

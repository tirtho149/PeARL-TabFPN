"""
PEaRL Baseline vs TabPFN Follow-up Comparison

Compares performance of MLP heads (baseline) vs TabPFN heads (follow-up).
"""
import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json

from pearl_config import cfg
from pearl_data import load_hest_sample, create_dataloader
from pearl_models import PEaRL, ContrastiveLoss, SupervisedLoss
from pearl_models_tabpfn import PEaRLWithTabPFN, SupervisedLossTabPFN, ContrastiveLoss as ContrastiveLossTabPFN
from pearl_eval import compute_metrics


class ComparisonTrainer:
    """Trainer for comparing baseline vs TabPFN variants."""

    def __init__(self, device: torch.device, output_dir: str = "./comparison_results"):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {
            "baseline": {},
            "tabpfn": {},
            "timestamp": datetime.now().isoformat(),
        }

    def train_baseline(
        self,
        train_loader,
        val_loader,
        epochs_stage1: int = 30,
        epochs_stage2: int = 20,
    ):
        """Train baseline PEaRL with MLP heads."""
        print("\n" + "="*80)
        print("BASELINE: MLP Prediction Heads")
        print("="*80)

        model = PEaRL(
            n_pathways=cfg.N_PATHWAYS,
            n_genes=cfg.N_GENES,
            embed_dim=cfg.EMBED_DIM,
            use_imagenet_pretrain=True,
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Stage 1: Contrastive pretraining
        print("\nStage 1: Contrastive Pretraining")
        contrastive_loss = ContrastiveLoss(temperature=cfg.TEMPERATURE)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage1)

        train_losses = []
        val_losses = []

        for epoch in range(epochs_stage1):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_stage1}", leave=False):
                patches = batch["patch"].to(self.device)
                pathways = batch["pathway"].to(self.device)
                coords = batch["coord"].to(self.device)

                h_image, h_pathway = model.forward_contrastive(patches, pathways, coords)
                loss = contrastive_loss(h_image, h_pathway)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}")

        self.results["baseline"]["contrastive_train_loss"] = train_losses

        # Stage 2: Supervised fine-tuning
        print("\nStage 2: Supervised Fine-tuning")
        supervised_loss = SupervisedLoss()
        optimizer = optim.AdamW(
            list(model.pathway_head.parameters()) + list(model.gene_head.parameters()),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage2)

        train_losses = []
        for epoch in range(epochs_stage2):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_stage2}", leave=False):
                patches = batch["patch"].to(self.device)
                pathways = batch["pathway"].to(self.device)
                genes = batch["gene"].to(self.device)

                pathway_pred, gene_pred = model.forward_supervised(patches)
                pathway_loss, gene_loss = supervised_loss(pathway_pred, pathways, gene_pred, genes)
                loss = pathway_loss + gene_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}")

        self.results["baseline"]["supervised_train_loss"] = train_losses

        return model

    def train_tabpfn(
        self,
        train_loader,
        val_loader,
        epochs_stage1: int = 30,
        epochs_stage2: int = 20,
    ):
        """Train PEaRL with TabPFN heads."""
        print("\n" + "="*80)
        print("FOLLOW-UP: TabPFN Prediction Heads")
        print("="*80)

        model = PEaRLWithTabPFN(
            n_pathways=cfg.N_PATHWAYS,
            n_genes=cfg.N_GENES,
            embed_dim=cfg.EMBED_DIM,
            use_imagenet_pretrain=True,
            use_tabpfn=True,
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Stage 1: Contrastive pretraining
        print("\nStage 1: Contrastive Pretraining")
        contrastive_loss = ContrastiveLossTabPFN(temperature=cfg.TEMPERATURE)
        optimizer = optim.AdamW(
            list(model.pathway_encoder.parameters()) + list(model.vision_encoder.parameters()),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage1)

        train_losses = []

        for epoch in range(epochs_stage1):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_stage1}", leave=False):
                patches = batch["patch"].to(self.device)
                pathways = batch["pathway"].to(self.device)
                coords = batch["coord"].to(self.device)

                h_image, h_pathway = model.forward_contrastive(patches, pathways, coords)
                loss = contrastive_loss(h_image, h_pathway)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}")

        self.results["tabpfn"]["contrastive_train_loss"] = train_losses

        # Stage 2: Supervised fine-tuning (fit TabPFN heads)
        print("\nStage 2: Supervised Fine-tuning (Fitting TabPFN Heads)")
        supervised_loss = SupervisedLossTabPFN()

        # Collect embeddings for TabPFN fitting
        model.eval()
        X_train_list = []
        y_pathway_list = []
        y_gene_list = []

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Collecting embeddings", leave=False):
                patches = batch["patch"].to(self.device)
                X_train_list.append(model.forward_vision_encoder(patches).cpu().numpy())
                y_pathway_list.append(batch["pathway"].numpy())
                y_gene_list.append(batch["gene"].numpy())

        X_train = np.concatenate(X_train_list, axis=0)
        y_pathway_train = np.concatenate(y_pathway_list, axis=0)
        y_gene_train = np.concatenate(y_gene_list, axis=0)

        # Fit TabPFN heads
        print("Fitting TabPFN pathway head...")
        model.fit_tabpfn_heads(X_train, y_pathway_train, y_gene_train)

        self.results["tabpfn"]["supervised_fitted"] = True

        return model

    def evaluate(self, model, data_loader, variant: str = "baseline"):
        """Evaluate model on data loader."""
        model.eval()
        all_pathway_pred = []
        all_pathway_true = []
        all_gene_pred = []
        all_gene_true = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {variant}", leave=False):
                patches = batch["patch"].to(self.device)
                pathways = batch["pathway"].to(self.device)
                genes = batch["gene"].to(self.device)

                pathway_pred, gene_pred = model.forward_supervised(patches)

                all_pathway_pred.append(pathway_pred.cpu().numpy())
                all_pathway_true.append(pathways.cpu().numpy())
                all_gene_pred.append(gene_pred.cpu().numpy())
                all_gene_true.append(genes.cpu().numpy())

        pathway_pred = np.concatenate(all_pathway_pred, axis=0)
        pathway_true = np.concatenate(all_pathway_true, axis=0)
        gene_pred = np.concatenate(all_gene_pred, axis=0)
        gene_true = np.concatenate(all_gene_true, axis=0)

        pathway_metrics = compute_metrics(pathway_pred, pathway_true)
        gene_metrics = compute_metrics(gene_pred, gene_true)

        return {
            "pathway": pathway_metrics,
            "gene": gene_metrics,
        }

    def compare_results(self):
        """Print comparison results."""
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)

        baseline_pathway = self.results["baseline"].get("pathway_metrics", {})
        baseline_gene = self.results["baseline"].get("gene_metrics", {})
        tabpfn_pathway = self.results["tabpfn"].get("pathway_metrics", {})
        tabpfn_gene = self.results["tabpfn"].get("gene_metrics", {})

        print("\nPATHWAY EXPRESSION PREDICTION")
        print("-" * 80)
        print(f"{'Metric':<15} {'Baseline':<20} {'TabPFN':<20} {'Improvement':<20}")
        print("-" * 80)

        for metric in ["PCC", "MSE", "MAE"]:
            baseline_val = baseline_pathway.get(metric, 0.0)
            tabpfn_val = tabpfn_pathway.get(metric, 0.0)

            if metric == "PCC":
                improvement = ((tabpfn_val - baseline_val) / abs(baseline_val) * 100) if baseline_val != 0 else 0
            else:
                improvement = ((baseline_val - tabpfn_val) / baseline_val * 100) if baseline_val != 0 else 0

            print(
                f"{metric:<15} {baseline_val:<20.4f} {tabpfn_val:<20.4f} {improvement:+.2f}%"
            )

        print("\nGENE EXPRESSION PREDICTION")
        print("-" * 80)
        print(f"{'Metric':<15} {'Baseline':<20} {'TabPFN':<20} {'Improvement':<20}")
        print("-" * 80)

        for metric in ["PCC", "MSE", "MAE"]:
            baseline_val = baseline_gene.get(metric, 0.0)
            tabpfn_val = tabpfn_gene.get(metric, 0.0)

            if metric == "PCC":
                improvement = ((tabpfn_val - baseline_val) / abs(baseline_val) * 100) if baseline_val != 0 else 0
            else:
                improvement = ((baseline_val - tabpfn_val) / baseline_val * 100) if baseline_val != 0 else 0

            print(
                f"{metric:<15} {baseline_val:<20.4f} {tabpfn_val:<20.4f} {improvement:+.2f}%"
            )

    def save_results(self):
        """Save results to JSON."""
        results_path = os.path.join(self.output_dir, "comparison_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Breast", help="Dataset name (Breast/Skin/Lymph)")
    parser.add_argument("--data-dir", type=str, default="./hest_data", help="Path to HEST data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-stage1", type=int, default=30)
    parser.add_argument("--epochs-stage2", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="./comparison_results")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    patches, genes, pathways, coords = load_hest_sample(
        hest_dir=args.data_dir,
        sample_id="TENX99",
        n_genes=cfg.N_GENES,
        n_pathways=cfg.N_PATHWAYS,
        max_spots=400,
    )

    train_loader = create_dataloader(patches, genes, pathways, coords, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(patches, genes, pathways, coords, batch_size=args.batch_size, shuffle=False)

    # Initialize comparison trainer
    trainer = ComparisonTrainer(device, args.output_dir)

    # Train baseline
    baseline_model = trainer.train_baseline(
        train_loader, val_loader, args.epochs_stage1, args.epochs_stage2
    )

    # Evaluate baseline
    baseline_results = trainer.evaluate(baseline_model, val_loader, "baseline")
    trainer.results["baseline"]["pathway_metrics"] = baseline_results["pathway"]
    trainer.results["baseline"]["gene_metrics"] = baseline_results["gene"]

    # Train TabPFN variant
    tabpfn_model = trainer.train_tabpfn(
        train_loader, val_loader, args.epochs_stage1, args.epochs_stage2
    )

    # Evaluate TabPFN variant
    tabpfn_results = trainer.evaluate(tabpfn_model, val_loader, "tabpfn")
    trainer.results["tabpfn"]["pathway_metrics"] = tabpfn_results["pathway"]
    trainer.results["tabpfn"]["gene_metrics"] = tabpfn_results["gene"]

    # Show comparison
    trainer.compare_results()

    # Save results
    trainer.save_results()


if __name__ == "__main__":
    main()

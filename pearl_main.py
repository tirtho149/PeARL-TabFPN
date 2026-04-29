"""PEaRL: Main Training & Evaluation Script"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging

from pearl_config import cfg
from pearl_data import load_hest_sample, create_dataloader
from pearl_models import PEaRL
from pearl_train import train_pearl
from pearl_eval import PEaRLEvaluator, visualize_spatial_prediction, plot_training_curves
from pearl_survival import evaluate_survival_prediction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_device():
    """Setup compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    return device


def load_dataset(dataset_name: str = "Breast", max_spots: int = 400):
    """Load one HEST-1k sample."""
    sample_id = cfg.HEST_IDS[dataset_name]

    logger.info(f"Loading {dataset_name} sample: {sample_id}")

    patches, genes, pathways, coords = load_hest_sample(
        hest_dir=cfg.HEST_DATA_ROOT,
        sample_id=sample_id,
        n_genes=cfg.N_GENES,
        n_pathways=cfg.N_PATHWAYS,
        patch_size=cfg.HEST_MODEL_PATCH,
        max_spots=max_spots,
    )

    logger.info(
        f"Loaded: patches {patches.shape}, genes {genes.shape}, "
        f"pathways {pathways.shape}, coords {coords.shape}"
    )

    return patches, genes, pathways, coords


def split_train_val(
    patches: np.ndarray,
    genes: np.ndarray,
    pathways: np.ndarray,
    coords: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """Split into train/val."""
    rng = np.random.default_rng(seed)
    n_samples = len(patches)
    n_train = int(n_samples * train_ratio)

    idx = rng.permutation(n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    return (
        patches[train_idx],
        genes[train_idx],
        pathways[train_idx],
        coords[train_idx],
        patches[val_idx],
        genes[val_idx],
        pathways[val_idx],
        coords[val_idx],
    )


def main():
    """Main training & evaluation pipeline."""

    logger.info("=" * 80)
    logger.info("PEaRL: Pathway-Enhanced Representation Learning")
    logger.info("=" * 80)

    # Setup
    device = setup_device()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Load data (use Breast dataset as example)
    try:
        patches, genes, pathways, coords = load_dataset("Breast", max_spots=cfg.HEST_MAX_SPOTS)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Using synthetic data for demonstration...")
        # Synthetic data for demo
        n_samples = 100
        patches = np.random.randn(n_samples, 3, 224, 224).astype(np.float32)
        genes = np.random.randn(n_samples, cfg.N_GENES).astype(np.float32)
        pathways = np.random.randn(n_samples, cfg.N_PATHWAYS).astype(np.float32)
        coords = np.random.rand(n_samples, 2).astype(np.float32)

    # Split train/val
    (
        patches_train,
        genes_train,
        pathways_train,
        coords_train,
        patches_val,
        genes_val,
        pathways_val,
        coords_val,
    ) = split_train_val(patches, genes, pathways, coords)

    # Create dataloaders
    train_loader = create_dataloader(
        patches_train,
        genes_train,
        pathways_train,
        coords_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = create_dataloader(
        patches_val,
        genes_val,
        pathways_val,
        coords_val,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )

    logger.info(f"Train samples: {len(patches_train)}, Val samples: {len(patches_val)}")

    # Initialize model
    logger.info("Initializing PEaRL model...")
    model = PEaRL(
        n_pathways=cfg.N_PATHWAYS,
        n_genes=cfg.N_GENES,
        embed_dim=cfg.EMBED_DIM,
        pathway_hidden=cfg.PATHWAY_HIDDEN,
        use_imagenet_pretrain=cfg.USE_IMAGENET_PRETRAIN,
    )
    model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    try:
        history_stage1, history_stage2 = train_pearl(
            model,
            train_loader,
            val_loader,
            device,
            output_dir=os.path.join(cfg.OUTPUT_DIR, "checkpoints"),
        )

        # Save training curves
        plot_training_curves(
            history_stage1,
            history_stage2,
            output_path=os.path.join(cfg.OUTPUT_DIR, "training_curves.png"),
        )
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Load best model
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "checkpoints", "best_supervised.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Loaded best model checkpoint")

    # Evaluation
    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    evaluator = PEaRLEvaluator()

    # Get predictions on val set
    all_pathway_pred = []
    all_gene_pred = []
    all_pathway_true = []
    all_gene_true = []

    with torch.no_grad():
        for batch in val_loader:
            pathway_pred, gene_pred, pathway_true, gene_true = evaluator.predict_batch(
                model,
                batch["patch"],
                batch["pathway"],
                batch["coord"],
                batch["gene"],
                device,
            )

            all_pathway_pred.append(pathway_pred)
            all_gene_pred.append(gene_pred)
            all_pathway_true.append(pathway_true)
            all_gene_true.append(gene_true)

    pathway_pred = np.concatenate(all_pathway_pred, axis=0)
    gene_pred = np.concatenate(all_gene_pred, axis=0)
    pathway_true = np.concatenate(all_pathway_true, axis=0)
    gene_true = np.concatenate(all_gene_true, axis=0)

    # Compute metrics
    metrics = evaluator.evaluate_expression_prediction(pathway_pred, pathway_true, gene_pred, gene_true)

    logger.info("\n" + "=" * 80)
    logger.info("Expression Prediction Metrics")
    logger.info("=" * 80)
    for key, val in metrics.items():
        logger.info(f"{key}: {val:.4f}")

    # Survival analysis
    logger.info("\n" + "=" * 80)
    logger.info("Survival Analysis")
    logger.info("=" * 80)

    # Simulate survival data
    from pearl_survival import simulate_survival_data

    times, events, durations = simulate_survival_data(len(pathway_pred))
    survival_metrics = evaluate_survival_prediction(pathway_pred, durations, events)

    for key, val in survival_metrics.items():
        logger.info(f"{key}: {val:.4f}")

    # Visualizations
    logger.info("Generating visualizations...")

    visualize_spatial_prediction(
        coords_val,
        pathway_pred,
        pathway_true,
        feature_names=[f"Pathway_{i}" for i in range(min(4, cfg.N_PATHWAYS))],
        output_path=os.path.join(cfg.OUTPUT_DIR, "spatial_pathway_pred.png"),
    )

    visualize_spatial_prediction(
        coords_val,
        gene_pred,
        gene_true,
        feature_names=[f"Gene_{i}" for i in range(min(4, cfg.N_GENES))],
        output_path=os.path.join(cfg.OUTPUT_DIR, "spatial_gene_pred.png"),
    )

    logger.info("=" * 80)
    logger.info("Pipeline completed!")
    logger.info(f"Outputs saved to: {cfg.OUTPUT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

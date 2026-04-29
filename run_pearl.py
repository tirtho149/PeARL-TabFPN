#!/usr/bin/env python3
"""
PEaRL Complete Pipeline Orchestrator

Run the full PEaRL pipeline:
1. Data loading & preprocessing
2. Model initialization
3. Stage 1: Contrastive pretraining
4. Stage 2: Supervised fine-tuning
5. Evaluation & metrics
6. Survival analysis
7. Figure generation
8. LaTeX paper generation

Usage:
    python run_pearl.py                    # Run with defaults
    python run_pearl.py --dataset Breast   # Specific dataset
    python run_pearl.py --help             # Show options
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

# Import PEaRL modules
from pearl_config import cfg
from pearl_data import load_hest_sample, create_dataloader
from pearl_models import PEaRL
from pearl_train import train_pearl
from pearl_eval import PEaRLEvaluator, plot_training_curves
from pearl_survival import simulate_survival_data, evaluate_survival_prediction
from pearl_figures import generate_all_figures
from pearl_paper_generator import generate_pearl_latex


# Setup logging
def setup_logging(output_dir: str):
    """Configure logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "pearl_run.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PEaRL: Pathway-Enhanced Representation Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pearl.py                    # Full pipeline with defaults
  python run_pearl.py --dataset Breast   # Train on Breast cancer
  python run_pearl.py --no-figures       # Skip figure generation
  python run_pearl.py --max-spots 200    # Use fewer samples
        """,
    )

    parser.add_argument(
        "--dataset",
        choices=["Breast", "Skin", "Lymph"],
        default="Breast",
        help="Dataset to train on",
    )

    parser.add_argument(
        "--max-spots",
        type=int,
        default=cfg.HEST_MAX_SPOTS,
        help="Maximum spots per sample",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.BATCH_SIZE,
        help="Batch size",
    )

    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )

    parser.add_argument(
        "--no-paper",
        action="store_true",
        help="Skip paper generation",
    )

    parser.add_argument(
        "--output-dir",
        default=cfg.OUTPUT_DIR,
        help="Output directory",
    )

    parser.add_argument(
        "--checkpoint",
        help="Load model from checkpoint",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_with_fallback(logger, dataset_name: str, max_spots: int):
    """Load real data or use synthetic fallback."""
    try:
        logger.info(f"Loading {dataset_name} dataset...")
        patches, genes, pathways, coords = load_hest_sample(
            hest_dir=cfg.HEST_DATA_ROOT,
            sample_id=cfg.HEST_IDS[dataset_name],
            n_genes=cfg.N_GENES,
            n_pathways=cfg.N_PATHWAYS,
            patch_size=cfg.HEST_MODEL_PATCH,
            max_spots=max_spots,
        )
        logger.info(f"✓ Loaded: {patches.shape}, {genes.shape}, {pathways.shape}")
        return patches, genes, pathways, coords

    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
        logger.info("Using synthetic data for demonstration...")

        n_samples = max_spots
        patches = np.random.randn(n_samples, 3, 224, 224).astype(np.float32) / 255.0
        genes = np.random.randn(n_samples, cfg.N_GENES).astype(np.float32)
        pathways = np.random.randn(n_samples, cfg.N_PATHWAYS).astype(np.float32)
        coords = np.random.rand(n_samples, 2).astype(np.float32)

        logger.info(f"✓ Generated synthetic: {patches.shape}, {genes.shape}, {pathways.shape}")
        return patches, genes, pathways, coords


def split_data(patches, genes, pathways, coords, train_ratio: float = 0.8, seed: int = 42):
    """Split into train/val."""
    rng = np.random.default_rng(seed)
    n = len(patches)
    idx = rng.permutation(n)
    split = int(n * train_ratio)

    train_idx, val_idx = idx[:split], idx[split:]

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
    args = parse_args()
    set_seed(args.seed)

    # Setup
    logger = setup_logging(args.output_dir)

    logger.info("=" * 80)
    logger.info("PEaRL: Pathway-Enhanced Representation Learning")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Max spots: {args.max_spots}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data loading
    patches, genes, pathways, coords = load_dataset_with_fallback(
        logger, args.dataset, args.max_spots
    )

    # Split
    (
        patches_train,
        genes_train,
        pathways_train,
        coords_train,
        patches_val,
        genes_val,
        pathways_val,
        coords_val,
    ) = split_data(patches, genes, pathways, coords)

    logger.info(f"Train: {len(patches_train)}, Val: {len(patches_val)}")

    # Dataloaders
    train_loader = create_dataloader(
        patches_train,
        genes_train,
        pathways_train,
        coords_train,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        patches_val,
        genes_val,
        pathways_val,
        coords_val,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Model
    logger.info("Initializing PEaRL model...")
    model = PEaRL(
        n_pathways=cfg.N_PATHWAYS,
        n_genes=cfg.N_GENES,
        embed_dim=cfg.EMBED_DIM,
        pathway_hidden=cfg.PATHWAY_HIDDEN,
        use_imagenet_pretrain=cfg.USE_IMAGENET_PRETRAIN,
    )

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Training
    logger.info("=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)

    try:
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        history_stage1, history_stage2 = train_pearl(
            model,
            train_loader,
            val_loader,
            device,
            output_dir=checkpoint_dir,
        )

        logger.info("✓ Training completed")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Load best model
    best_model_path = os.path.join(args.output_dir, "checkpoints", "best_supervised.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Loaded best model")

    # Save training curves
    plot_training_curves(
        history_stage1,
        history_stage2,
        output_path=os.path.join(args.output_dir, "training_curves.png"),
    )

    # Evaluation
    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    evaluator = PEaRLEvaluator()

    # Get predictions
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

    pathway_pred = np.concatenate(all_pathway_pred)
    gene_pred = np.concatenate(all_gene_pred)
    pathway_true = np.concatenate(all_pathway_true)
    gene_true = np.concatenate(all_gene_true)

    # Metrics
    metrics = evaluator.evaluate_expression_prediction(
        pathway_pred, pathway_true, gene_pred, gene_true
    )

    logger.info("Expression Prediction Metrics:")
    for key, val in metrics.items():
        logger.info(f"  {key}: {val:.4f}")

    # Survival analysis
    logger.info("\nSurvival Analysis:")
    times, events, durations = simulate_survival_data(len(pathway_pred))
    survival_metrics = evaluate_survival_prediction(pathway_pred, durations, events)

    for key, val in survival_metrics.items():
        logger.info(f"  {key}: {val:.4f}")

    # Figures
    if not args.no_figures:
        logger.info("=" * 80)
        logger.info("GENERATING FIGURES")
        logger.info("=" * 80)

        try:
            generate_all_figures(
                results_dict={"PEaRL": metrics},
                history_stage1=history_stage1,
                coords=coords_val,
                gene_pred=gene_pred,
                pathway_pred=pathway_pred,
                gene_true=gene_true,
                pathway_true=pathway_true,
                survival_cindex={"PEaRL": survival_metrics["c_index"]},
                output_dir=args.output_dir,
            )
            logger.info("✓ Figures generated")

        except Exception as e:
            logger.warning(f"Figure generation failed: {e}")

    # Paper
    if not args.no_paper:
        logger.info("=" * 80)
        logger.info("GENERATING LATEX PAPER")
        logger.info("=" * 80)

        try:
            latex_path = generate_pearl_latex(
                output_dir=args.output_dir, results=metrics
            )
            logger.info(f"✓ Paper generated: {latex_path}")
            logger.info(f"Compile with: pdflatex {latex_path}")

        except Exception as e:
            logger.warning(f"Paper generation failed: {e}")

    # Summary
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Outputs saved to: {args.output_dir}")
    logger.info("Files:")
    logger.info(f"  - Checkpoints: {os.path.join(args.output_dir, 'checkpoints')}")
    logger.info(f"  - Figures: {args.output_dir}/fig*.png")
    logger.info(f"  - Paper: {args.output_dir}/pearl_paper.tex")
    logger.info(f"  - Log: {args.output_dir}/pearl_run.log")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

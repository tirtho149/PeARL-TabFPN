"""PEaRL Configuration"""
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Config:
    # Dataset & Paths
    HEST_DATA_ROOT: str = os.environ.get("HEST_DATA_ROOT", "./hest_data")
    HF_HEST_REPO: str = "MahmoodLab/hest"
    OUTPUT_DIR: str = "./pearl_outputs"

    # Sample IDs (HEST-1k cohorts)
    HEST_IDS: Dict[str, str] = None

    # Data preprocessing
    N_GENES: int = 1000
    N_PATHWAYS: int = 200
    HEST_MODEL_PATCH: int = 224
    HEST_MAX_SPOTS: int = int(os.environ.get("HEST_MAX_SPOTS", "400"))

    # Model architecture
    EMBED_DIM: int = 256
    PATHWAY_HIDDEN: int = 512
    N_TRANSFORMER_LAYERS: int = 2
    N_ATTENTION_HEADS: int = 8

    # Training - Stage 1 (Contrastive Pretraining)
    PRETRAIN_EPOCHS: int = 30
    BATCH_SIZE: int = int(os.environ.get("PEARL_BATCH_SIZE", "8"))
    TEMPERATURE: float = 0.07
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-3

    # Training - Stage 2 (Supervised Fine-tuning)
    FINETUNE_EPOCHS: int = 20

    # Hardware
    USE_AMP: bool = os.environ.get("PEARL_NO_AMP", "").strip() not in ("1", "true", "yes")
    USE_IMAGENET_PRETRAIN: bool = os.environ.get("PEARL_SCRATCH_ENCODERS", "").strip() not in ("1", "true", "yes")

    # Dataset names & pathways per dataset
    DATASETS: List[str] = None
    DATASET_PATHWAYS: Dict[str, int] = None

    def __post_init__(self):
        if self.HEST_IDS is None:
            self.HEST_IDS = {
                "Breast": os.environ.get("HEST_ID_BREAST", "TENX99"),
                "Skin": os.environ.get("HEST_ID_SKIN", "TENX158"),
                "Lymph": os.environ.get("HEST_ID_LYMPH", "TENX143"),
            }

        if self.DATASETS is None:
            self.DATASETS = ["Breast", "Skin", "Lymph"]

        if self.DATASET_PATHWAYS is None:
            self.DATASET_PATHWAYS = {
                "Breast": 775,
                "Skin": 609,
                "Lymph": 1100,
            }

        os.makedirs(self.HEST_DATA_ROOT, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

cfg = Config()

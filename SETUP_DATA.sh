#!/bin/bash
# Download HEST-1k data from HuggingFace Hub

echo "Setting up HEST-1k dataset..."
echo "Creating data directories..."
mkdir -p hest_data/st hest_data/patches

echo "Installing huggingface-hub if needed..."
pip install huggingface-hub

echo ""
echo "Downloading HEST-1k data from HuggingFace..."
echo "This will download ~3.9GB of spatial transcriptomics data"
echo ""

# Download using huggingface_hub
python << 'PYTHON_EOF'
from huggingface_hub import snapshot_download
import os

# Download the dataset
repo_id = "HistologyBench/HEST"
local_dir = "./hest_data"

print(f"Downloading {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    resume_download=True,
    force_download=False
)
print(f"✓ Data downloaded to {local_dir}")
PYTHON_EOF

echo ""
echo "✓ HEST-1k data setup complete"
echo "Data is now available at: ./hest_data/"
echo ""
echo "You can now run:"
echo "  python run_comparison.py --dataset Breast"

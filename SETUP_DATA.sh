#!/bin/bash
# Download HEST-1k data from HuggingFace Hub.
#
# IMPORTANT: HEST-1k is GATED. You must:
#   1. Create a free HuggingFace account at https://huggingface.co
#   2. Visit https://huggingface.co/datasets/MahmoodLab/hest and click "Agree
#      and access repository"
#   3. Run `huggingface-cli login` (or `export HF_TOKEN=hf_...`) so this
#      script can authenticate.
#
# The legacy mirror `HistologyBench/HEST` referenced by older versions of
# this script no longer exists — `MahmoodLab/hest` is the canonical
# location per cfg.HF_HEST_REPO in pearl_tabpfn.config.

set -e

echo "Setting up HEST-1k dataset..."
echo "Creating data directories..."
mkdir -p hest_data/st hest_data/patches

echo "Installing huggingface-hub if needed..."
pip install huggingface-hub

# Sanity check: warn if no credentials are present before we try.
python << 'AUTH_CHECK' || exit 1
import os, sys
try:
    from huggingface_hub import get_token
    token = get_token()
except Exception:
    token = None
token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    print(
        "\nERROR: no HuggingFace token found. HEST-1k is gated; you must\n"
        "  1. Create a HF account at https://huggingface.co\n"
        "  2. Accept gating terms at https://huggingface.co/datasets/MahmoodLab/hest\n"
        "  3. Run `huggingface-cli login` and paste a token with READ access\n"
        "Aborting download.\n", file=sys.stderr,
    )
    sys.exit(1)
print(f"Found HF token (length {len(token)}); proceeding.")
AUTH_CHECK

echo ""
echo "Downloading HEST-1k data from HuggingFace (MahmoodLab/hest)..."
echo "This will download ~3.9GB of spatial transcriptomics data"
echo ""

python << 'PYTHON_EOF'
from huggingface_hub import snapshot_download

repo_id = "MahmoodLab/hest"
local_dir = "./hest_data"

print(f"Downloading {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    resume_download=True,
    force_download=False,
)
print(f"✓ Data downloaded to {local_dir}")
PYTHON_EOF

echo ""
echo "✓ HEST-1k data setup complete"
echo "Data is now available at: ./hest_data/"
echo ""
echo "Next: run \`python scripts/verify_data.py\` to confirm the data loader"
echo "produces real PCC numbers on real targets."

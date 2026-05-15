#!/usr/bin/env python
"""Entry point: PEaRL+TabPFN only (no MLP-baseline side).

Convenience wrapper around `pearl_tabpfn.reproduction.main` that forces
`--head-mode tabpfn`. Run this after `train_baseline.py` if you split
the head-to-head into two SLURM jobs. The TabPFN-pure side is the
expensive one (~45 hours on a 24 GB GPU because of the 1,775
`TabPFNRegressor` fits per fold), so it usually goes on a longer
SLURM allocation than the MLP baseline.

Typical apple-to-apple TabPFN run:

    python scripts/train_tabpfn.py --apple-to-apple \\
        --n-sections 36 --folds 5 \\
        --output-dir ./results/tabpfn
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if "--head-mode" not in sys.argv:
    sys.argv += ["--head-mode", "tabpfn"]

from pearl_tabpfn.reproduction import main

if __name__ == "__main__":
    sys.exit(main() or 0)

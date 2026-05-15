#!/usr/bin/env python
"""Entry point: PEaRL+MLP baseline only (no TabPFN side).

Convenience wrapper around `pearl_tabpfn.reproduction.main` that forces
`--head-mode mlp`. Run this when you want the baseline numbers in
isolation (e.g., to populate the `PEaRL+MLP (ours)` column of the BIBM
Table 1 without burning the ~45 hours that the TabPFN-pure side costs).

Typical apple-to-apple baseline run:

    python scripts/train_baseline.py --apple-to-apple \\
        --n-sections 36 --folds 5 \\
        --output-dir ./results/baseline
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Inject --head-mode mlp before main() reads sys.argv.
if "--head-mode" not in sys.argv:
    sys.argv += ["--head-mode", "mlp"]

from pearl_tabpfn.reproduction import main

if __name__ == "__main__":
    sys.exit(main() or 0)

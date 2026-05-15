#!/usr/bin/env python
"""Entry point: 5-fold apple-to-apple head-to-head reproduction.

Thin wrapper that calls into `pearl_tabpfn.reproduction.main`. All CLI
flags are defined there. Typical canonical run for the BIBM 2026 paper:

    python scripts/run_reproduction.py --apple-to-apple \\
        --n-sections 36 --folds 5 \\
        --epochs-stage1 100 --epochs-stage2 100 --patience 15 \\
        --batch-size 128 --encoder uni \\
        --output-dir ./results/reproduction_apple_to_apple
"""
import os
import sys

# Make `src/pearl_tabpfn` importable without `pip install -e .`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pearl_tabpfn.reproduction import main

if __name__ == "__main__":
    sys.exit(main() or 0)

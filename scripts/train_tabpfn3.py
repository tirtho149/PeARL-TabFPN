#!/usr/bin/env python
"""Entry point: PEaRL + TabPFN-3 (WACV third combination).

Convenience wrapper around `pearl_tabpfn.reproduction.main` that forces
`--head-mode tabpfn3`. See docs/WACV_PIPELINE.md for the full pipeline.

Typical apple-to-apple WACV TabPFN-3 run (Breast, 5 folds):

    python scripts/train_tabpfn3.py --apple-to-apple \\
        --cohort Breast --n-sections 36 --folds 5 \\
        --save-posteriors \\
        --output-dir ./wacv_results/phase1/Breast

For Skin / Lymph:

    python scripts/train_tabpfn3.py --apple-to-apple \\
        --cohort Skin --n-sections 24 --folds 5 --save-posteriors \\
        --output-dir ./wacv_results/phase1/Skin

For the anchor pairing (MLP + TabPFN-3 in the same fold loop, used by
Phase 1 paired significance):

    python scripts/train_tabpfn3.py --apple-to-apple \\
        --head-mode both3 --cohort Breast --n-sections 36 --folds 5 \\
        --save-posteriors --output-dir ./wacv_results/phase1/Breast_paired
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Inject the v3 head mode unless the caller overrode it. `both3` is also
# acceptable — it pairs MLP and TabPFN-3 in the same fold loop for paired
# significance, so we leave it alone when the user explicitly asks for it.
if "--head-mode" not in sys.argv:
    sys.argv += ["--head-mode", "tabpfn3"]

from pearl_tabpfn.reproduction import main

if __name__ == "__main__":
    sys.exit(main() or 0)

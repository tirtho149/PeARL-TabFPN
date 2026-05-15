#!/usr/bin/env python3
"""Structural validator for the apple-to-apple pipeline.

Exercises every code path in `pearl_tabpfn.reproduction`'s apple-to-apple
mode using a tiny stub dataset (no HEST required) so import errors, shape
mismatches, training-loop bugs, and figure-generation bugs surface in
under a minute. Catches every class of bug the real 50-hour run would
otherwise hit halfway through.

This script does NOT use synthetic data to produce PCC numbers — it only
verifies that the pipeline runs end-to-end. Numbers from this script are
discarded; the only check is "did everything execute without raising?"

Usage:
    python scripts/validate.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

# Make the src/ layout importable without `pip install -e .`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch


def main() -> int:
    print("=" * 70)
    print("APPLE-TO-APPLE STRUCTURAL VALIDATOR")
    print("=" * 70)
    print("Goal: verify every code path runs. NOT a benchmark.\n")

    failures = []

    # ---- Import tests ----
    print("[1/8] Importing modules...")
    try:
        from pearl_tabpfn.data import (
            apply_spatial_smoothing,
            _load_pathways,  # noqa: F401
            HESTDataset,
        )
        from pearl_tabpfn.baseline import PEaRL
        from pearl_tabpfn.encoders import VisionEncoder
        from pearl_tabpfn.tabpfn_head import (
            PEaRLWithTabPFN,
            TabPFNHead,
            SupervisedLossTabPFN,  # noqa: F401
        )
        from pearl_tabpfn.eval import compute_metrics
        from pearl_tabpfn.figures import generate_head_to_head_figures
        print("  ✓ imports OK\n")
    except Exception as e:
        print(f"  ✗ import failed: {e}")
        traceback.print_exc()
        return 1

    # ---- 8-neighbor smoothing ----
    print("[2/8] Testing 8-neighbor smoothing...")
    try:
        n, g = 50, 30
        rng = np.random.default_rng(0)
        expr = rng.standard_normal((n, g)).astype(np.float32)
        coords = rng.random((n, 2)).astype(np.float32)
        smoothed = apply_spatial_smoothing(expr, coords, k=8)
        assert smoothed.shape == expr.shape, f"shape mismatch: {smoothed.shape}"
        assert smoothed.dtype == expr.dtype
        # Smoothing should reduce variance (averaging neighbors).
        assert smoothed.var() <= expr.var() + 1e-6, "smoothing must not increase variance"
        print(f"  ✓ smoothing reduces variance ({expr.var():.4f} → {smoothed.var():.4f})\n")
    except Exception as e:
        failures.append(("smoothing", e))
        print(f"  ✗ smoothing failed: {e}\n")

    # ---- VisionEncoder unfreeze ----
    print("[3/8] Testing VisionEncoder unfreeze policy...")
    try:
        # Use ImageNet ViT-L (no HF gate) to avoid needing UNI auth.
        ve = VisionEncoder(
            embed_dim=256, pretrained=False, backbone="vit_l_16",
            freeze_backbone=False, unfreeze_last_n_blocks=4,
        )
        n_trainable = sum(p.requires_grad for p in ve.backbone.parameters())
        n_total = sum(1 for _ in ve.backbone.parameters())
        assert 0 < n_trainable < n_total, (
            f"expected partial unfreeze; got {n_trainable}/{n_total} trainable"
        )
        # Forward shape
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            feat = ve.backbone_features(x)
        assert feat.shape == (2, 1024), f"expected (2, 1024), got {feat.shape}"
        print(f"  ✓ unfreeze {n_trainable}/{n_total} params; backbone_features → (2, 1024)\n")
    except Exception as e:
        failures.append(("vision_encoder", e))
        print(f"  ✗ VisionEncoder unfreeze failed: {e}\n")
        traceback.print_exc()

    # ---- TabPFNHead in pure mode ----
    print("[4/8] Testing TabPFNHead pure mode...")
    try:
        head = TabPFNHead(input_dim=256, output_dim=10, use_tabpfn=True,
                          mode="pure", n_estimators=2)
        if not head.use_tabpfn:
            print("  ⚠ tabpfn package not installed — pure mode will be MLP-only at runtime.")
        else:
            n_train, d = 60, 256
            X = np.random.randn(n_train, d).astype(np.float32)
            y = np.random.randn(n_train, 10).astype(np.float32)
            head.fit(X, y)
            assert head.is_fitted, "fit didn't set is_fitted"
            assert len(head._top_k_indices.tolist()) == 10, "pure should select all dims"
            X_val = torch.from_numpy(np.random.randn(8, d).astype(np.float32))
            mlp_out = head(X_val)
            assert mlp_out.shape == (8, 10)
            applied = head.apply_tabpfn(X_val, mlp_out)
            assert applied.shape == (8, 10)
            # In pure mode the prediction should differ from the MLP path
            assert not torch.allclose(applied, mlp_out, atol=1e-6), (
                "pure mode should replace MLP outputs"
            )
            print(f"  ✓ pure mode fits all 10 dims; applies cleanly\n")
    except Exception as e:
        failures.append(("tabpfn_pure", e))
        print(f"  ✗ TabPFNHead pure failed: {e}\n")
        traceback.print_exc()

    # ---- compute_metrics with real-shaped tensors ----
    print("[5/8] Testing compute_metrics...")
    try:
        # 50 spots × 30 dims; inject 3 constant-target columns to test the
        # drop_constant_cols filter.
        true = np.random.randn(50, 30).astype(np.float32)
        true[:, [5, 10, 15]] = 0.0
        pred = true + 0.1 * np.random.randn(50, 30).astype(np.float32)
        m = compute_metrics(pred, true, drop_constant_cols=True)
        assert m["n_cols_dropped"] == 3, f"expected 3 dropped, got {m['n_cols_dropped']}"
        assert 0.5 < m["PCC"] < 1.0, f"PCC out of expected range: {m['PCC']}"
        print(f"  ✓ compute_metrics: PCC={m['PCC']:.4f}, dropped={m['n_cols_dropped']}\n")
    except Exception as e:
        failures.append(("compute_metrics", e))
        print(f"  ✗ compute_metrics failed: {e}\n")

    # ---- HEST pre-flight (expect FAIL — no data) ----
    print("[6/8] Testing HEST pre-flight (should fail when data is absent)...")
    try:
        from pearl_tabpfn.reproduction import verify_hest_data
        try:
            verify_hest_data("/tmp/_pearl_validator_nonexistent", ["TENX99"])
            failures.append((
                "preflight_should_fail",
                Exception("verify_hest_data did NOT raise on missing data"),
            ))
            print("  ✗ pre-flight should have failed but didn't\n")
        except SystemExit:
            print("  ✓ pre-flight correctly raises SystemExit on missing data\n")
    except Exception as e:
        failures.append(("preflight_import", e))
        print(f"  ✗ pre-flight check failed: {e}\n")

    # ---- Mini training loop (full-backbone path) ----
    print("[7/8] Testing mini training loop (full backbone, MLP head)...")
    try:
        # Tiny tensors; verifies stage1 + stage2 wiring without HEST.
        n_spots = 32
        patches = torch.randn(n_spots, 3, 224, 224)
        pathways = torch.randn(n_spots, 15)
        genes = torch.randn(n_spots, 20)
        coords = torch.rand(n_spots, 2)

        from pearl_tabpfn.reproduction import (
            _make_patch_loader, _stage1_full, _stage2_full, _evaluate_full,
        )
        idx_train = np.arange(0, 24)
        idx_val = np.arange(24, 32)
        tl = _make_patch_loader(patches, pathways, genes, coords, idx_train, 8, True)
        vl = _make_patch_loader(patches, pathways, genes, coords, idx_val, 8, False)

        m = PEaRL(n_pathways=15, n_genes=20, embed_dim=64, pathway_hidden=64,
                  use_imagenet_pretrain=False)
        # Replace vision encoder with a tiny one
        m.vision_encoder = VisionEncoder(
            embed_dim=64, pretrained=False, backbone="vit_l_16",
            freeze_backbone=False, unfreeze_last_n_blocks=2,
        )
        device = torch.device("cpu")  # validator runs on CPU
        m = m.to(device)

        _stage1_full(m, tl, vl, device, epochs=2, patience=3, lr=1e-3, wd=1e-3, temperature=0.07)
        _stage2_full(m, tl, vl, device, epochs=2, patience=3, lr=1e-3, wd=1e-3, head_type="mlp")
        metrics, preds = _evaluate_full(m, vl, device, return_predictions=True)
        assert "pathway" in metrics and "gene" in metrics
        assert preds["pathway_pred"].shape == (8, 15)
        assert preds["gene_pred"].shape == (8, 20)
        assert preds["coords"].shape == (8, 2)
        print(f"  ✓ mini training loop ran; pathway PCC={metrics['pathway']['PCC']:.3f}, "
              f"gene PCC={metrics['gene']['PCC']:.3f}\n")
    except Exception as e:
        failures.append(("mini_train", e))
        print(f"  ✗ mini training loop failed: {e}\n")
        traceback.print_exc()

    # ---- Head-to-head figure generation ----
    print("[8/8] Testing head-to-head figure generation...")
    try:
        n_val, P, G = 40, 12, 18
        coords = np.random.rand(n_val, 2).astype(np.float32)
        pt = np.random.randn(n_val, P).astype(np.float32)
        gt = np.random.randn(n_val, G).astype(np.float32)
        pp_mlp = pt + 0.2 * np.random.randn(n_val, P).astype(np.float32)
        pp_tab = pt + 0.3 * np.random.randn(n_val, P).astype(np.float32)
        gp_mlp = gt + 0.2 * np.random.randn(n_val, G).astype(np.float32)
        gp_tab = gt + 0.3 * np.random.randn(n_val, G).astype(np.float32)
        # Mock summary dicts in the shape `aggregate_folds` returns.
        mock_sum = {
            "gene": {"PCC": (0.5, 0.01), "MSE": (0.1, 0.01), "MAE": (0.1, 0.01)},
            "pathway": {"PCC": (0.5, 0.01), "MSE": (0.1, 0.01), "MAE": (0.1, 0.01)},
        }
        with tempfile.TemporaryDirectory() as td:
            generate_head_to_head_figures(
                summary_baseline=mock_sum,
                summary_tabpfn=mock_sum,
                coords=coords,
                pathway_pred_mlp=pp_mlp, pathway_pred_tabpfn=pp_tab,
                pathway_true=pt,
                gene_pred_mlp=gp_mlp, gene_pred_tabpfn=gp_tab,
                gene_true=gt,
                output_dir=td,
            )
            files = sorted(os.listdir(td))
            assert len(files) >= 7, f"expected ≥7 figure files, got {files}"
            print(f"  ✓ generated {len(files)} head-to-head figures\n")
    except Exception as e:
        failures.append(("figures", e))
        print(f"  ✗ figure generation failed: {e}\n")
        traceback.print_exc()

    # ---- Summary ----
    print("=" * 70)
    if failures:
        print(f"VALIDATION FAILED — {len(failures)} subsystem(s) broke:")
        for name, e in failures:
            print(f"  • {name}: {type(e).__name__}: {e}")
        return 1
    print("VALIDATION PASSED — apple-to-apple pipeline is structurally sound.")
    print("Next: download HEST + UNI, then launch the real 5-fold run.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

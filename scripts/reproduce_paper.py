#!/usr/bin/env python3
"""ONE-COMMAND WACV paper reproduction.

Drives the full pipeline end-to-end:

  Stage 1 — Pre-flight: every smoke gate must pass
  Stage 2 — Per cohort × head-mode: run 5-fold CV via scripts/run_reproduction.py
  Stage 3 — (optional) Survival arm: scripts/train_survival.py
  Stage 4 — Aggregate per-fold predictions + metrics into cohort-level results
  Stage 5 — Generate Figures 3-10 + Tables 1-3
  Stage 6 — (optional) Render paper/wacv2027/paper.tex with filled-in numbers

Each stage gates the next: if Stage 1 fails, we don't burn 50 hr in Stage 2.

Typical full run on Nova (week-long; see slurm/wacv/full_paper.sh for the
SBATCH equivalent that chains everything via --dependency=afterok):

    python scripts/reproduce_paper.py \\
        --apple-to-apple \\
        --cohorts Breast,Skin,Lymph \\
        --head-modes mlp,tabpfn3 \\
        --include-survival \\
        --wsi-dir /scratch/tcga_brca_wsi \\
        --output-dir wacv_results/paper_run_001

Iterating? Skip stages:
    --skip-smokes        # if you already gated this venv
    --skip-training      # if predictions/ already exist
    --skip-survival      # if you don't have TCGA-BRCA on disk
    --skip-figures       # if you only want metrics
    --dry-run            # print the plan, don't execute
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Make src/ importable without `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------- Stage helpers ----------
def _run(cmd: list[str], desc: str, dry_run: bool = False, log_file: Optional[Path] = None) -> int:
    """Run a subprocess and stream output. Returns its exit code."""
    print(f"\n{'='*70}")
    print(f"==> {desc}")
    print(f"    $ {' '.join(cmd)}")
    print(f"{'='*70}")
    if dry_run:
        print("    [dry-run, skipping]")
        return 0
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    bufsize=1, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            proc.wait()
            return proc.returncode
    else:
        return subprocess.call(cmd)


def stage1_smokes(dry_run: bool = False) -> bool:
    """Pre-flight gates. Every smoke must pass before training starts."""
    gates = [
        ["python", "scripts/smoke_no_data.py"],
        ["python", "scripts/smoke_tabpfn3.py"],
        ["python", "scripts/smoke_gpu.py"],
        ["python", "scripts/smoke_survival.py", "--no-wsi"],
        ["python", "scripts/validate.py"],
        ["python", "scripts/verify_data.py"],
    ]
    fails = []
    for cmd in gates:
        rc = _run(cmd, f"SMOKE: {' '.join(cmd[1:])}", dry_run=dry_run)
        if rc != 0:
            fails.append(cmd[1])
    if fails:
        print(f"\n✗ Smoke gates failed: {fails}")
        print("  Fix these before launching training (which takes days).")
        return False
    return True


def stage2_train_per_cohort(args, output_dir: Path) -> dict:
    """Per cohort × head-mode: run 5-fold CV. Returns paths to per-cohort dirs."""
    cohort_dirs = {}
    for cohort in args.cohorts:
        cohort_out = output_dir / cohort
        cohort_out.mkdir(parents=True, exist_ok=True)
        # Single subprocess call per cohort, passing all head-modes
        head_arg = "both" if set(args.head_modes) == {"mlp", "tabpfn3"} else args.head_modes[0]
        cmd = [
            "python", "scripts/run_reproduction.py",
            "--cohort", cohort,
            "--head-mode", head_arg,
            "--folds", str(args.folds),
            "--n-sections", str(args.n_sections),
            "--output-dir", str(cohort_out),
        ]
        if args.apple_to_apple:
            cmd.append("--apple-to-apple")
        log = output_dir / "logs" / f"train_{cohort}.log"
        rc = _run(cmd, f"TRAIN: {cohort} ({head_arg})", dry_run=args.dry_run, log_file=log)
        if rc != 0:
            print(f"✗ training failed for {cohort} (exit {rc}); continuing with other cohorts")
        cohort_dirs[cohort] = cohort_out
    return cohort_dirs


def stage3_survival(args, output_dir: Path) -> Optional[Path]:
    """Survival arm. Returns path to survival_results.json or None."""
    surv_dir = output_dir / "survival"
    cmd = [
        "python", "scripts/train_survival.py",
        "--wsi-dir", args.wsi_dir or "tcga_brca_wsi",
        "--clinical-tsv", args.clinical_tsv,
        "--output-dir", str(surv_dir),
        "--cache-dir", str(surv_dir / "embeddings"),
        "--encoder", "uni",
        "--epochs", "30",
        "--n-folds", "5",
    ]
    log = output_dir / "logs" / "survival.log"
    rc = _run(cmd, "SURVIVAL: TCGA-BRCA 5-fold AB-MIL + Cox", dry_run=args.dry_run, log_file=log)
    if rc != 0:
        print("✗ survival failed; Table 3 + Figure 3 right panel will be empty")
        return None
    return surv_dir / "survival_results.json"


def stage4_aggregate(cohort_dirs: dict) -> dict:
    """Load each cohort's prediction npzs + summary into one dict that
    paper_figures.generate_paper_figures consumes."""
    cohort_results = {}
    for cohort, cdir in cohort_dirs.items():
        pred_dir = Path(cdir) / "predictions"
        summary_json = Path(cdir) / "reproduction_results.json"
        if not pred_dir.is_dir() or not summary_json.exists():
            print(f"  [aggregate] {cohort}: missing predictions/ or summary — skipping")
            continue
        # Concatenate per-fold predictions
        import numpy as np
        coords_l, pp_l, pt_l, gp_l, gt_l = [], [], [], [], []
        pathway_names = None
        gene_names = None
        for npz_path in sorted(pred_dir.glob("fold_*.npz")):
            z = np.load(npz_path, allow_pickle=True)
            coords_l.append(z["coords"])
            # If the npz stored both MLP + TabPFN, prefer TabPFN for paper headlines
            pp_key = "pathway_pred_tabpfn" if "pathway_pred_tabpfn" in z.files else "pathway_pred"
            gp_key = "gene_pred_tabpfn"    if "gene_pred_tabpfn"    in z.files else "gene_pred"
            pp_l.append(z[pp_key]); pt_l.append(z["pathway_true"])
            gp_l.append(z[gp_key]); gt_l.append(z["gene_true"])
            if "pathway_names" in z.files and pathway_names is None:
                pathway_names = list(z["pathway_names"])
            if "gene_names" in z.files and gene_names is None:
                gene_names = list(z["gene_names"])
        if not coords_l:
            continue
        with open(summary_json) as f:
            summary = json.load(f)
        cohort_results[cohort] = {
            "coords":         np.concatenate(coords_l),
            "pathway_pred":   np.concatenate(pp_l),
            "pathway_true":   np.concatenate(pt_l),
            "gene_pred":      np.concatenate(gp_l),
            "gene_true":      np.concatenate(gt_l),
            "pathway_names":  pathway_names,
            "gene_names":     gene_names,
            "summary":        summary.get("summary") or summary,
        }
        print(f"  [aggregate] {cohort}: {cohort_results[cohort]['coords'].shape[0]} spots, "
              f"{cohort_results[cohort]['gene_pred'].shape[1]} genes, "
              f"{cohort_results[cohort]['pathway_pred'].shape[1]} pathways")
    return cohort_results


def stage5_figures_tables(cohort_results: dict, survival_results: Optional[dict],
                          output_dir: Path):
    """Run every figure + table function."""
    from pearl_tabpfn.paper_figures import generate_paper_figures
    fig_dir = output_dir / "paper_figures"
    return generate_paper_figures(cohort_results, str(fig_dir), survival_results)


def stage6_render_paper(output_dir: Path, manifest: dict):
    """Find paper/wacv2027/paper.tex, substitute \\TBD cells with values."""
    tex_in = REPO_ROOT / "paper" / "wacv2027" / "paper.tex"
    if not tex_in.exists():
        print(f"  [render-paper] {tex_in} not found — skipping (paper skeleton not built yet)")
        return None
    tex_text = tex_in.read_text()
    # Replace \TBD{key} placeholders with numbers from manifest
    table_1_2 = manifest.get("tables_1_2.json", {})
    table_3 = manifest.get("table_3.json", {})

    def fmt(v):
        if v is None: return r"\TBD"
        if isinstance(v, tuple): return f"{v[0]:.4f} ± {v[1]:.4f}"
        if isinstance(v, dict) and "mean" in v: return f"{v['mean']:.4f} ± {v.get('std', 0):.4f}"
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    def replace(match):
        key = match.group(1)
        # Format: \TBD{table1_gene_Breast_PCC}
        if key.startswith("table1_gene_"):
            _, _, cohort, metric = key.split("_", 3)
            return fmt((table_1_2.get("table1_gene", {}).get(cohort) or {}).get(metric))
        if key.startswith("table2_pathway_"):
            _, _, cohort, metric = key.split("_", 3)
            return fmt((table_1_2.get("table2_pathway", {}).get(cohort) or {}).get(metric))
        if key.startswith("table3_"):
            return fmt(table_3.get("PEaRL+TabPFN-v3", {}).get("c_index_mean"))
        return match.group(0)

    import re
    rendered = re.sub(r"\\TBD\{([^}]+)\}", replace, tex_text)
    tex_out = output_dir / "paper.tex"
    tex_out.write_text(rendered)
    print(f"  [render-paper] wrote {tex_out}")
    return tex_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorts", default="Breast,Skin,Lymph",
                    help="Comma-separated cohorts to run.")
    ap.add_argument("--head-modes", default="mlp,tabpfn3",
                    help="Comma-separated head modes (mlp, tabpfn3, both).")
    ap.add_argument("--apple-to-apple", action="store_true",
                    help="Bundle paper-faithful flags.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-sections", type=int, default=36)
    ap.add_argument("--include-survival", action="store_true",
                    help="Run survival arm (needs TCGA-BRCA WSIs on disk).")
    ap.add_argument("--wsi-dir", default=None, help="TCGA-BRCA WSI directory.")
    ap.add_argument("--clinical-tsv", default="tcga_smoke/brca_survival.tsv")
    ap.add_argument("--output-dir", default="wacv_results/paper_run")
    ap.add_argument("--skip-smokes", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-survival", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--skip-render-paper", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan; don't execute subprocesses.")
    args = ap.parse_args()
    args.cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    args.head_modes = [h.strip() for h in args.head_modes.split(",") if h.strip()]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    print("=" * 70)
    print(" WACV PAPER REPRODUCTION — ONE-COMMAND ORCHESTRATOR")
    print("=" * 70)
    print(f"  output_dir       : {output_dir}")
    print(f"  cohorts          : {args.cohorts}")
    print(f"  head_modes       : {args.head_modes}")
    print(f"  apple_to_apple   : {args.apple_to_apple}")
    print(f"  folds            : {args.folds}")
    print(f"  include_survival : {args.include_survival}")
    print(f"  dry_run          : {args.dry_run}")
    print("=" * 70)

    started = time.time()

    # Stage 1
    if not args.skip_smokes:
        if not stage1_smokes(dry_run=args.dry_run):
            return 1

    # Stage 2
    if args.skip_training:
        print("\n==> SKIPPING training (--skip-training) — reusing existing predictions")
        cohort_dirs = {c: output_dir / c for c in args.cohorts}
    else:
        cohort_dirs = stage2_train_per_cohort(args, output_dir)

    # Stage 3
    survival_path = None
    if args.include_survival and not args.skip_survival:
        survival_path = stage3_survival(args, output_dir)
    elif args.include_survival and args.skip_survival:
        survival_path = output_dir / "survival" / "survival_results.json"

    # Stage 4
    print(f"\n{'='*70}\n==> AGGREGATE per-fold predictions across cohorts\n{'='*70}")
    cohort_results = stage4_aggregate(cohort_dirs)
    if not cohort_results:
        print("✗ No cohort produced predictions; cannot build figures/tables.")
        return 1
    survival_results = None
    if survival_path and Path(survival_path).exists():
        with open(survival_path) as f:
            survival_results = json.load(f)

    # Stage 5
    manifest = {}
    if not args.skip_figures:
        print(f"\n{'='*70}\n==> FIGURES + TABLES\n{'='*70}")
        manifest = stage5_figures_tables(cohort_results, survival_results, output_dir)

    # Stage 6
    if not args.skip_render_paper:
        print(f"\n{'='*70}\n==> RENDER paper.tex\n{'='*70}")
        stage6_render_paper(output_dir, manifest)

    elapsed = time.time() - started
    print(f"\n{'='*70}")
    print(f" DONE in {elapsed/3600:.1f}h — outputs in {output_dir}")
    print(f" Key files:")
    print(f"   {output_dir}/paper_figures/        # Figures 3-10")
    print(f"   {output_dir}/paper_figures/tables_1_2.json")
    print(f"   {output_dir}/paper_figures/table_3.json")
    if (output_dir / 'paper.tex').exists():
        print(f"   {output_dir}/paper.tex            # filled-in manuscript")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

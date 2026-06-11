"""Aggregate all tracks × cohorts × metrics into one markdown results table.

Reads whatever has completed:
  reproduction_results_<cohort>/reproduction_results.json  (PEaRL+MLP, PEaRL+TabPFN)
  la_results_<cohort>.json                                 (LA-3B)
and writes la_regression/RESULTS.md (+ prints it). Re-run any time.

    python la_regression/aggregate_results.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pearl_tabpfn.reproduction import COHORTS  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
COHORT_LIST = ["breast", "skin", "lymph"]
METRICS = [("PCC_per_dim_mean", "PCC"), ("SCC_per_dim_mean", "SCC"),
           ("R2_per_dim_mean", "R²"), ("MSE", "MSE"), ("MAE", "MAE")]
PAPER_KEY = {"PCC_per_dim_mean": "PCC", "MSE": "MSE", "MAE": "MAE"}


def fmt(t):
    return f"{t[0]:.4f}±{t[1]:.4f}" if t and t[0] == t[0] else "—"


def method_summaries(cohort):
    out = {}
    pj = os.path.join(ROOT, f"reproduction_results_{cohort}", "reproduction_results.json")
    if os.path.isfile(pj):
        s = json.load(open(pj)).get("summary", {})
        if "baseline" in s:
            out["PEaRL+MLP"] = s["baseline"]
        if "tabpfn" in s:
            out["PEaRL+TabPFN"] = s["tabpfn"]
    la = os.path.join(HERE, f"la_results_{cohort}.json")
    if os.path.isfile(la):
        out["LA-3B"] = json.load(open(la))["summary"]
    return out


def main():
    lines = ["# Results — PEaRL reproduction + extensions (all cohorts, all metrics)\n"]
    for cohort in COHORT_LIST:
        spec = COHORTS[cohort]
        summ = method_summaries(cohort)
        lines.append(f"\n## {cohort.capitalize()} "
                     f"({spec['n_sections']} sections, {spec['n_pathways']} pathways)\n")
        if not summ:
            lines.append("_no results yet_\n")
            continue
        methods = [m for m in ("PEaRL+MLP", "PEaRL+TabPFN", "LA-3B") if m in summ]
        for tgt in ("gene", "pathway"):
            lines.append(f"\n**{tgt.capitalize()}**\n")
            head = "| Metric | " + " | ".join(methods) + " | Paper |"
            lines.append(head)
            lines.append("|" + "---|" * (len(methods) + 2))
            for key, label in METRICS:
                cells = [fmt(summ[m][tgt].get(key)) for m in methods]
                pk = PAPER_KEY.get(key)
                paper = fmt(spec["paper"][tgt][pk]) if pk else "—"
                lines.append(f"| {label} | " + " | ".join(cells) + f" | {paper} |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "RESULTS.md"), "w").write(md)
    print(md)
    print("written -> la_regression/RESULTS.md")


if __name__ == "__main__":
    main()

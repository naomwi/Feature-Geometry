#!/usr/bin/env python
"""
reproduce_all.py — One-click reproduction of all paper results.

Usage:
    python reproduce_all.py                   # Run everything
    python reproduce_all.py --table 2         # Run specific table
    python reproduce_all.py --figure 3        # Run specific figure
    python reproduce_all.py --section 4.2     # Run specific section

Requirements:
    pip install -r requirements.txt
    python scripts/download_data.py
    python scripts/download_checkpoints.py
"""
import argparse
import subprocess
import sys
import os

EXPERIMENTS = {
    # Paper Tables
    "table2": {
        "desc": "Table 2: All metrics + SC for N=11 SSL ViT-B/16 (BSDS, COCO, ADE20K)",
        "script": "experiments/run_unified_table2.py",
        "section": "4.1",
    },
    "table1": {
        "desc": "Table 1: Clustering invariance (K-Means / GMM / Spectral)",
        "script": "experiments/run_clustering_invariance.py",
        "section": "3.2",
    },

    # Paper Figures
    "figure2": {
        "desc": "Figure 2: PCA feature visualization (iBOT vs OpenCLIP vs BEiTv2)",
        "script": "experiments/generate_pca_figure.py",
        "section": "4.1",
    },
    "figure3": {
        "desc": "Figure 3: PSA vs SC scatter plot (16 backbones)",
        "script": "experiments/generate_figures.py",
        "section": "4.1",
    },
    "figure4": {
        "desc": "Figure 4: Leave-one-out stability bar chart",
        "script": "experiments/generate_loo_figure.py",
        "section": "4.2",
    },

    # Paper Sections
    "cross_dataset": {
        "desc": "Sect 4.2: Cross-dataset generalization (PSA→SC on COCO, ADE20K)",
        "script": "experiments/run_unified_ade20k.py",
        "section": "4.2",
    },
    "coco": {
        "desc": "Sect 4.2: COCO evaluation",
        "script": "experiments/run_unified_voc_coco.py",
        "section": "4.2",
    },
    "boundary": {
        "desc": "Sect 4.3: Boundary conditions (16-model analysis)",
        "script": "experiments/run_boundary.py",
        "section": "4.3",
    },
    "psa_ablation": {
        "desc": "Sect 4.4: PSA variants ablation (4/8-conn, L2, weighted)",
        "script": "experiments/run_psa_ablation.py",
        "section": "4.4",
    },
    "psa_selection": {
        "desc": "Sect 4.5: PSA-guided backbone selection",
        "script": "experiments/run_psa_selection.py",
        "section": "4.5",
    },
    "within_backbone": {
        "desc": "Sect 4.6: Within-backbone spectral analysis (n80 vs SC)",
        "script": "experiments/run_within_backbone.py",
        "section": "4.6",
    },
}

# Ordered execution for full reproduction
FULL_ORDER = [
    "table1", "table2", "cross_dataset", "coco",
    "boundary", "psa_ablation", "psa_selection", "within_backbone",
    "figure2", "figure3", "figure4",
]


def run_experiment(name, info):
    print(f"\n{'='*70}")
    print(f"  [{name}] {info['desc']}")
    print(f"  Section: {info['section']}  |  Script: {info['script']}")
    print(f"{'='*70}\n")

    script_path = os.path.join(os.path.dirname(__file__), info["script"])
    if not os.path.exists(script_path):
        print(f"  ⚠ Script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit code {result.returncode})")
        return False
    print(f"  ✓ Done")
    return True


def main():
    parser = argparse.ArgumentParser(description="Reproduce all paper results")
    parser.add_argument("--table", type=str, help="Run specific table (e.g., 1, 2)")
    parser.add_argument("--figure", type=str, help="Run specific figure (e.g., 2, 3, 4)")
    parser.add_argument("--section", type=str, help="Run specific section (e.g., 4.2)")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for name, info in EXPERIMENTS.items():
            print(f"  {name:20s} | Sect {info['section']} | {info['desc']}")
        return

    if args.table:
        key = f"table{args.table}"
        if key in EXPERIMENTS:
            run_experiment(key, EXPERIMENTS[key])
        else:
            print(f"Unknown table: {args.table}. Use --list to see options.")
        return

    if args.figure:
        key = f"figure{args.figure}"
        if key in EXPERIMENTS:
            run_experiment(key, EXPERIMENTS[key])
        else:
            print(f"Unknown figure: {args.figure}. Use --list to see options.")
        return

    if args.section:
        found = [(k, v) for k, v in EXPERIMENTS.items() if v["section"] == args.section]
        if found:
            for name, info in found:
                run_experiment(name, info)
        else:
            print(f"No experiments for section {args.section}. Use --list.")
        return

    # Run everything
    print("=" * 70)
    print("  FULL REPRODUCTION — Feature Geometry Does Not Predict SC")
    print("  Running all experiments in order...")
    print("=" * 70)

    passed, failed = 0, 0
    for name in FULL_ORDER:
        if run_experiment(name, EXPERIMENTS[name]):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {passed} passed, {failed} failed out of {len(FULL_ORDER)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

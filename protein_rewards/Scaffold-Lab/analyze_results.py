#!/usr/bin/env python3
"""Analyze ESM complete results by selecting best RMSD per backbone."""

import argparse
import pandas as pd
from pathlib import Path


def analyze_results(experiment_path: str) -> None:
    """Analyze results from esm_complete_results.csv."""
    csv_path = Path(experiment_path) / "esm_summary_results.csv"

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return

    # Load the CSV
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Unique backbones: {df['backbone_path'].nunique()}")

    # Group by backbone_path and take the row with minimum rmsd
    best_per_backbone = df.loc[df.groupby('backbone_path')['motif_rmsd'].idxmin()]

    print(f"\nSelected {len(best_per_backbone)} best samples (one per backbone)")

    # Report mean motif_rmsd and backbone_rmsd
    mean_motif_rmsd = best_per_backbone['motif_rmsd'].mean()
    mean_total_rmsd = best_per_backbone['rmsd'].mean()

    print(f"\nResults (best RMSD per backbone):")
    print(f"  Mean motif_rmsd:    {mean_motif_rmsd:.4f}")
    print(f"  Mean total_rmsd: {mean_total_rmsd:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ESM complete results")
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to experiment directory (e.g., ./outputs/2KL8/experiment_name)"
    )
    args = parser.parse_args()

    analyze_results(args.experiment_path)

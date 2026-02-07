#!/usr/bin/env python3
"""Make a box plot of scRMSD distributions grouped by protein length bins."""

from __future__ import annotations

import argparse
import os
import re
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None


LENGTH_PATTERN = re.compile(r"length_(\d+)$")
SAMPLE_PATTERN = re.compile(r"sample_\d+")


def iter_length_dirs(root: str) -> Iterable[Tuple[int, str]]:
    """Yield (length, path) pairs for every length_* directory in root."""
    for name in os.listdir(root):
        match = LENGTH_PATTERN.match(name)
        if not match:
            continue
        yield int(match.group(1)), os.path.join(root, name)


def load_sc_rmsd(sample_dir: str) -> List[float]:
    """Return scRMSD (minimum RMSD within the sample) from sc_results.csv."""
    csv_path = os.path.join(sample_dir, "sc_results.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)

    # scRMSD is defined as the minimum RMSD within a sample.
    if "rmsd" in df.columns:
        values = df["rmsd"].dropna().astype(float)
        return [values.min()] if not values.empty else []

    # Fallback: if only sc_rmsd is present, still take the minimum.
    if "sc_rmsd" in df.columns:
        values = df["sc_rmsd"].dropna().astype(float)
        return [values.min()] if not values.empty else []

    return []


def gather_records(root: str) -> pd.DataFrame:
    """Collect scRMSD values with their protein lengths."""
    records = []
    for length, length_dir in iter_length_dirs(root):
        for sample_name in sorted(os.listdir(length_dir)):
            if not SAMPLE_PATTERN.fullmatch(sample_name):
                continue
            sample_dir = os.path.join(length_dir, sample_name)
            for value in load_sc_rmsd(sample_dir):
                records.append(
                    {
                        "length": length,
                        "sample": sample_name,
                        "sc_rmsd": value,
                    }
                )
    return pd.DataFrame(records)


def add_length_bins(df: pd.DataFrame, bin_size: int) -> Tuple[pd.DataFrame, List[str]]:
    """Assign length bins and return ordered labels."""
    min_len, max_len = df["length"].min(), df["length"].max()
    start = (min_len // bin_size) * bin_size
    stop = ((max_len // bin_size) + 1) * bin_size
    bins = list(range(start, stop + bin_size, bin_size))
    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
    df = df.copy()
    df["length_bin"] = pd.cut(
        df["length"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    df["length_bin"] = pd.Categorical(df["length_bin"], categories=labels, ordered=True)
    return df, labels


def plot_boxplot(
    df: pd.DataFrame,
    labels: List[str],
    bin_size: int,
    output_path: str,
    show: bool,
) -> None:
    """Draw and save the box plot."""
    plt.figure(figsize=(max(8, len(labels) * 0.6), 6))
    if sns is not None:
        sns.boxplot(
            data=df,
            x="length_bin",
            y="sc_rmsd",
            order=labels,
            showfliers=False,
        )
    else:
        grouped = [
            (label, df.loc[df["length_bin"] == label, "sc_rmsd"].values)
            for label in labels
        ]
        grouped = [(label, values) for label, values in grouped if len(values) > 0]
        if not grouped:
            raise ValueError("No scRMSD values available for plotting.")
        labels, data = zip(*grouped)
        plt.boxplot(data, labels=labels, showfliers=False)

    plt.xlabel(f"Protein length bins (bin={bin_size})")
    plt.ylabel("scRMSD (Å)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot scRMSD distributions grouped by protein length bins from "
            "length_*/sample_* outputs (e.g., integration_100)."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to directory containing length_* subfolders (e.g., integration_100).",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=10,
        help="Width of protein length bins.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for the plot (PNG). Defaults to <input>/sc_rmsd_boxplot.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = os.path.abspath(args.input)
    output_path = args.output or os.path.join(input_dir, "sc_rmsd_boxplot.png")

    df = gather_records(input_dir)
    if df.empty:
        raise FileNotFoundError(
            f"No sc_results.csv files found under {input_dir}. "
            "Ensure the directory structure is length_*/sample_*/sc_results.csv."
        )

    df, labels = add_length_bins(df, args.bin_size)
    plot_boxplot(df, labels, args.bin_size, output_path, args.show)
    print(f"Saved box plot to {output_path}")


if __name__ == "__main__":
    main()


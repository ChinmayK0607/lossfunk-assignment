#!/usr/bin/env python3
"""
compute_passat8_accuracy_all.py
================================



Usage
-----
    # default (passat8/1000 + passat8/100)
    python compute_passat8_accuracy_all.py

    # different root folder
    python compute_passat8_accuracy_all.py --base_dir my_outputs
"""
import argparse, glob, os, pandas as pd
from typing import Tuple


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def accuracy_from_csv(path: str) -> Tuple[int, int, float]:
    """Return (# correct, # total, accuracy %) for one CSV file."""
    df = pd.read_csv(path)
    total   = len(df)
    correct = (df["verdict"].str.lower() == "correct").sum()
    return correct, total, 100.0 * correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="passat8",
        help="Root folder containing 1000/ and 100/ sub-directories.",
    )
    args = parser.parse_args()

    # collect every CSV directly under passat8/*/
    search_pattern = os.path.join(args.base_dir, "*", "*.csv")
    csv_paths = sorted(glob.glob(search_pattern))
    if not csv_paths:
        print(f"⚠️  No CSV files found under {search_pattern}")
        return

    print(f"\nFound {len(csv_paths)} CSV file(s) under {args.base_dir}/**/")
    print("-------------------------------------------------------------")
    for path in csv_paths:
        correct, total, acc = accuracy_from_csv(path)
        rel_path = os.path.relpath(path, args.base_dir)
        print(f"{rel_path:<45}  {correct:>4}/{total:<4}  ({acc:6.2f} %)")

    print("-------------------------------------------------------------")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
calc_accuracy.py

Compute accuracy from the CSV produced by the vLLM benchmark.
Usage:
    python calc_accuracy.py path_to_csv_file.csv
"""

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute accuracy from vLLM benchmark CSV.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # preferred: rely on the verdict column
    if "verdict" in df.columns:
        correct = (df["verdict"].str.lower() == "correct").sum()
    else:
        # fallback comparison if verdict column absent
        correct = (df["model_answer"] == df["correct_answer"]).sum()

    total    = len(df)
    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {accuracy:.2%}  ({correct}/{total})")

if __name__ == "__main__":
    main()

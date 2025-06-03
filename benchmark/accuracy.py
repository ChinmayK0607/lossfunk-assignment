#!/usr/bin/env python
"""
calc_accuracy.py

Compute accuracy from the CSV produced by the vLLM benchmark.
Simply edit CSV_PATH below with your file’s name.
"""

import pandas as pd

# ─── 1.  Edit this line only ──────────────────────────────────────────────
CSV_PATH = ""      # ← put your CSV file here
# ─────────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(CSV_PATH)

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

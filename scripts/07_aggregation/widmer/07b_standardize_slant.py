"""
07b_normalize_newspaper_slant.py

Normalize newspaper-year slant within Congress (Z-score).
"""

import os
import argparse
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--congress", type=int, required=True)
args = parser.parse_args()
CONGRESS = args.congress

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

IN_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "aggregated"
OUT_DIR = IN_DIR

IN_FILE = IN_DIR / f"newspaper_year_slant_congress_{CONGRESS}.parquet"
OUT_FILE = IN_DIR / f"newspaper_year_slant_congress_{CONGRESS}_z.parquet"

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print(f"Normalizing slant for Congress {CONGRESS}")
    print(f"Input: {IN_FILE}")

    df = pd.read_parquet(IN_FILE)

    required = {"slant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mean_slant = df["slant"].mean()
    std_slant = df["slant"].std(ddof=0)

    if std_slant == 0:
        raise ValueError("Standard deviation is zero; cannot normalize")

    df = df.copy()
    df["slant_z"] = (df["slant"] - mean_slant) / std_slant

    df.to_parquet(OUT_FILE)

    print(f"Saved: {OUT_FILE}")
    print(f"Mean (raw): {mean_slant:.6f}")
    print(f"SD   (raw): {std_slant:.6f}")
    print(f"Rows: {len(df):,}")

if __name__ == "__main__":
    main()

"""
07a_aggregate_newspaper_year.py

Aggregate article-level slant to newspaper-year level
using word_count-weighted averages (Widmer-style).
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

IN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "newspapers"
    / "slant"
    / f"news_slant_congress_{CONGRESS}.parquet"
)

OUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "newspapers"
    / "aggregated"
)

OUT_FILE = OUT_DIR / f"newspaper_year_slant_congress_{CONGRESS}.parquet"

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating newspaper-year slant for Congress {CONGRESS}")
    print(f"Input: {IN_FILE}")

    df = pd.read_parquet(IN_FILE)

    required_cols = {"paper", "year", "slant", "word_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # drop pathological rows
    df = df[df["word_count"] > 0].copy()

    # weighted aggregation
    agg = (
        df
        .groupby(["paper", "year"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "slant": (g["slant"] * g["word_count"]).sum() / g["word_count"].sum(),
                "total_words": g["word_count"].sum(),
                "n_articles": len(g),
            })
        )
        .reset_index(drop=True)
    )

    agg["congress"] = CONGRESS

    agg.to_parquet(OUT_FILE)

    print(f"Saved: {OUT_FILE}")
    print(f"Rows: {len(agg):,}")

if __name__ == "__main__":
    main()

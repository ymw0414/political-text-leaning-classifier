"""
FILE: 07c_aggregate_slant_panel_monthly.py

DESCRIPTION:
    - Aggregate article-level CONTINUOUS Widmer slant
      to newspaper–year–month panel
    - Keeps BOTH year and month variables
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get(
    "SHIFTING_SLANT_DIR",
    r"C:\Users\ymw04\Dropbox\shifting_slant"
))

SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant_preNAFTA"
TARGET_LIST_FILE = BASE_DIR / "data" / "temp" / "final_target_papers.csv"

OUT_DIR = BASE_DIR / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "newspaper_panel_1986_2004_monthly_continuous.csv"

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MIN_YEAR = 1986
MAX_YEAR = 2004
WEIGHT_BY_TERMS = True

NAME_FIX_MAP = {
    "Richmond Times-Dispatch (VA)": "Richmond Times-Dispatch",
    "St. Louis Post-Dispatch (MO)": "St. Louis Post-Dispatch",
    "SACRAMENTO BEE": "Sacramento Bee, The (CA)",
    "THE SAN FRANCISCO CHRONICLE": "San Francisco Chronicle (CA)",
    "THE SEATTLE TIMES": "Seattle Times, The (WA)",
    "Star Tribune: Newspaper of the Twin Cities": "Star Tribune (Minneapolis, MN)"
}

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    print(">>> Loading target newspaper list")
    target_set = set(
        pd.read_csv(TARGET_LIST_FILE)["paper"]
          .astype(str)
          .str.strip()
    )

    files = sorted(SLANT_DIR.glob("news_slant_preNAFTA_on_congress_*.parquet"))
    if not files:
        print("❌ No slant parquet files found.")
        return

    chunks = []
    print(">>> Aggregating article-level continuous slant to YEAR–MONTH panel")

    for f in tqdm(files, desc="Processing files"):

        df = pd.read_parquet(
            f,
            columns=["paper", "date", "slant", "used_terms"]
        )

        # Clean paper names
        df["paper"] = (
            df["paper"]
            .replace(NAME_FIX_MAP)
            .astype(str)
            .str.strip()
        )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Explicit time variables
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["year_month"] = df["year"] * 100 + df["month"]  # e.g. 199307

        df = df[
            df["year"].between(MIN_YEAR, MAX_YEAR) &
            df["paper"].isin(target_set)
        ].dropna(subset=["paper", "year", "month", "slant", "used_terms"])

        if df.empty:
            continue

        df["slant"] = df["slant"].astype(float)
        df["used_terms"] = df["used_terms"].astype(float)

        if WEIGHT_BY_TERMS:
            df["w_slant"] = df["slant"] * df["used_terms"]

            agg = (
                df.groupby(["paper", "year", "month", "year_month"], as_index=False)
                  .agg(
                      n_articles=("slant", "count"),
                      sum_w_slant=("w_slant", "sum"),
                      sum_weights=("used_terms", "sum")
                  )
            )

            agg["slant_mean"] = agg["sum_w_slant"] / agg["sum_weights"]

        else:
            agg = (
                df.groupby(["paper", "year", "month", "year_month"], as_index=False)
                  .agg(
                      n_articles=("slant", "count"),
                      slant_mean=("slant", "mean")
                  )
            )

        chunks.append(agg)

    if not chunks:
        print("No data after aggregation.")
        return

    panel = pd.concat(chunks, ignore_index=True)
    panel = panel.sort_values(["paper", "year", "month"])

    panel.to_csv(OUT_FILE, index=False)

    print("-" * 60)
    print("Monthly aggregation completed (continuous Widmer slant).")
    print(f"Saved to: {OUT_FILE}")
    print(f"Rows: {len(panel):,}")
    print("-" * 60)


if __name__ == "__main__":
    main()

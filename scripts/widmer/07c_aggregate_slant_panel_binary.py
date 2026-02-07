"""
FILE: 07c_aggregate_slant_panel_binary.py
DESCRIPTION:
    - Aggregate article-level binary partisan classification (rep)
      to newspaper–year panel
    - Widmer-style outcome:
        slant_it = (# Republican articles) / (# total articles)
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

OUT_FILE = OUT_DIR / "newspaper_panel_1986_2004_binary.csv"

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MIN_YEAR = 1986
MAX_YEAR = 2004

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

    # --------------------------------------------------
    # Find slant files (support both naming conventions)
    # --------------------------------------------------
    patterns = [
        "news_rep_preNAFTA_congress_*.parquet",      # binary projection (recommended)
        "news_slant_preNAFTA_on_congress_*.parquet"  # older naming
    ]

    files = []
    for pat in patterns:
        files = sorted(SLANT_DIR.glob(pat))
        if files:
            print(f">>> Found {len(files)} files using pattern: {pat}")
            break

    if not files:
        print("❌ No parquet files found in SLANT_DIR.")
        print(f"    SLANT_DIR: {SLANT_DIR}")
        print(f"    Tried patterns: {patterns}")
        return

    chunks = []
    print(">>> Aggregating article-level binary slant (rep)")

    for f in tqdm(files, desc="Processing files"):
        # Read minimal columns
        try:
            df = pd.read_parquet(f, columns=["paper", "date", "rep"])
        except Exception as e:
            print(f"\n❌ Failed reading required columns from: {f.name}")
            print("   This usually means this file was created by the old projection code")
            print("   (no 'rep' column). Re-run 06_project_widmer_binary.py.")
            print(f"   Error: {e}")
            return

        df["paper_clean"] = (
            df["paper"]
            .replace(NAME_FIX_MAP)
            .astype(str)
            .str.strip()
        )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

        df = df[
            df["year"].between(MIN_YEAR, MAX_YEAR) &
            df["paper_clean"].isin(target_set)
        ].dropna(subset=["paper_clean", "year", "rep"])

        if df.empty:
            continue

        # Ensure rep is numeric 0/1
        df["rep"] = df["rep"].astype(float)

        agg = (
            df.groupby(["paper_clean", "year"], as_index=False)
              .agg(
                  n_articles=("rep", "count"),
                  n_rep=("rep", "sum")
              )
        )
        chunks.append(agg)

    if not chunks:
        print("No data found after filtering.")
        return

    panel = pd.concat(chunks, ignore_index=True)
    panel = panel.groupby(["paper_clean", "year"], as_index=False).sum()

    panel["slant_share_rep"] = panel["n_rep"] / panel["n_articles"]

    panel = (
        panel.sort_values(["paper_clean", "year"])
             .rename(columns={"paper_clean": "paper"})
    )

    panel.to_csv(OUT_FILE, index=False)

    print("-" * 60)
    print("Aggregation completed (Widmer binary style).")
    print(f"Saved to: {OUT_FILE}")
    print(f"Rows: {len(panel):,}")
    print("-" * 60)

if __name__ == "__main__":
    main()

"""
FILE: 07c_aggregate_slant_panel_econ_fast.py
DESCRIPTION:
    - Aggregate article-level slant to newspaper–year panel
    - Keep ONLY econ-related articles (p_econ >= threshold)
    - Fast econ lookup via MultiIndex reindex (no merge)
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))

SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant_preNAFTA"
ECON_FILE = BASE_DIR / "data" / "processed" / "newspapers" / "classification" / "econ_scores_all_articles.parquet"
TARGET_LIST_FILE = BASE_DIR / "data" / "temp" / "final_target_papers.csv"

OUT_DIR = BASE_DIR / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MIN_YEAR = 1986
MAX_YEAR = 2004
ECON_THRESHOLD = 0.8

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
    print(">>> Loading target paper list")
    target_set = set(pd.read_csv(TARGET_LIST_FILE)["paper"].astype(str).str.strip())

    print(">>> Loading econ scores (minimal columns) and building fast lookup")
    econ = pd.read_parquet(ECON_FILE, columns=["paper", "date", "p_econ"])

    # Normalize types ONCE (must match slant files)
    econ["paper_clean"] = econ["paper"].replace(NAME_FIX_MAP).astype(str).str.strip()
    econ["date"] = pd.to_datetime(econ["date"], errors="coerce")
    econ["year"] = econ["date"].dt.year

    econ = econ[
        econ["year"].between(MIN_YEAR, MAX_YEAR) &
        econ["paper_clean"].isin(target_set)
    ][["paper_clean", "date", "p_econ"]].dropna(subset=["paper_clean", "date", "p_econ"])

    # If duplicates exist for same (paper_clean, date), keep max p_econ (conservative for filtering)
    econ = econ.sort_values("p_econ").drop_duplicates(["paper_clean", "date"], keep="last")

    # Build lookup Series with MultiIndex
    econ_lookup = econ.set_index(["paper_clean", "date"])["p_econ"]
    # Optional: free memory
    del econ

    files = sorted(SLANT_DIR.glob("news_slant_preNAFTA_on_congress_*.parquet"))
    chunks = []

    print(f">>> Aggregating with econ threshold >= {ECON_THRESHOLD}")
    for f in tqdm(files, desc="Processing slant files"):
        df = pd.read_parquet(f, columns=["paper", "date", "slant", "used_terms"])

        df["paper_clean"] = df["paper"].replace(NAME_FIX_MAP).astype(str).str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

        df = df[
            df["year"].between(MIN_YEAR, MAX_YEAR) &
            df["paper_clean"].isin(target_set)
        ].dropna(subset=["paper_clean", "date", "slant", "used_terms"])
        if df.empty:
            continue

        # FAST econ attach: MultiIndex reindex (no merge)
        key = pd.MultiIndex.from_arrays([df["paper_clean"].to_numpy(), df["date"].to_numpy()])
        df["p_econ"] = econ_lookup.reindex(key).to_numpy()

        df = df[df["p_econ"] >= ECON_THRESHOLD]
        if df.empty:
            continue

        df["weighted_slant"] = df["slant"].astype(float) * df["used_terms"].astype(float)

        agg = (
            df.groupby(["paper_clean", "year"], as_index=False)
              .agg(
                  n_articles=("slant", "count"),
                  sum_weighted_slant=("weighted_slant", "sum"),
                  sum_weights=("used_terms", "sum")
              )
        )
        chunks.append(agg)

    if not chunks:
        print("No data after econ filtering.")
        return

    panel = pd.concat(chunks, ignore_index=True)
    panel = panel.groupby(["paper_clean", "year"], as_index=False).sum()
    panel = panel[panel["sum_weights"] > 0]
    panel["slant_weighted"] = panel["sum_weighted_slant"] / panel["sum_weights"]
    panel = panel.sort_values(["paper_clean", "year"]).rename(columns={"paper_clean": "paper"})

    out_file = OUT_DIR / f"newspaper_panel_1986_2004_econ_p{int(ECON_THRESHOLD*100)}.csv"
    panel.to_csv(out_file, index=False)

    print("-" * 60)
    print(f"Saved: {out_file}")
    print(f"Rows: {len(panel):,}")
    print("-" * 60)

if __name__ == "__main__":
    main()

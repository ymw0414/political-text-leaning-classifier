# -------------------------------------------------------------------------
# FILE: 07a_extract_raw_candidates.py
# DESCRIPTION:
#   - Scans pre-NAFTA slant outputs for the 1992–1995 window
#   - Identifies paper names that meet the 100+ articles/year stability threshold
#   - Outputs the RAW list of names for manual/hard-coded review in 07b
# -------------------------------------------------------------------------

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
if "SHIFTING_SLANT_DIR" not in os.environ:
    BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant")
else:
    BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant_preNAFTA"
OUTPUT_FILE = BASE_DIR / "data" / "temp" / "candidate_papers_1992_1995.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REQUIRED_YEARS = {1992, 1993, 1994, 1995}
MIN_ARTICLES = 100

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    files = sorted(SLANT_DIR.glob("news_slant_preNAFTA_on_congress_*.parquet"))
    paper_year_counts = {}

    print(">>> [07a] Scanning pre-NAFTA slant data for 1992–1995 stability...")

    for f in tqdm(files, desc="Scanning Parquet Files"):
        try:
            df = pd.read_parquet(f, columns=["paper", "date"])
            df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

            # Restrict to target window
            df = df[df["year"].isin(REQUIRED_YEARS)]
            if df.empty:
                continue

            counts = df.groupby(["paper", "year"]).size()

            for (p, y), cnt in counts.items():
                paper_year_counts[(p, y)] = paper_year_counts.get((p, y), 0) + cnt

        except Exception as e:
            print(f"Error scanning {f.name}: {e}")

    # Identify stable papers
    all_papers = sorted({p for p, _ in paper_year_counts.keys()})
    stable_papers = []

    for p in all_papers:
        if all(paper_year_counts.get((p, y), 0) >= MIN_ARTICLES for y in REQUIRED_YEARS):
            stable_papers.append(p)

    # Save raw list
    pd.DataFrame({"paper": stable_papers}).to_csv(OUTPUT_FILE, index=False)

    print("-" * 60)
    print(f"Extraction complete.")
    print(f"Stable papers found: {len(stable_papers)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("-" * 60)

if __name__ == "__main__":
    main()

"""
FILE: 08d_merge_location_to_panel.py
DESCRIPTION:
    - Loads the NORMALIZED Slant Panel (from 07d).
    - Loads the County Map (from 08c).
    - Merges them to create the final analysis file with FIPS codes.
    - Output: 'newspaper_panel_with_geo.csv'
"""

import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))

# [IMPORTANT] Input: Use the NORMALIZED file from Step 07d
PANEL_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_1986_2004_normalized.csv"

# Input 2: County Map (Paper -> FIPS)
GEO_MAP_FILE = BASE_DIR / "data" / "geo" / "newspaper_county_map.csv"

# Output: Final Merged Panel for Stata
OUTPUT_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_with_geo.csv"

def main():
    print(">>> [Step 08d] Merging Normalized Slant Panel with Geographic Info...")

    # 1. Load Data
    if not PANEL_FILE.exists():
        print(f"❌ Error: Normalized file not found: {PANEL_FILE}")
        print("Please run 07d_normalize_slant.py first.")
        return
    if not GEO_MAP_FILE.exists():
        print(f"❌ Error: Geo Map file not found: {GEO_MAP_FILE}")
        return

    df_panel = pd.read_csv(PANEL_FILE)
    df_geo = pd.read_csv(GEO_MAP_FILE)

    print(f"    - Loaded Normalized Panel: {len(df_panel):,} rows")
    print(f"    - Loaded Geo Map:          {len(df_geo):,} unique papers")

    # 2. Pre-merge: Format FIPS as 5-digit string
    if 'fips' in df_geo.columns:
        df_geo['fips'] = df_geo['fips'].astype(str).str.split('.').str[0].str.zfill(5)

    # 3. Merge (Left Join)
    df_merged = pd.merge(df_panel, df_geo, on="paper", how="left")

    # 4. Verification: Check for missing FIPS
    unmatched = df_merged[df_merged['fips'].isnull()]
    if not unmatched.empty:
        print("\n" + "!" * 60)
        print("WARNING: Some papers are missing FIPS codes!")
        print(unmatched['paper'].unique()[:5])
        print("!" * 60 + "\n")
    else:
        print("✅ Success: All observations successfully matched with FIPS codes.")

    # 5. Save Final CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Ensure 'slant_normalized' is in the columns
    if 'slant_normalized' in df_merged.columns:
        print("✅ Confirmed: 'slant_normalized' column is present.")

    df_merged.to_csv(OUTPUT_FILE, index=False)

    print("-" * 60)
    print(f"Final Merged Panel Saved: {OUTPUT_FILE}")
    print(f"Total Rows: {len(df_merged):,}")
    print("-" * 60)

if __name__ == "__main__":
    main()
"""
FILE: 08d_merge_location_to_panel.py
DESCRIPTION:
    - Loads the FINAL newspaper-year slant panel (Widmer binary version).
    - Loads the County Map (paper -> FIPS).
    - Merges them to create the final analysis dataset.
    - Output: data/analysis/newspaper_panel_with_geo.csv
"""

import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get(
    "SHIFTING_SLANT_DIR",
    r"C:\Users\ymw04\Dropbox\shifting_slant"
))

# INPUT: Widmer-style binary slant panel (from 07c)
PANEL_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "newspaper_panel_1986_2004_monthly_continuous.csv"
)

# INPUT: newspaper -> county FIPS map
GEO_MAP_FILE = (
    BASE_DIR
    / "data"
    / "geo"
    / "newspaper_county_map.csv"
)

# OUTPUT
OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "newspaper_panel_with_geo.csv"
)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print(">>> [08d] Merging slant panel with geographic info")

    if not PANEL_FILE.exists():
        print(f"❌ Panel file not found: {PANEL_FILE}")
        return

    if not GEO_MAP_FILE.exists():
        print(f"❌ Geo map file not found: {GEO_MAP_FILE}")
        return

    # --------------------------------------------------
    # Load
    # --------------------------------------------------
    df_panel = pd.read_csv(PANEL_FILE)
    df_geo = pd.read_csv(GEO_MAP_FILE)

    print(f"    - Loaded panel: {len(df_panel):,} rows")
    print(f"    - Loaded geo map: {len(df_geo):,} papers")

    # --------------------------------------------------
    # Clean FIPS
    # --------------------------------------------------
    if "fips" in df_geo.columns:
        df_geo["fips"] = (
            df_geo["fips"]
            .astype(str)
            .str.split(".", expand=False)
            .str[0]
            .str.zfill(5)
        )

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------
    df_merged = df_panel.merge(df_geo, on="paper", how="left")

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    missing = df_merged[df_merged["fips"].isna()]
    if not missing.empty:
        print("\n" + "!" * 60)
        print("WARNING: Missing FIPS for some papers")
        print(missing["paper"].unique()[:10])
        print("!" * 60 + "\n")
    else:
        print("✅ All papers successfully matched with FIPS codes")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(OUTPUT_FILE, index=False)

    print("-" * 60)
    print(f"Final panel saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(df_merged):,}")
    print("-" * 60)

if __name__ == "__main__":
    main()

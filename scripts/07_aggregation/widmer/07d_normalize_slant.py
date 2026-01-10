"""
FILE: 07d_normalize_slant.py
DESCRIPTION:
    - Loads the aggregated panel data (from 07c).
    - Performs GLOBAL standardization (Z-score) on 'slant_weighted'.
    - Formula: Z = (X - Mean) / Std_Dev
    - Preserves time trends (does not normalize year-by-year).
    - Output: 'newspaper_panel_1986_2004_normalized.csv'
"""

import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
# Adjust base directory if necessary
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))

# Input: Output from 07c
INPUT_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_1986_2004.csv"

# Output: Normalized file
OUTPUT_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_1986_2004_normalized.csv"


def main():
    print("-" * 60)
    print(">>> [Step 07d] Normalizing Slant Scores...")
    print("-" * 60)

    # 1. Load Data
    if not INPUT_FILE.exists():
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        print("Please run 07c_aggregate_slant_panel.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df):,} rows from {INPUT_FILE.name}")

    # 2. Check for missing or infinite values
    initial_count = len(df)
    df = df.dropna(subset=['slant_weighted'])
    if len(df) < initial_count:
        print(f"⚠️ Dropped {initial_count - len(df)} rows with missing slant values.")

    # 3. Calculate Global Statistics
    # We use global mean/std to preserve trends over time (e.g., if everyone moves right).
    global_mean = df['slant_weighted'].mean()
    global_std = df['slant_weighted'].std()

    print(f"\n📊 Statistics before Normalization:")
    print(f"   - Mean: {global_mean:.6f}")
    print(f"   - Std : {global_std:.6f}")
    print(f"   - Min : {df['slant_weighted'].min():.6f}")
    print(f"   - Max : {df['slant_weighted'].max():.6f}")

    # 4. Apply Normalization (Standardization)
    # Z = (X - mu) / sigma
    df['slant_normalized'] = (df['slant_weighted'] - global_mean) / global_std

    # 5. Verify Results
    print(f"\n📊 Statistics after Normalization (should be ~0.0 and ~1.0):")
    print(f"   - Mean: {df['slant_normalized'].mean():.6f}")
    print(f"   - Std : {df['slant_normalized'].std():.6f}")

    # 6. Save Output
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 60)
    print(f"✅ Normalization Complete.")
    print(f"💾 Saved to: {OUTPUT_FILE}")
    print(f"NOTE: Positive values (+) indicate Republican leaning in your model.")
    print("-" * 60)


if __name__ == "__main__":
    main()
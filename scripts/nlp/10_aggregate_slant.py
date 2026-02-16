"""
10_aggregate_slant.py

Merge normalized article-level slant scores (step 09) with labels (step 04),
filter, and aggregate into a newspaper-year panel dataset.

Filtering:
  - is_news == True only (exclude death notices, classifieds, etc.)

Aggregation:
  Unconditional:
    Mean slant of ALL news articles per newspaper-year.

  Extensive margin:
    ext_nonzero = share of articles with net_slant != 0 (any partisan bigrams)
    ext_R       = share of articles with net_slant > 0 (R-leaning bigrams)
    ext_D       = share of articles with net_slant < 0 (D-leaning bigrams)

Grouping:
  newspaper x year (not congress) -- aligns with yearly economic and
  financial data for downstream regression analysis.

Inputs:
  - data/processed/newspapers/04_newspaper_labeled_cong_{cong}.parquet
  - data/processed/newspapers/09_article_slant_norm_cong_{cong}.parquet

Outputs:
  - data/processed/newspapers/10_newspaper_year_panel.parquet
"""

import gc
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Paths (from pipeline_config -- change RUN_NAME there for new runs)
# ------------------------------------------------------------------
import pipeline_config as cfg

LABEL_DIR     = cfg.NEWSPAPER_LABELS   # fixed: 04_newspaper_labeled_*.parquet
NEWSPAPER_DIR = cfg.NEWS_DIR           # run-dependent: 09_*.parquet
OUT_DIR       = cfg.NEWS_DIR

SLANT_COLS_RAW = ["right_intensity", "left_intensity", "net_slant", "politicization"]
SLANT_COLS_NORM = ["right_norm", "left_norm", "net_slant_norm", "politicization_norm"]
SLANT_COLS = SLANT_COLS_RAW + SLANT_COLS_NORM


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    congresses = cfg.get_congresses()

    print("Building newspaper-year panel ...\n")
    pipeline_start = time.time()

    chunks = []

    for cong in congresses:
        label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
        slant_path = NEWSPAPER_DIR / f"09_article_slant_norm_cong_{cong}.parquet"

        if not label_path.exists() or not slant_path.exists():
            print(f"  WARNING: Congress {cong} files not found, skipping")
            continue

        # Load and merge (row-aligned)
        meta = pd.read_parquet(label_path)
        slant = pd.read_parquet(slant_path)

        # If newspaper subsampling was used (step 07), filter meta to match
        sample_idx_path = NEWSPAPER_DIR / f"07_sample_idx_cong_{cong}.npy"
        if sample_idx_path.exists():
            idx = np.load(sample_idx_path)
            meta = meta.iloc[idx].reset_index(drop=True)

        assert len(meta) == len(slant), \
            f"Congress {cong}: meta ({len(meta)}) != slant ({len(slant)})"

        df = pd.concat([meta, slant], axis=1)
        n_total = len(df)

        # Filter: news only
        df = df[df["is_news"]].copy()
        n_news = len(df)

        # Count non-zero articles
        n_nonzero = (df["net_slant"] != 0).sum()
        pct_nonzero = n_nonzero / n_news * 100 if n_news > 0 else 0

        print(f"  Congress {cong}: {n_total:,} total -> {n_news:,} news -> "
              f"{n_nonzero:,} non-zero ({pct_nonzero:.1f}%)")

        # --- Unconditional means (all articles) ---
        agg = (
            df.groupby(["paper", "year"])
            .agg(
                n_articles=("net_slant", "count"),
                **{col: (col, "mean") for col in SLANT_COLS},
            )
            .reset_index()
        )

        # --- Extensive margins ---
        grp = df.groupby(["paper", "year"])["net_slant"]
        agg["ext_nonzero"] = grp.apply(lambda x: (x != 0).mean()).values
        agg["ext_R"] = grp.apply(lambda x: (x > 0).mean()).values
        agg["ext_D"] = grp.apply(lambda x: (x < 0).mean()).values

        chunks.append(agg)

        del meta, slant, df, agg
        gc.collect()

    # Combine all congresses
    print("\nCombining across congresses ...")
    panel = pd.concat(chunks, ignore_index=True)

    # Sort
    panel = panel.sort_values(["paper", "year"]).reset_index(drop=True)

    # Save
    out_path = OUT_DIR / "10_newspaper_year_panel.parquet"
    panel.to_parquet(out_path)

    elapsed = time.time() - pipeline_start

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Newspaper-Year Panel")
    print("=" * 72)

    n_papers = panel["paper"].nunique()
    n_years = panel["year"].nunique()
    print(f"\n  Panel dimensions: {len(panel):,} rows  "
          f"({n_papers} papers x {n_years} years)")
    print(f"  Year range: {panel['year'].min()} - {panel['year'].max()}")

    print(f"\n  Unconditional:")
    print(f"    Total news articles:   {panel['n_articles'].sum():,}")
    print(f"    Mean net_slant_norm:   {panel['net_slant_norm'].mean():.4f}")
    print(f"    Std net_slant_norm:    {panel['net_slant_norm'].std():.4f}")

    print(f"\n  Extensive margin:")
    print(f"    Mean ext_nonzero:      {panel['ext_nonzero'].mean():.4f}")
    print(f"    Mean ext_R:            {panel['ext_R'].mean():.4f}")
    print(f"    Mean ext_D:            {panel['ext_D'].mean():.4f}")

    # Per-year summary
    print(f"\n  Per-year averages:")
    yearly = panel.groupby("year").agg(
        papers=("paper", "nunique"),
        articles=("n_articles", "sum"),
        net_slant_norm=("net_slant_norm", "mean"),
        ext_nonzero=("ext_nonzero", "mean"),
        ext_R=("ext_R", "mean"),
        ext_D=("ext_D", "mean"),
    )
    for year, row in yearly.iterrows():
        print(f"    {year}: {row['papers']:>3} papers  "
              f"{row['articles']:>9,.0f} articles  "
              f"slant_norm={row['net_slant_norm']:>7.3f}  "
              f"ext_nz={row['ext_nonzero']:.3f}  "
              f"ext_R={row['ext_R']:.3f}  "
              f"ext_D={row['ext_D']:.3f}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Saved to -> {out_path}")
    print("=" * 72)

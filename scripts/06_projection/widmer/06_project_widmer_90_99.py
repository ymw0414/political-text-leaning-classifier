"""
06_project_widmer_90_99.py

Widmer-style projection on test set (1990–1999):
- slant_m = sum_b f_bm * phi_b
- f_bm = relative bigram frequency
- NO labels
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed")

X_PATH = BASE_DIR / "widmer_X_bigram_counts.npz"
META_PATH = BASE_DIR / "widmer_doc_metadata.parquet"
BETA_PATH = BASE_DIR / "widmer_bigram_beta_73_89.csv"

OUT_PATH = BASE_DIR / "widmer_slant_90_99.parquet"

# --------------------------------------------------
# Load data
# --------------------------------------------------
print("Loading data...")

X = sp.load_npz(X_PATH)
meta = pd.read_parquet(META_PATH)

beta = pd.read_csv(BETA_PATH)["beta"].to_numpy()

# --------------------------------------------------
# Restrict to TEST set: 1990–1999
# Congress mapping:
# 102–106 = 1991–1999
# --------------------------------------------------
test_mask = ((meta["congress"] >= 102) & (meta["congress"] <= 106)).to_numpy()

X_test = X[test_mask]
meta_test = meta.loc[test_mask].reset_index(drop=True)

print(f"Test documents: {X_test.shape[0]}")

# --------------------------------------------------
# Widmer slant computation
# slant_m = (x_m @ beta) / total_bigram_count_m
# --------------------------------------------------
print("Computing Widmer slant (relative frequency)...")

doc_bigram_counts = np.asarray(X_test.sum(axis=1)).ravel()

slant = np.zeros_like(doc_bigram_counts, dtype=float)
valid = doc_bigram_counts > 0

slant[valid] = (X_test[valid] @ beta) / doc_bigram_counts[valid]

# --------------------------------------------------
# Save output
# --------------------------------------------------
out = meta_test[["speech_id", "congress"]].copy()
out["slant_widmer"] = slant

out.to_parquet(OUT_PATH)

print(f"Saved Widmer slant to {OUT_PATH}")
print("Done.")

# -------------------------------------------------------------------------
# FILE: 06_project_widmer.py
# DESCRIPTION:
#   - Widmer-style article-level slant (RATIO FORM)
#   - Pre-NAFTA ruler (Congress 98–100 pooled)
#   - NO binarization
#   - NO aggregation
#   - Length-normalized at article level
# -------------------------------------------------------------------------

import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# --------------------------------------------------
# Fixed settings
# --------------------------------------------------
TARGET_CONGRESSES = range(99, 109)   # 1986–2002
TITLE_WEIGHT = 3                     # already applied upstream

# --------------------------------------------------
# Paths
# --------------------------------------------------
if "SHIFTING_SLANT_DIR" not in os.environ:
    os.environ["SHIFTING_SLANT_DIR"] = r"C:\Users\ymw04\Dropbox\shifting_slant"

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

NEWS_X_DIR    = BASE_DIR / "data" / "processed" / "newspapers" / "final"
NEWS_META_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "bigrams"

MODEL_DIR = BASE_DIR / "data" / "processed" / "models" / "widmer"
OUT_DIR   = BASE_DIR / "data" / "processed" / "newspapers" / "slant_preNAFTA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHI_PATH   = MODEL_DIR / "phi_preNAFTA_83_88.npy"
VOCAB_PATH = MODEL_DIR / "vocab_used_preNAFTA_83_88.csv"

# --------------------------------------------------
# Load fixed ruler
# --------------------------------------------------
phi = np.load(PHI_PATH).ravel()
model_vocab = pd.read_csv(VOCAB_PATH)["bigram"].astype(str).tolist()
term_to_model_idx = {t: i for i, t in enumerate(model_vocab)}

print(f"Loaded pre-NAFTA ruler | vocab size = {len(model_vocab):,}")

# --------------------------------------------------
# Projection
# --------------------------------------------------
def run_projection(congress: int):

    t0 = time.time()
    print(f"\nProjecting newspapers for Congress {congress}")

    X_PATH = NEWS_X_DIR / f"X_newspapers_congress_{congress}_titleW{TITLE_WEIGHT}.npz"
    META_PATH = NEWS_META_DIR / f"meta_newspapers_congress_{congress}.parquet"
    VOCAB_NEWS_PATH = NEWS_META_DIR / f"vocab_newspapers_congress_{congress}.csv"

    OUT_PATH = OUT_DIR / f"news_slant_preNAFTA_on_congress_{congress}.parquet"

    if not X_PATH.exists():
        print("  [SKIP] missing X")
        return

    X = sp.load_npz(X_PATH)
    meta = pd.read_parquet(META_PATH)
    vocab_news = pd.read_csv(VOCAB_NEWS_PATH)["term"].astype(str).tolist()

    cols = []
    phi_vals = []

    for j, term in enumerate(vocab_news):
        mi = term_to_model_idx.get(term)
        if mi is not None:
            cols.append(j)
            phi_vals.append(phi[mi])

    if len(cols) == 0:
        slant = np.zeros(X.shape[0])
        used_terms = np.zeros(X.shape[0])
        matched = 0
    else:
        X_sub = X[:, cols]
        phi_sub = np.asarray(phi_vals)

        numer = X_sub.dot(phi_sub)
        denom = np.asarray(X_sub.sum(axis=1)).ravel()

        slant = np.zeros_like(numer, dtype=float)
        valid = denom > 0
        slant[valid] = numer[valid] / denom[valid]

        used_terms = denom
        matched = len(cols)

    out = meta.copy()
    out["slant"] = slant
    out["used_terms"] = used_terms
    out["matched_terms"] = matched
    out["title_weight"] = TITLE_WEIGHT

    out.to_parquet(OUT_PATH, index=False)

    dt = time.time() - t0
    print(f"  Saved: {OUT_PATH.name} | N={len(out):,} | matched={matched:,} | {dt:.1f}s")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    for c in TARGET_CONGRESSES:
        run_projection(c)
    print("\n>>> All article-level slants computed (Widmer ratio form).")

if __name__ == "__main__":
    main()

"""
06_project_widmer.py

Project Widmer partisan scores onto newspaper articles
(per Congress, shared vocabulary only).
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--congress", type=int, required=True)
args = parser.parse_args()
CONGRESS = args.congress

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

NEWS_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "bigrams"
VOCAB_DIR = BASE_DIR / "data" / "processed" / "shared_vocab"
MODEL_DIR = BASE_DIR / "data" / "processed" / "models" / "widmer"
OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant"

X_NEWS = NEWS_DIR / f"X_newspapers_congress_{CONGRESS}.npz"
META_NEWS = NEWS_DIR / f"meta_newspapers_congress_{CONGRESS}.parquet"
VOCAB_NEWS = NEWS_DIR / f"vocab_newspapers_congress_{CONGRESS}.csv"

SHARED_VOCAB = VOCAB_DIR / f"vocab_shared_congress_{CONGRESS}.csv"

PHI_PATH = MODEL_DIR / f"phi_congress_{CONGRESS}.npy"
INTERCEPT_PATH = MODEL_DIR / f"intercept_congress_{CONGRESS}.npy"

OUT_SLANT = OUT_DIR / f"news_slant_congress_{CONGRESS}.parquet"

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Projecting slant for Congress {CONGRESS}")

    X = sp.load_npz(X_NEWS)
    meta = pd.read_parquet(META_NEWS)

    vocab_news = pd.read_csv(VOCAB_NEWS)["term"].tolist()
    vocab_shared = pd.read_csv(SHARED_VOCAB)["bigram"].tolist()

    phi = np.load(PHI_PATH)
    intercept = float(np.load(INTERCEPT_PATH))

    index = {v: i for i, v in enumerate(vocab_news)}
    keep_idx = [index[v] for v in vocab_shared if v in index]

    if len(keep_idx) != len(phi):
        raise RuntimeError("Shared vocab and phi dimension mismatch")

    X_shared = X[:, keep_idx]

    slant = X_shared.dot(phi) + intercept
    used_terms = np.asarray(X_shared.sum(axis=1)).ravel()

    meta = meta.copy()
    meta["slant"] = slant
    meta["used_terms"] = used_terms

    meta.to_parquet(OUT_SLANT)

    print(f"Saved: {OUT_SLANT}")
    print(f"Rows: {len(meta):,}")

if __name__ == "__main__":
    main()

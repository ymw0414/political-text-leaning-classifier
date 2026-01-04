"""
05a_intersect_vocab.py

Intersect newspaper bigram vocabulary with congressional bigram vocabulary
(per Congress). No estimation. Matrix alignment only.
"""

import os
import argparse
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
SPEECH_DIR = BASE_DIR / "data" / "processed" / "speeches" / "bigrams"
OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "aligned"

NEWS_X = NEWS_DIR / f"X_newspapers_congress_{CONGRESS}.npz"
NEWS_V = NEWS_DIR / f"vocab_newspapers_congress_{CONGRESS}.csv"
NEWS_M = NEWS_DIR / f"meta_newspapers_congress_{CONGRESS}.parquet"

SPEECH_V = SPEECH_DIR / f"vocab_congress_{CONGRESS}.csv"

OUT_X = OUT_DIR / f"X_news_aligned_congress_{CONGRESS}.npz"
OUT_V = OUT_DIR / f"vocab_aligned_congress_{CONGRESS}.csv"
OUT_M = OUT_DIR / f"meta_newspapers_congress_{CONGRESS}.parquet"

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Aligning newspaper vocab to Congress {CONGRESS}")

    X_news = sp.load_npz(NEWS_X)
    vocab_news = pd.read_csv(NEWS_V)["bigram"].tolist()
    vocab_cong = set(pd.read_csv(SPEECH_V)["bigram"].tolist())

    keep_idx = [i for i, t in enumerate(vocab_news) if t in vocab_cong]

    if len(keep_idx) == 0:
        raise RuntimeError("Empty intersection between news and congress vocab")

    X_aligned = X_news[:, keep_idx]
    vocab_aligned = [vocab_news[i] for i in keep_idx]

    sp.save_npz(OUT_X, X_aligned)
    pd.DataFrame({"bigram": vocab_aligned}).to_csv(OUT_V, index=False)
    pd.read_parquet(NEWS_M).to_parquet(OUT_M)

    print(f"Final aligned shape: {X_aligned.shape}")
    print(f"Saved X: {OUT_X}")
    print(f"Saved vocab: {OUT_V}")
    print(f"Saved meta: {OUT_M}")

if __name__ == "__main__":
    main()

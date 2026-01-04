"""
05a_intersect_vocab.py

Build Congress–Newspaper shared vocabulary (per Congress).
Filters congressional vocab using same-Congress newspaper vocab.
No model estimation.
"""

import os
import argparse
import pandas as pd
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

SPEECH_DIR = BASE_DIR / "data" / "processed" / "speeches" / "bigrams"
NEWS_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "bigrams"
OUT_DIR = BASE_DIR / "data" / "processed" / "shared_vocab"

SPEECH_V = SPEECH_DIR / f"vocab_congress_{CONGRESS}.csv"
NEWS_V = NEWS_DIR / f"vocab_newspapers_congress_{CONGRESS}.csv"

OUT_V = OUT_DIR / f"vocab_shared_congress_{CONGRESS}.csv"

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building shared vocab for Congress {CONGRESS}")

    vocab_cong = pd.read_csv(SPEECH_V)["bigram"].tolist()
    vocab_news = set(pd.read_csv(NEWS_V)["term"].tolist())

    shared = [t for t in vocab_cong if t in vocab_news]

    if len(shared) == 0:
        raise RuntimeError("Empty Congress–News vocab intersection")

    pd.DataFrame({"bigram": shared}).to_csv(OUT_V, index=False)

    print(f"Shared vocab size: {len(shared):,}")
    print(f"Saved: {OUT_V}")

if __name__ == "__main__":
    main()

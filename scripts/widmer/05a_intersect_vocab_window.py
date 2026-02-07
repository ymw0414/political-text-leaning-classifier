# -------------------------------------------------------------------------
# FILE: 05a_intersect_vocab_window.py
# DESCRIPTION:
#   - Build shared vocabulary for a pooled speech window (98–100 Congress)
#   - Intersect with newspaper vocabulary pooled over 1986–2004
# -------------------------------------------------------------------------

import os
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--speech_congresses", nargs="+", type=int, required=True)
parser.add_argument("--news_congresses", nargs="+", type=int, required=True)
args = parser.parse_args()

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

SPEECH_DIR = BASE_DIR / "data" / "processed" / "speeches" / "bigrams"
NEWS_DIR   = BASE_DIR / "data" / "processed" / "newspapers" / "bigrams"
OUT_DIR    = BASE_DIR / "data" / "processed" / "shared_vocab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():

    speech_vocab = set()
    for c in args.speech_congresses:
        v = pd.read_csv(SPEECH_DIR / f"vocab_congress_{c}.csv")["bigram"]
        speech_vocab |= set(v.tolist())

    news_vocab = set()
    for c in args.news_congresses:
        v = pd.read_csv(NEWS_DIR / f"vocab_newspapers_congress_{c}.csv")["term"]
        news_vocab |= set(v.tolist())

    shared = sorted(speech_vocab & news_vocab)

    if len(shared) == 0:
        raise RuntimeError("Empty shared vocabulary")

    out_path = OUT_DIR / "vocab_shared_speech_83_88__news_86_04.csv"
    pd.DataFrame({"bigram": shared}).to_csv(out_path, index=False)

    print(f"Shared vocab size: {len(shared):,}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

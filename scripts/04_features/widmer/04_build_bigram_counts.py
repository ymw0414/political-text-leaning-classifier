"""
04_build_bigram_counts.py

Widmer-style feature construction:
- raw contiguous bigram counts
- NO spell correction
- NO lemmatization / stemming
- NO TF-IDF
- OCR noise handled only by rule-based dropping + rare bigram trimming
- Training corpus restricted to 1973+ (93rd Congress and later)
"""

import re
from pathlib import Path
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed")

INPUT_PATH = BASE_DIR / "speeches_with_party.parquet"
OUT_X_PATH = BASE_DIR / "widmer_X_bigram_counts.npz"
OUT_VOCAB_PATH = BASE_DIR / "widmer_bigram_vocab.csv"
OUT_META_PATH = BASE_DIR / "widmer_doc_metadata.parquet"

# -------------------------------------------------------------------
# Column names
# -------------------------------------------------------------------
TEXT_COL = "speech"

# -------------------------------------------------------------------
# Safe preprocessing (rule-based only)
# -------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z]{2,}")  # alphabetic tokens, length >= 2


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    return " ".join(tokens)


def main():

    df = pd.read_parquet(INPUT_PATH)

    df = df.dropna(subset=[TEXT_COL, "party"])
    df = df[df["party"].isin(["D", "R"])]

    df["congress"] = df["congress"].astype(int)
    df = df[df["congress"] >= 93].reset_index(drop=True)

    df["clean_text"] = df[TEXT_COL].map(clean_text)

    vectorizer = CountVectorizer(
        ngram_range=(2, 2),
        min_df=100,          # Widmer cutoff
        max_df=0.95,
        lowercase=False,
        tokenizer=str.split,
        token_pattern=None
    )

    X = vectorizer.fit_transform(df["clean_text"])

    sp.save_npz(OUT_X_PATH, X)

    vocab = pd.DataFrame({
        "bigram": vectorizer.get_feature_names_out()
    })
    vocab.to_csv(OUT_VOCAB_PATH, index=False)

    meta = df[["speech_id", "congress", "party"]]
    meta.to_parquet(OUT_META_PATH)

    print("Saved bigram matrix:", X.shape)
    print("Saved vocab size:", len(vocab))
    print("Congress range:", meta["congress"].min(), "-", meta["congress"].max())


if __name__ == "__main__":
    main()

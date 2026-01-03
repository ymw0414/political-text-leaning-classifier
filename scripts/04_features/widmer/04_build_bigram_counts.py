"""
04_build_bigram_counts.py

Widmer (2020) Appendix B compliant feature construction:
- lowercase
- punctuation / digits removal
- stopword removal
- Porter stemming
- contiguous bigrams
- party-conditional frequency filtering
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed")

INPUT_PATH = BASE_DIR / "speeches_with_party.parquet"
OUT_X_PATH = BASE_DIR / "widmer_X_bigram_counts.npz"
OUT_VOCAB_PATH = BASE_DIR / "widmer_bigram_vocab.csv"
OUT_META_PATH = BASE_DIR / "widmer_doc_metadata.parquet"

# --------------------------------------------------
# Setup
# --------------------------------------------------
TEXT_COL = "speech"

TOKEN_RE = re.compile(r"[a-z]+")
STEMMER = PorterStemmer()

STOPWORDS = set(stopwords.words("english"))

# minimal Congress-specific stopwords (placeholder)
CONGRESS_STOPWORDS = {
    "mr", "mrs", "speaker", "chairman", "gentleman",
    "yield", "committee", "bill", "amendment"
}

STOPWORDS = STOPWORDS.union(CONGRESS_STOPWORDS)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def clean_and_stem(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    tokens = TOKEN_RE.findall(text)

    processed = []
    for t in tokens:
        stem = STEMMER.stem(t)
        if stem not in STOPWORDS:
            processed.append(stem)

    return " ".join(processed)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    print("Loading data...")
    df = pd.read_parquet(INPUT_PATH)

    df = df.dropna(subset=[TEXT_COL, "party"])
    df = df[df["party"].isin(["D", "R"])]

    df["congress"] = df["congress"].astype(int)
    df = df[df["congress"] >= 93].reset_index(drop=True)

    print("Preprocessing text...")
    df["clean_text"] = df[TEXT_COL].map(clean_and_stem)

    # --------------------------------------------------
    # Initial bigram construction (no filtering yet)
    # --------------------------------------------------
    print("Building initial bigram matrix...")
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),
        lowercase=False,
        tokenizer=str.split,
        token_pattern=None
    )

    X_full = vectorizer.fit_transform(df["clean_text"])
    vocab = np.array(vectorizer.get_feature_names_out())

    # --------------------------------------------------
    # Party-conditional document frequency
    # --------------------------------------------------
    print("Applying Widmer frequency filtering...")

    is_R = (df["party"] == "R").to_numpy()
    is_D = (df["party"] == "D").to_numpy()

    X_bin = (X_full > 0).astype(int)

    df_R = np.asarray(X_bin[is_R].sum(axis=0)).ravel()
    df_D = np.asarray(X_bin[is_D].sum(axis=0)).ravel()

    n_R = is_R.sum()
    n_D = is_D.sum()

    freq_R = df_R / n_R
    freq_D = df_D / n_D

    keep = (
        ((freq_R >= 0.001) | (freq_D >= 0.001)) &
        ((freq_R >= 0.0001) & (freq_D >= 0.0001))
    )

    vocab_kept = vocab[keep]
    X = X_full[:, keep]

    print("Final vocab size:", len(vocab_kept))

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    sp.save_npz(OUT_X_PATH, X)

    pd.DataFrame({"bigram": vocab_kept}).to_csv(OUT_VOCAB_PATH, index=False)

    meta = df[["speech_id", "congress", "party"]]
    meta.to_parquet(OUT_META_PATH)

    print("Saved bigram matrix:", X.shape)
    print("Done.")

if __name__ == "__main__":
    main()

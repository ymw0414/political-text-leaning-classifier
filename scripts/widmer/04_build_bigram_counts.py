"""
04_build_bigram_counts.py

Congressional bigram construction (per Congress)
Aligned with newspaper preprocessing
Widmer (2020) style frequency filtering
Additional removal of legislator last names
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

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

INTER_DIR = BASE_DIR / "data" / "intermediate" / "speeches"
OUT_DIR = BASE_DIR / "data" / "processed" / "speeches" / "bigrams"

INPUT_PATH = INTER_DIR / "speeches_with_party.parquet"
SPEAKER_MAP_PATH = INTER_DIR / "speaker_map.parquet"

OUT_X = OUT_DIR / f"X_congress_{CONGRESS}.npz"
OUT_VOCAB = OUT_DIR / f"vocab_congress_{CONGRESS}.csv"
OUT_META = OUT_DIR / f"meta_congress_{CONGRESS}.parquet"

# --------------------------------------------------
# Preprocessing (identical to newspapers)
# --------------------------------------------------

TOKEN_RE = re.compile(r"[a-z]{2,}")
STEMMER = PorterStemmer()

BASE_STOPWORDS = set(stopwords.words("english"))

STATE_WORDS = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut",
    "delaware","florida","georgia","hawaii","idaho","illinois","indiana","iowa",
    "kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan",
    "minnesota","mississippi","missouri","montana","nebraska","nevada",
    "new","york","jersey","mexico","north","south","carolina","dakota",
    "ohio","oklahoma","oregon","pennsylvania","rhode","island",
    "tennessee","texas","utah","vermont","virginia","washington",
    "west","wisconsin","wyoming"
}

PROCEDURAL_WORDS = {
    "mr","mrs","speaker","chairman","gentleman",
    "yield","committee","bill","amendment","motion"
}

# --------------------------------------------------
# Legislator last names
# --------------------------------------------------

def load_legislator_last_names(path: Path):
    df = pd.read_parquet(path)

    lname_cols = [c for c in df.columns if "last" in c.lower()]
    if not lname_cols:
        raise RuntimeError("No last-name column found in speaker_map")

    col = lname_cols[0]

    names = (
        df[col]
        .dropna()
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z]", "", regex=True)
        .tolist()
    )

    return {STEMMER.stem(n) for n in names if len(n) >= 2}

LEGISLATOR_LAST_NAMES = load_legislator_last_names(SPEAKER_MAP_PATH)

STOPWORDS = (
    BASE_STOPWORDS
    | STATE_WORDS
    | PROCEDURAL_WORDS
    | LEGISLATOR_LAST_NAMES
)

# --------------------------------------------------
# Cleaning function
# --------------------------------------------------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    tokens = TOKEN_RE.findall(text.lower())
    return " ".join(
        STEMMER.stem(t) for t in tokens if t not in STOPWORDS
    )

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building congressional bigrams for Congress {CONGRESS}")

    df = pd.read_parquet(INPUT_PATH)

    df["congress"] = df["congress"].astype(int)
    df = df[
        (df["congress"] == CONGRESS) &
        (df["party"].isin(["D", "R"]))
    ].reset_index(drop=True)

    print(f"Documents before cleaning: {len(df):,}")

    df["clean_text"] = [
        clean_text(t)
        for t in tqdm(df["speech"], desc="Preprocessing text")
    ]

    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    after = len(df)

    print(f"Dropped empty documents: {before - after:,}")
    print(f"Documents after cleaning: {after:,}")

    if len(df) == 0:
        raise RuntimeError(f"No usable documents for Congress {CONGRESS}")

    print("Building bigram count matrix...")
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),
        tokenizer=str.split,
        lowercase=False
    )

    X_full = vectorizer.fit_transform(df["clean_text"])
    vocab = np.array(vectorizer.get_feature_names_out())

    print("Applying Widmer frequency filtering...")
    is_R = (df["party"] == "R").to_numpy()
    is_D = (df["party"] == "D").to_numpy()

    X_bin = (X_full > 0).astype(int)

    df_R = np.asarray(X_bin[is_R].sum(axis=0)).ravel()
    df_D = np.asarray(X_bin[is_D].sum(axis=0)).ravel()

    freq_R = df_R / is_R.sum()
    freq_D = df_D / is_D.sum()

    keep = (
        ((freq_R >= 0.001) | (freq_D >= 0.001)) &
        ((freq_R >= 0.0001) & (freq_D >= 0.0001))
    )

    X = X_full[:, keep]
    vocab_kept = vocab[keep]

    print(f"Final vocabulary size: {len(vocab_kept):,}")

    sp.save_npz(OUT_X, X)
    pd.DataFrame({"bigram": vocab_kept}).to_csv(OUT_VOCAB, index=False)
    df[["speech_id", "congress", "party"]].to_parquet(OUT_META)

    print(f"Saved X: {OUT_X}")
    print(f"Saved vocab: {OUT_VOCAB}")
    print(f"Saved meta: {OUT_META}")
    print(f"X shape: {X.shape}")

if __name__ == "__main__":
    main()

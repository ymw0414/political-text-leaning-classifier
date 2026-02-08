"""
05_build_tfidf.py

Build a uniform TF-IDF matrix from congressional speeches (Congresses 99-108).

Steps:
  1. Load speech text (01) and partisan-core labels (04).
  2. Aggregate all speeches by legislator-congress (icpsr + congress).
  3. Fit a single TF-IDF vectorizer on the full corpus (uniform vocabulary).
  4. Save the sparse TF-IDF matrix, labels, and the fitted vectorizer.

The rolling-window model training (e.g. pooling Congress 100+101)
happens in a later step -- this script only prepares the features.

TF-IDF settings:
  ngram_range  = (1, 2)
  stop_words   = 'english'
  min_df       = 0.001
  sublinear_tf = True
  stemming     = None (preserve partisan nuance)
"""

import os
import joblib
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
INTER_DIR = BASE_DIR / "data" / "intermediate" / "speeches"

SPEECHES_PATH = INTER_DIR / "01_speeches_merged.parquet"
LABELS_PATH = INTER_DIR / "04_speeches_with_partisan_core.parquet"

OUT_DIR = BASE_DIR / "data" / "processed" / "speeches" / "minwoo"
OUT_TFIDF = OUT_DIR / "05_tfidf_matrix.npz"
OUT_META = OUT_DIR / "05_tfidf_meta.parquet"
OUT_VECTORIZER = OUT_DIR / "05_tfidf_vectorizer.joblib"

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
print("Loading speech text ...")
speeches = pd.read_parquet(SPEECHES_PATH)

print("Loading partisan-core labels ...")
labels = pd.read_parquet(LABELS_PATH)

# ------------------------------------------------------------------
# 2. Merge text with labels
# ------------------------------------------------------------------
# labels has speaker metadata + labels but no text
# speeches has speech_id + speech text + congress
# Join on speech_id
speeches["speech_id"] = speeches["speech_id"].astype(str)
labels["speech_id"] = labels["speech_id"].astype(str)

merged = labels.merge(
    speeches[["speech_id", "speech"]],
    on="speech_id",
    how="inner",
)

print(f"  Merged: {len(merged):,} speeches with text + labels")

# ------------------------------------------------------------------
# 3. Aggregate text by legislator-congress
# ------------------------------------------------------------------
print("Aggregating by legislator-congress ...")

agg = (
    merged
    .groupby(["icpsr", "congress_int"])
    .agg(
        text=("speech", lambda x: " ".join(x.astype(str))),
        party_code=("party", "first"),
        label_rep_core=("label_rep_core", "max"),
        label_dem_core=("label_dem_core", "max"),
        nokken_poole_dim1=("nokken_poole_dim1", "first"),
        n_speeches=("speech_id", "count"),
    )
    .reset_index()
)

print(f"  Legislator-congress documents: {len(agg):,}")

# ------------------------------------------------------------------
# 4. Fit TF-IDF vectorizer (uniform vocabulary)
# ------------------------------------------------------------------
print("Fitting TF-IDF vectorizer ...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=0.001,
    sublinear_tf=True,
)

tfidf_matrix = vectorizer.fit_transform(agg["text"])

print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
print(f"  Non-zero entries: {tfidf_matrix.nnz:,}")

# ------------------------------------------------------------------
# 5. Save outputs
# ------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sparse matrix
sp.save_npz(OUT_TFIDF, tfidf_matrix)
print(f"\n  Saved TF-IDF matrix -> {OUT_TFIDF}")

# Metadata (one row per legislator-congress, same order as matrix rows)
meta = agg.drop(columns=["text"])
meta.to_parquet(OUT_META)
print(f"  Saved metadata -> {OUT_META}")

# Vectorizer (needed to transform newspaper text later)
joblib.dump(vectorizer, OUT_VECTORIZER)
print(f"  Saved vectorizer -> {OUT_VECTORIZER}")

# ------------------------------------------------------------------
# 6. Validation summary
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("VALIDATION: Documents per congress")
print("=" * 72)

summary = (
    meta
    .groupby("congress_int")
    .agg(
        n_legislators=("icpsr", "count"),
        rep_core=("label_rep_core", "sum"),
        dem_core=("label_dem_core", "sum"),
        avg_speeches=("n_speeches", "mean"),
    )
)
summary["avg_speeches"] = summary["avg_speeches"].round(1)
summary[["rep_core", "dem_core"]] = summary[["rep_core", "dem_core"]].astype(int)
print(summary.to_string())
print("=" * 72)

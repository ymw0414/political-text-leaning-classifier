"""
05_train_dmr_poisson.py

Widmer / Taddy DMR (Poisson MN) training
- Train window: 1973–1989 (93–101 Congress)
- Output: bigram-level beta + intercept
- Progress via solver verbose
- NO projection
- NO evaluation
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
import time

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed")

X_PATH = BASE_DIR / "widmer_X_bigram_counts.npz"
META_PATH = BASE_DIR / "widmer_doc_metadata.parquet"
VOCAB_PATH = BASE_DIR / "widmer_bigram_vocab.csv"

OUT_BETA_PATH = BASE_DIR / "widmer_bigram_beta_73_89.csv"
OUT_INTERCEPT_PATH = BASE_DIR / "widmer_intercept_73_89.npy"

# -------------------------------------------------
# Load data
# -------------------------------------------------
print("Loading data...")
X = sp.load_npz(X_PATH)
meta = pd.read_parquet(META_PATH)
vocab = pd.read_csv(VOCAB_PATH)

# -------------------------------------------------
# Train sample: 1973–1989 (93–101 Congress)
# -------------------------------------------------
train_mask = (meta["congress"] >= 93) & (meta["congress"] <= 101)

X_train = X[train_mask.values]
y_train = (meta.loc[train_mask, "party"].values == "R").astype(int)

# -------------------------------------------------
# Basic scale info (for progress intuition)
# -------------------------------------------------
print("Training setup")
print("  Documents:", X_train.shape[0])
print("  Features:", X_train.shape[1])
print("  Nonzeros:", X_train.nnz)
print("  R share:", y_train.mean())
print()

# -------------------------------------------------
# Train MN MLE via logistic regression
# -------------------------------------------------
print("Starting training...")
start_time = time.time()

model = LogisticRegression(
    penalty="l2",
    C=1.0,                  # Widmer-style fixed regularization
    fit_intercept=True,     # absorb length / class imbalance
    solver="liblinear",
    max_iter=500,
    verbose=1               # ← progress output
)

model.fit(X_train, y_train)

elapsed = time.time() - start_time
print()
print(f"Training finished in {elapsed/60:.1f} minutes")

# -------------------------------------------------
# Extract parameters
# -------------------------------------------------
beta = model.coef_.ravel()
intercept = model.intercept_[0]  # nuisance, saved only

# -------------------------------------------------
# Save outputs
# -------------------------------------------------
out = vocab.copy()
out["beta"] = beta
out.to_csv(OUT_BETA_PATH, index=False)

np.save(OUT_INTERCEPT_PATH, intercept)

print("Saved beta to:", OUT_BETA_PATH)
print("Saved intercept to:", OUT_INTERCEPT_PATH)
print("Done.")

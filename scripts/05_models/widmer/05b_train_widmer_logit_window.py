"""
05b_train_widmer_logit.py

Estimate partisan bigram loadings (phi) per Congress
using shared Congress–Newspaper vocabulary only.

Implementation:
- Penalized logistic regression with L1 penalty
- Computational substitute for Poisson-approximated multinomial (GST/Widmer)
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.linear_model import LogisticRegression

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
VOCAB_DIR = BASE_DIR / "data" / "processed" / "shared_vocab"
OUT_DIR = BASE_DIR / "data" / "processed" / "models" / "widmer"

X_PATH = SPEECH_DIR / f"X_congress_{CONGRESS}.npz"
META_PATH = SPEECH_DIR / f"meta_congress_{CONGRESS}.parquet"
VOCAB_PATH = SPEECH_DIR / f"vocab_congress_{CONGRESS}.csv"

SHARED_VOCAB_PATH = VOCAB_DIR / f"vocab_shared_congress_{CONGRESS}.csv"

OUT_BETA = OUT_DIR / f"phi_congress_{CONGRESS}.npy"
OUT_INTERCEPT = OUT_DIR / f"intercept_congress_{CONGRESS}.npy"
OUT_VOCAB = OUT_DIR / f"vocab_used_congress_{CONGRESS}.csv"

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training Widmer-style model for Congress {CONGRESS}")

    # Load data
    X = sp.load_npz(X_PATH)
    meta = pd.read_parquet(META_PATH)

    vocab_full = pd.read_csv(VOCAB_PATH)["bigram"].tolist()
    vocab_shared = pd.read_csv(SHARED_VOCAB_PATH)["bigram"].tolist()

    # Labels: Republican=1, Democrat=0
    y = (meta["party"] == "R").astype(int).to_numpy()

    # Align X to shared vocab
    index = {v: i for i, v in enumerate(vocab_full)}
    keep_idx = [index[v] for v in vocab_shared]

    X_shared = X[:, keep_idx]

    print(f"X shape after vocab filtering: {X_shared.shape}")

    # --------------------------------------------------
    # Penalized estimation (L1 logistic)
    # --------------------------------------------------

    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,              # regularization strength (can be tuned)
        max_iter=2000,
        fit_intercept=True
    )

    model.fit(X_shared, y)

    phi = model.coef_.ravel()
    intercept = model.intercept_[0]

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    np.save(OUT_BETA, phi)
    np.save(OUT_INTERCEPT, intercept)
    pd.DataFrame({"bigram": vocab_shared}).to_csv(OUT_VOCAB, index=False)

    print(f"Saved phi: {OUT_BETA}")
    print(f"Saved intercept: {OUT_INTERCEPT}")
    print(f"Final vocab size: {len(vocab_shared):,}")
    print(f"Non-zero coefficients: {(phi != 0).sum():,}")

if __name__ == "__main__":
    main()

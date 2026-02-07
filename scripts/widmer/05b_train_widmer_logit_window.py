# -------------------------------------------------------------------------
# FILE: 05b_train_widmer_logit_window.py
# DESCRIPTION:
#   - Train a single Widmer-style partisan ruler using pooled speeches
#     (98–100 Congress = 1983–1988)
#   - Align each Congress to a shared vocabulary (from 05a)
#   - L1-penalized logistic regression
# -------------------------------------------------------------------------

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
parser.add_argument("--speech_congresses", nargs="+", type=int, required=True)
args = parser.parse_args()

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

SPEECH_DIR = BASE_DIR / "data" / "processed" / "speeches" / "bigrams"
VOCAB_DIR  = BASE_DIR / "data" / "processed" / "shared_vocab"
OUT_DIR    = BASE_DIR / "data" / "processed" / "models" / "widmer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARED_VOCAB_PATH = VOCAB_DIR / "vocab_shared_speech_83_88__news_86_04.csv"

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    # Load shared vocab (global coordinate system)
    shared_vocab = pd.read_csv(SHARED_VOCAB_PATH)["bigram"].astype(str).tolist()
    p = len(shared_vocab)

    X_blocks = []
    y_blocks = []

    for c in args.speech_congresses:
        print(f"Loading Congress {c}")

        Xc = sp.load_npz(SPEECH_DIR / f"X_congress_{c}.npz")
        meta = pd.read_parquet(SPEECH_DIR / f"meta_congress_{c}.parquet")
        vocab_c = pd.read_csv(SPEECH_DIR / f"vocab_congress_{c}.csv")["bigram"].astype(str).tolist()

        # Map vocab -> indices for this Congress
        idx = {v: i for i, v in enumerate(vocab_c)}

        # Build X aligned to shared vocab (missing terms -> zero columns)
        cols = []
        for term in shared_vocab:
            if term in idx:
                cols.append(Xc[:, idx[term]])
            else:
                cols.append(sp.csr_matrix((Xc.shape[0], 1)))

        Xc_shared = sp.hstack(cols, format="csr")

        y = (meta["party"] == "R").astype(int).to_numpy()

        X_blocks.append(Xc_shared)
        y_blocks.append(y)

        print(f"  Rows: {Xc_shared.shape[0]}, Cols: {Xc_shared.shape[1]}")

    # Stack across Congresses
    X = sp.vstack(X_blocks, format="csr")
    y = np.concatenate(y_blocks)

    print(f"Final X shape: {X.shape}")

    # --------------------------------------------------
    # Estimate Widmer-style logit
    # --------------------------------------------------
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        max_iter=2000,
        fit_intercept=True
    )

    model.fit(X, y)

    phi = model.coef_.ravel()
    intercept = model.intercept_[0]

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    np.save(OUT_DIR / "phi_preNAFTA_83_88.npy", phi)
    np.save(OUT_DIR / "intercept_preNAFTA_83_88.npy", intercept)
    pd.DataFrame({"bigram": shared_vocab}).to_csv(
        OUT_DIR / "vocab_used_preNAFTA_83_88.csv", index=False
    )

    print("Saved pre-NAFTA ruler (83–88)")
    print(f"Vocab size: {len(shared_vocab):,}")
    print(f"Non-zero coefs: {(phi != 0).sum():,}")

if __name__ == "__main__":
    main()

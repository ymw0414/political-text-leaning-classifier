"""
05_train_poisson.py

Widmer-exact estimation:
- Multinomial logit approximated by Poisson
- Separate Poisson regression per document with party indicator
- Implemented as penalized Poisson GLM on bigram counts
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed")

X_PATH = BASE_DIR / "widmer_X_bigram_counts.npz"
META_PATH = BASE_DIR / "widmer_doc_metadata.parquet"
VOCAB_PATH = BASE_DIR / "widmer_bigram_vocab.csv"

OUT_COEF_PATH = BASE_DIR / "widmer_poisson_coefficients.csv"

# -------------------------------------------------------------------
# Load
# -------------------------------------------------------------------
X = sp.load_npz(X_PATH)
meta = pd.read_parquet(META_PATH)
vocab = pd.read_csv(VOCAB_PATH)

# -------------------------------------------------------------------
# Label
# -------------------------------------------------------------------
y = (meta["party"] == "R").astype(int).to_numpy()

# -------------------------------------------------------------------
# Time-safe split
# -------------------------------------------------------------------
train_idx = meta["congress"] <= 103
val_idx   = meta["congress"] >= 104

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]

# -------------------------------------------------------------------
# Poisson regression (log-link)
# -------------------------------------------------------------------
model = PoissonRegressor(
    alpha=1.0,        # L2 penalty strength
    max_iter=300,
    fit_intercept=True
)

model.fit(X_train, y_train)

# -------------------------------------------------------------------
# Validation (linear index, Widmer-style)
# -------------------------------------------------------------------
val_score = X_val @ model.coef_ + model.intercept_
auc = roc_auc_score(y_val, val_score)
print("Validation AUC:", auc)

# -------------------------------------------------------------------
# Save coefficients
# -------------------------------------------------------------------
coef_df = pd.DataFrame({
    "bigram": vocab["bigram"],
    "coef": model.coef_
}).sort_values("coef")

coef_df.to_csv(OUT_COEF_PATH, index=False)

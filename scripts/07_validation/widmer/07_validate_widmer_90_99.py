"""
07_validate_widmer_90_99.py

Widmer-style accuracy evaluation on test set (1990–1999):
- Use slant from 06 projection
- Classify as Republican if slant > 0
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed")

SLANT_PATH = BASE_DIR / "widmer_slant_90_99.parquet"
META_PATH = BASE_DIR / "widmer_doc_metadata.parquet"

# --------------------------------------------------
# Load data
# --------------------------------------------------
print("Loading data...")

slant_df = pd.read_parquet(SLANT_PATH)
meta = pd.read_parquet(META_PATH)

# --------------------------------------------------
# Restrict meta to TEST set (1990–1999)
# Congress mapping:
# 102–106 = 1991–1999
# --------------------------------------------------
test_mask = ((meta["congress"] >= 102) & (meta["congress"] <= 106))

meta_test = meta.loc[test_mask, ["speech_id", "party"]].reset_index(drop=True)

# --------------------------------------------------
# Merge slant with labels
# --------------------------------------------------
df = slant_df.merge(meta_test, on="speech_id", how="left")

# sanity check
assert df["party"].notna().all()

y_true = (df["party"] == "R").astype(int)
y_pred = (df["slant_widmer"] > 0).astype(int)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print("\nConfusion Matrix (Widmer rule: slant > 0):")
print(cm)

print(f"\nAccuracy: {acc:.4f}")

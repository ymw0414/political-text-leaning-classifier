"""
07a_generate_paper_crosswalk.py

STEP A.
Build a global newspaper name crosswalk for panel construction.

Canonical (Target):
- Newspapers that EXISTED in 1986–1993 (Fixed Universe)
- max yearly count >= 1000 within this period
- SORTED BY LENGTH (Longest First) to handle mergers correctly

Source (Dirty):
- Newspapers appearing in 1994–2008
- total count >= 50

Matching Priority:
1) Exact match (case-insensitive)
2) Startswith (Longest canonical name first)
3) Fuzzy match (rapidfuzz.partial_ratio >= 85)

Outputs:
- paper_name_crosswalk.csv (Main)
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from rapidfuzz import process, fuzz
from tqdm import tqdm

# --------------------------------------------------
# Paths (PROJECT STANDARD)
# --------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

YEARLY_DIR = BASE_DIR / "data" / "intermediate" / "newspapers" / "yearly"
OUT_DIR = BASE_DIR / "data" / "meta" / "newspapers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUT_DIR / "paper_name_crosswalk.csv"
OUT_EXACT = OUT_DIR / "paper_name_crosswalk_exact.csv"
OUT_FUZZY = OUT_DIR / "paper_name_crosswalk_fuzzy.csv"
OUT_FAIL = OUT_DIR / "paper_name_crosswalk_fail.csv"

# --------------------------------------------------
# Parameters (Event-Study Design)
# --------------------------------------------------
CANON_YEAR_START = 1986
CANON_YEAR_END = 1993          # Reference Period
CANON_MIN_YEARLY = 1000

DIRTY_YEAR_START = 1994        # Target Period
DIRTY_YEAR_END = 2008
DIRTY_MIN_TOTAL = 50

FUZZY_CUTOFF = 85

# --------------------------------------------------
# 1. Build Canonical list (1986–1993)
# --------------------------------------------------
yearly_counts = defaultdict(dict)

print(f"Building canonical list ({CANON_YEAR_START}–{CANON_YEAR_END})...")

for p in YEARLY_DIR.glob("newspapers_*.parquet"):
    year = int(p.stem.split("_")[1])
    if CANON_YEAR_START <= year <= CANON_YEAR_END:
        df = pd.read_parquet(p, columns=["paper"])
        vc = df["paper"].value_counts()
        for name, cnt in vc.items():
            yearly_counts[name][year] = cnt

canon_rows = []
for name, ydict in yearly_counts.items():
    canon_rows.append({
        "paper": name,
        "max_yearly": max(ydict.values()),
        "total": sum(ydict.values()),
    })

canon_df = pd.DataFrame(canon_rows)

# [CRITICAL FIX] Sort by LENGTH (descending), not frequency.
# This ensures "Arkansas Democrat-Gazette" comes before "Arkansas Democrat".
valid_canon_papers = canon_df[canon_df["max_yearly"] >= CANON_MIN_YEARLY]["paper"].tolist()
canonical = sorted(valid_canon_papers, key=len, reverse=True)

print(f"Canonical newspapers selected: {len(canonical)}")
if canonical:
    print(f"Top 5 Longest (Check): {canonical[:5]}")

# --------------------------------------------------
# 2. Build Dirty list (1994–2008)
# --------------------------------------------------
dirty_counts = defaultdict(int)

print(f"Building dirty-name list ({DIRTY_YEAR_START}–{DIRTY_YEAR_END})...")

for p in YEARLY_DIR.glob("newspapers_*.parquet"):
    year = int(p.stem.split("_")[1])
    if DIRTY_YEAR_START <= year <= DIRTY_YEAR_END:
        df = pd.read_parquet(p, columns=["paper"])
        vc = df["paper"].value_counts()
        for name, cnt in vc.items():
            dirty_counts[name] += cnt

dirty = [name for name, cnt in dirty_counts.items() if cnt >= DIRTY_MIN_TOTAL]

print(f"Dirty names to match: {len(dirty)}")

# --------------------------------------------------
# 3. Matching Logic
# --------------------------------------------------
results = []
canonical_set = set(c.lower() for c in canonical) # For O(1) exact lookup
canon_lower_list = [(c, c.lower()) for c in canonical] # Already sorted by length

print("Matching dirty names to canonical list...")

for d in tqdm(dirty, desc="Matching"):
    d_low = d.lower()
    matched = False

    # Priority 1: Exact Match (Fastest & Safest)
    if d_low in canonical_set:
        # Retrieve original casing from canonical list
        clean_name = next(c for c in canonical if c.lower() == d_low)
        results.append({
            "original_name": d,
            "clean_name": clean_name,
            "score": 100,
            "method": "exact_match",
        })
        continue

    # Priority 2: Startswith (Longest Match First)
    for c, c_low in canon_lower_list:
        # [Safety] Skip very short canonical names to prevent false positives
        if len(c) < 4:
            continue

        if d_low.startswith(c_low):
            results.append({
                "original_name": d,
                "clean_name": c,
                "score": 100,
                "method": "startswith",
            })
            matched = True
            break # Breaks immediately on longest match

    if matched:
        continue

    # Priority 3: Fuzzy Match (RapidFuzz)
    cand = process.extractOne(d, canonical, scorer=fuzz.partial_ratio)

    if cand and cand[1] >= FUZZY_CUTOFF:
        results.append({
            "original_name": d,
            "clean_name": cand[0],
            "score": cand[1],
            "method": "fuzzy",
        })
    else:
        results.append({
            "original_name": d,
            "clean_name": None,
            "score": 0,
            "method": "fail",
        })

# --------------------------------------------------
# 4. Save results
# --------------------------------------------------
df_out = pd.DataFrame(results)

# Save Main Crosswalk
df_out.to_csv(OUT_ALL, index=False)

# Save Debug Files
df_out[df_out["method"].isin(["exact_match", "startswith"])].to_csv(OUT_EXACT, index=False)
df_out[df_out["method"] == "fuzzy"].to_csv(OUT_FUZZY, index=False)
df_out[df_out["method"] == "fail"].to_csv(OUT_FAIL, index=False)

print("\n07a complete.")
print(f"Saved crosswalk to: {OUT_ALL}")
print("\nMatching Stats:")
print(df_out["method"].value_counts())
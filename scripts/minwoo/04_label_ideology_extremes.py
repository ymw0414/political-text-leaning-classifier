"""
04_label_ideology_extremes.py

Merge Hein-Bound speaker data with Voteview DW-NOMINATE scores
and label ideology extremes using nokken_poole_dim1.

Labels (per congress, per party):
  - label_radical_right : Republican top 20th percentile (most conservative)
  - label_radical_left  : Democrat  bottom 20th percentile (most liberal)

Matching strategy (Hein-Bound <-> Voteview):
  1. Normalize last names  (strip apostrophes, hyphens, etc.)
  2. Match on  last_name + state + congress
  3. Disambiguate ties via first-name token overlap
  4. Report unmatched / ambiguous cases for manual review
"""

import os
import re
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

SPEAKER_MAP_PATH = BASE_DIR / "data" / "intermediate" / "speeches" / "02_speaker_map.parquet"
HSALL_PATH = BASE_DIR / "data" / "raw" / "voteview_nominate" / "HSall_members.csv"
OUT_PATH = BASE_DIR / "data" / "intermediate" / "speeches" / "04_speeches_with_ideology.parquet"

CONGRESS_MIN, CONGRESS_MAX = 99, 108

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def norm_name(s: str) -> str:
    """Strip to uppercase ASCII letters only."""
    return re.sub(r"[^A-Z]", "", str(s).upper())


def first_name_tokens(s: str) -> set:
    """Return set of uppercase alpha tokens from a name string."""
    return {norm_name(t) for t in str(s).split() if len(norm_name(t)) >= 2}


# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
print("Loading speaker map ...")
sp = pd.read_parquet(SPEAKER_MAP_PATH)

print("Loading Voteview HSall_members ...")
hs = pd.read_csv(HSALL_PATH, low_memory=False)

# ------------------------------------------------------------------
# 2. Filter scope
# ------------------------------------------------------------------
sp = sp.copy()
sp["congress_int"] = sp["congress"].astype(int)
sp = sp[(sp["congress_int"] >= CONGRESS_MIN) & (sp["congress_int"] <= CONGRESS_MAX)]
sp = sp[sp["chamber"] == "H"]

hs = hs[(hs["congress"] >= CONGRESS_MIN) & (hs["congress"] <= CONGRESS_MAX)]
hs = hs[hs["chamber"] == "House"]
hs = hs[hs["party_code"].isin([100, 200])]
hs = hs.dropna(subset=["nokken_poole_dim1"])

print(f"  Hein-Bound speeches (House, {CONGRESS_MIN}-{CONGRESS_MAX}): {len(sp):,}")
print(f"  Voteview members  (House, D/R, {CONGRESS_MIN}-{CONGRESS_MAX}): {len(hs):,}")

# ------------------------------------------------------------------
# 3. Build crosswalk:  speakerid -> icpsr
# ------------------------------------------------------------------
# Unique speakers in Hein-Bound
sp_speakers = (
    sp[["speakerid", "lastname", "firstname", "state", "congress_int"]]
    .drop_duplicates(subset=["speakerid"])
    .copy()
)
sp_speakers["last_norm"] = sp_speakers["lastname"].apply(norm_name)

# Parse Voteview names
hs_xw = hs[["icpsr", "bioname", "state_abbrev", "congress", "party_code",
             "nokken_poole_dim1"]].copy()
hs_xw["last_norm"] = hs_xw["bioname"].str.split(",").str[0].apply(norm_name)
hs_xw["first_raw"] = hs_xw["bioname"].str.split(",").str[1].str.strip()

# --- Step A: join on last_norm + state + congress ---
merged = sp_speakers.merge(
    hs_xw,
    left_on=["last_norm", "state", "congress_int"],
    right_on=["last_norm", "state_abbrev", "congress"],
    how="left",
)

# --- Step B: for groups with >1 Voteview match, pick best first-name overlap ---
def pick_best(g):
    if len(g) == 1:
        return g
    hein_tokens = first_name_tokens(g.iloc[0]["firstname"])
    scores = g["first_raw"].apply(
        lambda x: len(hein_tokens & first_name_tokens(x))
    )
    return g.loc[[scores.idxmax()]]

merged = merged.groupby("speakerid", group_keys=False).apply(pick_best)

# --- Diagnostics ---
matched = merged["icpsr"].notna().sum()
total = len(merged)
print(f"\n  Crosswalk: {matched}/{total} matched ({matched/total*100:.1f}%)")

unmatched = merged[merged["icpsr"].isna()]
if len(unmatched) > 0:
    print(f"  Unmatched speakers ({len(unmatched)}):")
    for _, r in unmatched.head(15).iterrows():
        print(f"    cong {r['congress_int']}  {r['lastname']}, {r['firstname']}  ({r['state']})")
    if len(unmatched) > 15:
        print(f"    ... and {len(unmatched) - 15} more")

# Keep only matched
crosswalk = merged[merged["icpsr"].notna()][["speakerid", "icpsr"]].copy()
crosswalk["icpsr"] = crosswalk["icpsr"].astype(int)

# ------------------------------------------------------------------
# 4. Label ideology extremes (per congress, per party)
# ------------------------------------------------------------------
# Build legislator-level label table
leg = hs[["icpsr", "congress", "party_code", "nokken_poole_dim1"]].copy()

leg["label_radical_right"] = 0
leg["label_radical_left"] = 0

for cong, grp in leg.groupby("congress"):
    # Radical Right: Republicans in top 20% (most conservative)
    reps = grp[grp["party_code"] == 200]
    if len(reps) > 0:
        threshold = reps["nokken_poole_dim1"].quantile(0.80)
        mask = (
            (leg["congress"] == cong)
            & (leg["party_code"] == 200)
            & (leg["nokken_poole_dim1"] >= threshold)
        )
        leg.loc[mask, "label_radical_right"] = 1

    # Radical Left: Democrats in bottom 20% (most liberal)
    dems = grp[grp["party_code"] == 100]
    if len(dems) > 0:
        threshold = dems["nokken_poole_dim1"].quantile(0.20)
        mask = (
            (leg["congress"] == cong)
            & (leg["party_code"] == 100)
            & (leg["nokken_poole_dim1"] <= threshold)
        )
        leg.loc[mask, "label_radical_left"] = 1

# ------------------------------------------------------------------
# 5. Attach labels to speeches via crosswalk
# ------------------------------------------------------------------
sp_labeled = sp.merge(crosswalk, on="speakerid", how="inner")
sp_labeled = sp_labeled.merge(
    leg[["icpsr", "congress", "label_radical_right", "label_radical_left",
         "nokken_poole_dim1"]],
    left_on=["icpsr", "congress_int"],
    right_on=["icpsr", "congress"],
    how="left",
)

# Drop helper columns
sp_labeled = sp_labeled.drop(columns=["congress"], errors="ignore")

# ------------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sp_labeled.to_parquet(OUT_PATH)
print(f"\n  Saved -> {OUT_PATH}")
print(f"  Shape: {sp_labeled.shape}")

# ------------------------------------------------------------------
# 7. Validation summary
# ------------------------------------------------------------------

print("\n" + "=" * 72)
print("VALIDATION: Label counts per congress  (speech-level)")
print("=" * 72)

summary = (
    sp_labeled
    .groupby("congress_int")
    .agg(
        total_speeches=("speech_id", "count"),
        radical_right=("label_radical_right", "sum"),
        radical_left=("label_radical_left", "sum"),
    )
    .astype(int)
)
print(summary.to_string())
print("=" * 72)

# Unique legislator counts
print("\nVALIDATION: Unique legislators per label per congress")
print("=" * 72)
for label_col, label_name in [("label_radical_right", "Radical Right"),
                               ("label_radical_left", "Radical Left")]:
    sub = sp_labeled[sp_labeled[label_col] == 1]
    counts = sub.groupby("congress_int")["icpsr"].nunique()
    print(f"\n  {label_name} (unique members):")
    print(f"  {counts.to_dict()}")
print("=" * 72)

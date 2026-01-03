"""
03_add_party_label.py

Merge speeches with SpeakerMap using (speech_id, congress)
to attach party labels.
"""

import pandas as pd
from pathlib import Path

INTER_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\intermediate\speeches"
)

SPEECHES_PATH = INTER_DIR / "speeches_merged.parquet"
SPEAKER_MAP_PATH = INTER_DIR / "speaker_map.parquet"
OUT_PATH = INTER_DIR / "speeches_with_party.parquet"


def main():
    speeches = pd.read_parquet(SPEECHES_PATH)
    speaker_map = pd.read_parquet(SPEAKER_MAP_PATH)

    # Ensure consistent types
    speeches["speech_id"] = speeches["speech_id"].astype(str)
    speaker_map["speech_id"] = speaker_map["speech_id"].astype(str)

    speeches["congress"] = speeches["congress"].astype(str)
    speaker_map["congress"] = speaker_map["congress"].astype(str)

    # Keep required columns only
    speaker_map = speaker_map[
        ["speech_id", "congress", "party"]
    ]

    # Merge speeches with party labels
    merged = speeches.merge(
        speaker_map,
        on=["speech_id", "congress"],
        how="left",
        validate="m:1"
    )

    merged.to_parquet(OUT_PATH)
    print("Saved:", OUT_PATH, merged.shape)

    # Sanity check
    print("\nParty value counts:")
    print(merged["party"].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    main()

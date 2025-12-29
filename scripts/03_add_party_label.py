"""
03_add_party_label.py

Merge speeches with SpeakerMap using (speech_id, congress)
to correctly attach party labels.
"""

import pandas as pd
from pathlib import Path

PROC_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed"
)

SPEECHES_PATH = PROC_DIR / "speeches_merged.parquet"
SPEAKER_MAP_PATH = PROC_DIR / "speaker_map.parquet"
OUT_PATH = PROC_DIR / "speeches_with_party.parquet"


def main():
    speeches = pd.read_parquet(SPEECHES_PATH)
    speaker_map = pd.read_parquet(SPEAKER_MAP_PATH)

    # 1. 타입 통일 (매우 중요)
    speeches["speech_id"] = speeches["speech_id"].astype(str)
    speaker_map["speech_id"] = speaker_map["speech_id"].astype(str)

    speeches["congress"] = speeches["congress"].astype(str)
    speaker_map["congress"] = speaker_map["congress"].astype(str)

    # 2. 필요한 컬럼만 사용
    speaker_map = speaker_map[
        ["speech_id", "congress", "party"]
    ]

    # 3. 안전한 merge
    merged = speeches.merge(
        speaker_map,
        on=["speech_id", "congress"],
        how="left",
        validate="m:1"
    )

    merged.to_parquet(OUT_PATH)
    print("Saved:", OUT_PATH, merged.shape)

    # 4. 빠른 sanity check
    print("\nParty value counts:")
    print(merged["party"].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    main()

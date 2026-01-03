"""
02_merge_speaker_map.py

Load SpeakerMap files correctly and save speaker-party mapping.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\raw\speeches\hein-bound"
)

OUT_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\intermediate\speeches"
)


def load_speaker_map(raw_dir: Path) -> pd.DataFrame:
    dfs = []

    for i in range(43, 112):
        suffix = f"{i:03d}"
        file = raw_dir / f"{suffix}_SpeakerMap.txt"

        if not file.exists():
            print("skip:", file)
            continue

        df = pd.read_csv(
            file,
            sep="|",
            dtype=str,
            encoding="cp1252",
        )

        df["congress"] = suffix
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No SpeakerMap files loaded")

    return pd.concat(dfs, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_speaker_map(RAW_DIR)

    out = OUT_DIR / "speaker_map.parquet"
    df.to_parquet(out)

    print("Saved:", out, df.shape)


if __name__ == "__main__":
    main()

"""
01_load_speeches.py
Load Congressional speeches and save as parquet
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\raw\Congressional Speech Record Data\hein-bound"
)
OUT_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed"
)


def load_speeches(raw_dir: Path) -> pd.DataFrame:
    rows = []

    for i in range(43, 112):
        suffix = f"{i:03d}"
        file = raw_dir / f"speeches_{suffix}.txt"

        if not file.exists():
            print("skip:", file)
            continue

        with open(file, encoding="cp1252") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("|", 1)
                if len(parts) == 2:
                    rows.append((parts[0], parts[1], suffix))

    return pd.DataFrame(rows, columns=["speech_id", "speech", "congress"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_speeches(RAW_DIR)

    out = OUT_DIR / "speeches_merged.parquet"
    df.to_parquet(out)
    print("Saved:", out, df.shape)


if __name__ == "__main__":
    main()

"""
check_scrapes_raw_structure.py

Inspect file extensions and directory structure in scrapes_raw.
"""

from pathlib import Path
from collections import Counter

RAW_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\raw\newspapers\scrapes_raw"
)


def main():
    files = list(RAW_DIR.rglob("*"))
    files = [f for f in files if f.is_file()]

    print("Total files:", len(files))

    # Extensions
    exts = Counter(f.suffix.lower() for f in files)
    print("\nExtensions:")
    for k, v in exts.most_common():
        print(f"  {k or '<no suffix>'}: {v}")

    # Sample filenames
    print("\nSample file paths:")
    for f in files[:10]:
        print(" ", f.relative_to(RAW_DIR))

    # Detect year patterns
    years = Counter()
    for f in files:
        for y in range(2005, 2009):
            if str(y) in f.name or str(y) in str(f.parent):
                years[y] += 1

    print("\nFiles mentioning year:")
    for y in sorted(years):
        print(f"  {y}: {years[y]}")


if __name__ == "__main__":
    main()

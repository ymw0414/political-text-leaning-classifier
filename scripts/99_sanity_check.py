import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

BASE = r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed"
path = BASE + r"\speeches_with_party.parquet"

# 전체 읽지 말고 필요한 컬럼만
df = pd.read_parquet(
    path,
    columns=["speech_id", "congress", "party", "speech"]
)

# 샘플 먼저 뽑고
sample = df.sample(5, random_state=1).copy()

# 샘플에만 preview 생성 (핵심)
sample["speech_preview"] = (
    sample["speech"].str.split().str[:10].str.join(" ")
)

print("\n--- SAMPLE ROWS ---")
print(
    sample[
        ["speech_id", "congress", "party", "speech_preview"]
    ].to_string(index=False)
)

import pandas as pd
import os

# 1. 파일 경로 설정 (사용자 환경에 맞춤)
BASE_DIR = r"C:\Users\ymw04\Dropbox\shifting_slant"
CSV_PATH = os.path.join(BASE_DIR, "data", "analysis", "newspaper_panel_with_geo.csv")

def check_data():
    print("-" * 60)
    print(f"📂 Reading file: {CSV_PATH}")
    print("-" * 60)

    try:
        # 데이터 로드
        df = pd.read_csv(CSV_PATH)

        # 2. 기본 정보 출력
        print(f"\n✅ Total Rows: {len(df)}")
        print(f"✅ Total Columns: {len(df.columns)}")

        print("\n📋 [Variable Names & Data Types]")
        print(df.dtypes)

        # 3. 데이터 예시 (상위 3줄)
        print("\n👀 [Data Preview (First 3 rows)]")
        # 보기 좋게 전치(Transpose)해서 출력
        print(df.head(3).T)

        # 4. 핵심 키 변수 점검 (fips, year)
        print("\n🔍 [Key Variable Check]")
        if 'fips' in df.columns:
            print(f" - 'fips' example: {df['fips'].iloc[0]} (Type: {type(df['fips'].iloc[0])})")
        else:
            print(" - ⚠️ 'fips' column NOT found!")

        if 'year' in df.columns:
            print(f" - 'year' example: {df['year'].iloc[0]} (Type: {type(df['year'].iloc[0])})")
        else:
            print(" - ⚠️ 'year' column NOT found!")

    except FileNotFoundError:
        print("❌ Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_data()
import pandas as pd
from pathlib import Path
import os

# 경로 설정
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
META_DIR = BASE_DIR / "data" / "meta" / "newspapers"
FUZZY_PATH = META_DIR / "paper_name_crosswalk_fuzzy.csv"
OUTPUT_PATH = META_DIR / "paper_name_crosswalk_fuzzy_checked.csv"


def main():
    # 1. 파일 로드
    if not FUZZY_PATH.exists():
        print("파일이 없습니다.")
        return
    df = pd.read_csv(FUZZY_PATH)

    # 2. 판정 로직 적용 (기본값: 1-사용)
    df['valid'] = 1
    df['reason'] = 'OK'

    # (1) Jet (USA) 오매칭 처리
    mask_jet = df['clean_name'] == "Jet (USA)"
    df.loc[mask_jet, 'valid'] = 0
    df.loc[mask_jet, 'reason'] = 'Bad Match (Jet Magazine)'

    # (2) 짧은 단어 오매칭 처리
    bad_short_terms = ["Science", "Time", "Twin", "Mill", "Eagle", "Columbian"]
    mask_short = df['original_name'].isin(bad_short_terms)
    df.loc[mask_short, 'valid'] = 0
    df.loc[mask_short, 'reason'] = 'Bad Match (Short/Generic Word)'

    # (3) 지역 불일치 등 특정 오매칭
    bad_pairs = [
        ("Wapakoneta Daily News (OH)", "Dayton Daily News (OH)"),
        ("Morton Times", "The Washington Times"),
    ]
    for dirty, clean in bad_pairs:
        mask = (df['original_name'] == dirty) & (df['clean_name'] == clean)
        df.loc[mask, 'valid'] = 0
        df.loc[mask, 'reason'] = 'Location Mismatch'

    # 3. 결과 저장
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"검증 파일 생성 완료: {OUTPUT_PATH}")
    print(f"- 전체: {len(df)}개")
    print(f"- 사용(Valid=1): {len(df[df['valid'] == 1])}개")
    print(f"- 삭제(Valid=0): {len(df[df['valid'] == 0])}개")
    print("엑셀에서 'valid' 컬럼을 확인하세요.")


if __name__ == "__main__":
    main()
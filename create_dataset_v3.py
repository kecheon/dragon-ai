import pandas as pd
import pandas_ta as ta
import numpy as np
import config

def create_dataset_v3():
    """
    TTM 스퀴즈 등 고급 피처를 사용하여 학습 데이터셋 v3를 생성합니다.
    """
    print("--- 데이터셋 v3 생성 시작 (고급 피처) ---")
    
    # --- 1. 데이터 로드 ---
    print("데이터 로드를 시작합니다...")
    try:
        df = pd.read_csv(
            "BTCUSDT_5m_raw_data.csv.2022",
            index_col='Timestamp',
            parse_dates=True
        )
    except FileNotFoundError:
        print("오류: BTCUSDT_5m_raw_data.csv.2025 파일을 찾을 수 없습니다.")
        return

    # --- 2. 고급 피처 생성 ---
    print("TTM 스퀴즈 등 고급 피처들을 생성합니다...")
    
    # TTM 스퀴즈 지표 생성
    squeeze = ta.squeeze(df['High'], df['Low'], df['Close'], lazy_bear=False)
    # 스퀴즈 상태(SQZ_ON) 컬럼만 가져오기 (1: 스퀴즈 켜짐, 0: 꺼짐)
    squeeze_col_name = [col for col in squeeze.columns if 'SQZ_ON' in col][0]
    df['squeeze_on'] = squeeze[squeeze_col_name]

    # ATR (횡보 기간 카운트를 위해 필요)
    df['atr_normalized'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
    
    # 횡보 기간 카운트 피처
    # ATR이 매우 낮은 상태(예: 0.1% 미만)가 지난 2시간(24개 캔들) 동안 몇 번 있었는지 카운트
    LOW_ATR_THRESHOLD = 0.001 
    df['consolidation_count'] = (df['atr_normalized'] < LOW_ATR_THRESHOLD).rolling(window=24).sum()
    
    # 기존 피처 중 유효했던 것들 추가
    df['adx'] = ta.adx(high=df['High'], low=df['Low'], close=df['Close'], length=config.WINDOW).iloc[:, 0]
    bbands = ta.bbands(df['Close'], length=20, std=2)
    bb_width_col = [col for col in bbands.columns if 'BBB' in col][0]
    df['bb_width'] = bbands[bb_width_col]

    df.dropna(inplace=True)
    print("피처 생성 완료.")

    # --- 3. 레이블링 (이전과 동일) ---
    print("레이블링을 시작합니다...")
    PAST_WINDOW = 48
    FUTURE_WINDOW = 12
    VOL_MULTIPLIER = 2.0

    df['future_volatility'] = df['Close'].pct_change().rolling(FUTURE_WINDOW).std().shift(-FUTURE_WINDOW)
    df['past_volatility'] = df['Close'].pct_change().rolling(PAST_WINDOW).std()
    df['label'] = np.where(
        df['future_volatility'] > (df['past_volatility'] * VOL_MULTIPLIER), 1, 0
    )
    
    final_df = df.drop(columns=['future_volatility', 'past_volatility']).dropna()
    final_df['label'] = final_df['label'].astype(int)
    print("레이블링 완료.")

    # --- 4. 결과 확인 및 저장 ---
    features_v3 = ['squeeze_on', 'consolidation_count', 'adx', 'bb_width']
    output_df = final_df[features_v3 + ['label']]
    
    print("\n--- 최종 데이터셋 정보 (v3) ---")
    print(f"전체 데이터 개수: {len(output_df)}")
    print("\n레이블 분포:")
    print(output_df['label'].value_counts())
    
    output_filename = "volatility_data_v3.csv"
    output_df.to_csv(output_filename)
    print(f"\n고급 피처가 포함된 데이터셋 v3를 '{output_filename}'으로 저장했습니다.")

if __name__ == "__main__":
    create_dataset_v3()

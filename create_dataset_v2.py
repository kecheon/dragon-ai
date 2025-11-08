import pandas as pd
import pandas_ta as ta
import numpy as np
import config

def create_dataset_v2():
    """
    변동성 예측에 특화된 피처들을 사용하여 학습 데이터셋 v2를 생성합니다. (수정된 버전)
    """
    print("--- 데이터셋 v2 생성 시작 (수정된 버전) ---")
    
    # --- 1. 데이터 로드 ---
    print("데이터 로드를 시작합니다...")
    try:
        df = pd.read_csv(
            "BTCUSDT_5m_raw_data.csv.2025",
            index_col='Timestamp',
            parse_dates=True
        )
    except FileNotFoundError:
        print("오류: BTCUSDT_5m_raw_data.csv.2025 파일을 찾을 수 없습니다.")
        return

    # --- 2. 신규 피처 생성 ---
    print("변동성 예측에 특화된 피처들을 생성합니다...")
    
    # ==================================================================
    # 수정된 부분: 볼린저 밴드 폭을 동적으로 찾아서 할당
    # ==================================================================
    bbands = ta.bbands(df['Close'], length=20, std=2)
    
    # pandas_ta가 생성하는 볼린저 밴드 폭('BBB') 컬럼 이름을 동적으로 찾기
    try:
        bb_width_col = [col for col in bbands.columns if 'BBB' in col][0]
        df['bb_width'] = bbands[bb_width_col]
        print(f"볼린저 밴드 폭 피처 ('{bb_width_col}')를 성공적으로 생성했습니다.")
    except IndexError:
        print("오류: pandas_ta 라이브러리에서 볼린저 밴드 폭('BBB') 컬럼을 찾을 수 없습니다.")
        # 대체 방법: 수동 계산
        try:
            upper_col = [col for col in bbands.columns if 'BBU' in col][0]
            middle_col = [col for col in bbands.columns if 'BBM' in col][0]
            lower_col = [col for col in bbands.columns if 'BBL' in col][0]
            df['bb_width'] = (bbands[upper_col] - bbands[lower_col]) / bbands[middle_col]
            print("대체 방법으로 볼린저 밴드 폭을 수동 계산했습니다.")
        except IndexError:
            print("치명적 오류: 볼린저 밴드 관련 컬럼을 찾을 수 없어 'bb_width' 피처 생성에 실패했습니다.")
            return
    # ==================================================================

    # 정규화된 ATR
    df['atr_normalized'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
    
    # ADX
    adx_df = ta.adx(high=df['High'], low=df['Low'], close=df['Close'], length=config.WINDOW)
    # ADX 컬럼 이름 동적으로 찾기
    adx_col_name = [col for col in adx_df.columns if 'ADX' in col][0]
    df['adx'] = adx_df[adx_col_name]
    
    # 변동성 (가격 변동률의 표준편차) 및 변화율(ROC)
    df['volatility'] = df['Close'].pct_change().rolling(window=config.WINDOW).std()
    df['volatility_roc'] = df['volatility'].pct_change(periods=config.WINDOW)
    
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
    features_v2 = ['bb_width', 'atr_normalized', 'adx', 'volatility', 'volatility_roc']
    output_df = final_df[features_v2 + ['label']]
    
    print("\n--- 최종 데이터셋 정보 (v2) ---")
    print(f"전체 데이터 개수: {len(output_df)}")
    print("\n레이블 분포:")
    print(output_df['label'].value_counts())
    
    output_filename = "volatility_data_v2.csv"
    output_df.to_csv(output_filename)
    print(f"\n새로운 피처가 포함된 데이터셋을 '{output_filename}'으로 저장했습니다.")

if __name__ == "__main__":
    create_dataset_v2()
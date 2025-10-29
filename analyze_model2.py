import pandas as pd
import numpy as np
import pandas_ta as ta
import config # 설정 파일 임포트

def analyze_model2_conditions():
    """
    ADX와 불균형 지속 기간에 따른 모델 2 액션의 성공률을 분석합니다.
    """
    # ===================================
    # 1. train.py에서 데이터 준비 로직 복사
    # ===================================
    print("train.py와 동일한 조건으로 데이터를 준비합니다...")
    try:
        data = pd.read_csv("BTCUSDT_5m_raw_data.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("오류: BTCUSDT_5m_raw_data.csv 파일을 찾을 수 없습니다.")
        return

    # 특성 생성
    print("특성 및 이벤트를 생성합니다...")
    data['volatility'] = data['Close'].pct_change().rolling(config.WINDOW).std()
    ema_short = ta.ema(data['Close'], length=20)
    ema_long = ta.ema(data['Close'], length=100)
    data['price_vs_ema_short'] = data['Close'] / ema_short
    data['price_vs_ema_long'] = data['Close'] / ema_long
    data['ema_cross'] = ema_short / ema_long
    dmi_df = ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=config.WINDOW)
    data = data.join(dmi_df)
    data.rename(columns={f'ADX_{config.WINDOW}': 'adx', f'DMP_{config.WINDOW}': 'dmp', f'DMN_{config.WINDOW}': 'dmn'}, inplace=True)
    data.dropna(inplace=True)

    # 모델 2 이벤트 생성 (train.py와 동일)
    IMBALANCE_DURATIONS = [12, 24, 48, 96]
    model2_events = []
    for time_in_imbalance in IMBALANCE_DURATIONS:
        start_index = config.LOOKBACK_PERIOD + time_in_imbalance + config.WINDOW
        for i in range(start_index, len(data) - config.EVALUATION_WINDOW):
            initial_entry_price = data['Close'].iloc[i - config.LOOKBACK_PERIOD - time_in_imbalance]
            long_entry_price = initial_entry_price * (1 + config.FIXED_SPREAD / 2)
            short_entry_price = initial_entry_price * (1 - config.FIXED_SPREAD / 2)
            adx = data['adx'].iloc[i]
            plus_di = data['dmp'].iloc[i]
            minus_di = data['dmn'].iloc[i]
            params = {
                'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di,
                'ADX_TREND_THRESHOLD': config.ADX_TREND_THRESHOLD
            }
            action = config.get_model2_action_decision(params)
            if action is not None:
                if action == "RE_LOCK_FOR_SHORT":
                    long_size_imbalanced = 1.0 - config.PARTIAL_CLOSE_RATIO
                    short_size_imbalanced = 1.0
                elif action == "RE_LOCK_FOR_LONG":
                    long_size_imbalanced = 1.0
                    short_size_imbalanced = 1.0 - config.PARTIAL_CLOSE_RATIO
                current_price = data['Close'].iloc[i]
                pnl_larger_pos_at_relock = (short_entry_price - current_price) * short_size_imbalanced
                pnl_smaller_pos_at_relock = (current_price - long_entry_price) * long_size_imbalanced
                actual_outcome = pnl_larger_pos_at_relock + pnl_smaller_pos_at_relock
                price_at_eval = data['Close'].iloc[i + config.EVALUATION_WINDOW]
                hypothetical_pnl_larger_pos = (short_entry_price - price_at_eval) * short_size_imbalanced
                hypothetical_pnl_smaller_pos = (price_at_eval - long_entry_price) * long_size_imbalanced
                hypothetical_outcome = hypothetical_pnl_larger_pos + hypothetical_pnl_smaller_pos
                pnl_difference = actual_outcome - hypothetical_outcome
                label = 0
                if pnl_difference > 0: label = 1
                elif pnl_difference < 0: label = -1
                if label != 0:
                    event = {'adx': adx, 'time_in_imbalance': time_in_imbalance, 'label': label}
                    model2_events.append(event)

    if not model2_events:
        print("분석할 이벤트가 없습니다.")
        return

    events_df = pd.DataFrame(model2_events)
    print(f"총 {len(events_df)}개의 'RE_LOCK' 이벤트를 분석합니다.")

    # ===================================
    # 2. ADX 및 불균형 기간에 따른 교차 분석
    # ===================================
    print("\n--- ADX & 불균형 기간에 따른 재진입(RE_LOCK) 성공률 분석 ---\n")

    # ADX 값 구간 정의
    bins = [0, 15, 20, 25, 30, 40, 100]
    labels = ["0-15", "15-20", "20-25", "25-30", "30-40", "40+"]
    events_df['adx_bin'] = pd.cut(events_df['adx'], bins=bins, labels=labels, right=False)
    
    # 성공(1)만 필터링
    success_df = events_df[events_df['label'] == 1]

    # 피벗 테이블 생성: 총 이벤트 수
    total_pivot = pd.pivot_table(events_df, values='label', index='adx_bin', columns='time_in_imbalance', aggfunc='count', fill_value=0)
    
    # 피벗 테이블 생성: 성공 이벤트 수
    success_pivot = pd.pivot_table(success_df, values='label', index='adx_bin', columns='time_in_imbalance', aggfunc='count', fill_value=0)

    # 성공률 계산
    success_rate_pivot = (success_pivot / total_pivot * 100).replace(np.nan, 0)

    print("성공률(%) 테이블:")
    print(success_rate_pivot.to_string(float_format="%.2f"))
    print("\n총 이벤트 수 테이블:")
    print(total_pivot.to_string())

if __name__ == "__main__":
    analyze_model2_conditions()

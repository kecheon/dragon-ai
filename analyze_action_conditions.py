import pandas as pd
import numpy as np
import pandas_ta as ta

def analyze_conditions():
    """
    ADX 값에 따른 모델 1 액션의 성공/실패 분포를 분석합니다.
    """
    # ===================================
    # 1. train.py와 동일한 파라미터 및 데이터 로드
    # ===================================
    print("train.py와 동일한 조건으로 데이터를 준비합니다...")
    try:
        data = pd.read_csv("BTCUSDT_5m_raw_data.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("오류: BTCUSDT_5m_raw_data.csv 파일을 찾을 수 없습니다.")
        print("먼저 train.py를 한번 실행하여 데이터를 다운로드해야 합니다.")
        return

    WINDOW = 14
    PARTIAL_CLOSE_RATIO = 0.5
    EVALUATION_WINDOW = 24
    FIXED_SPREAD = 0.02
    LOOKBACK_PERIOD = 48
    ADX_TREND_THRESHOLD = 10 # 현재 train.py 설정값
    PROFIT_TAKE_PCT = 0.0046
    STOP_LOSS_PCT = 0.0021

    # ===================================
    # 2. train.py와 동일한 특성 생성 및 이벤트/레이블 생성
    # ===================================
    print("특성 및 이벤트를 생성합니다 (train.py 로직과 동일)...")
    
    # 특성 생성
    data['volatility'] = data['Close'].pct_change().rolling(WINDOW).std()
    dmi_df = ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=WINDOW)
    data = data.join(dmi_df)
    data.rename(columns={
        f'ADX_{WINDOW}': 'adx',
        f'DMP_{WINDOW}': 'dmp',
        f'DMN_{WINDOW}': 'dmn'
    }, inplace=True)
    data.dropna(inplace=True)

    # 모델 1 이벤트 생성
    model1_events = []
    for i in range(LOOKBACK_PERIOD + WINDOW, len(data) - EVALUATION_WINDOW):
        base_price = data['Close'].iloc[i - LOOKBACK_PERIOD]
        long_entry_price = base_price * (1 + FIXED_SPREAD / 2)
        short_entry_price = base_price * (1 - FIXED_SPREAD / 2)

        current_price = data['Close'].iloc[i]
        unrealized_long_pnl = (current_price - long_entry_price)
        unrealized_short_pnl = (short_entry_price - current_price)

        adx = data['adx'].iloc[i]
        plus_di = data['dmp'].iloc[i]
        minus_di = data['dmn'].iloc[i]

        action = None
        if unrealized_long_pnl > 0 and adx > ADX_TREND_THRESHOLD and plus_di < minus_di:
            action = "PARTIAL_CLOSE_LONG"
        elif unrealized_short_pnl > 0 and adx > ADX_TREND_THRESHOLD and plus_di > minus_di:
            action = "PARTIAL_CLOSE_SHORT"

        if action:
            pnl_at_action = unrealized_long_pnl + unrealized_short_pnl
            
            if action == "PARTIAL_CLOSE_LONG":
                remaining_long_size = 1.0 - PARTIAL_CLOSE_RATIO
                remaining_short_size = 1.0
            else:
                remaining_long_size = 1.0
                remaining_short_size = 1.0 - PARTIAL_CLOSE_RATIO

            label = 0
            profit_target = pnl_at_action + (base_price * PROFIT_TAKE_PCT)
            stop_loss = pnl_at_action - (base_price * STOP_LOSS_PCT)
            future_prices = data['Close'].iloc[i + 1 : i + 1 + EVALUATION_WINDOW]
            
            pnl_now = pnl_at_action
            for price in future_prices:
                pnl_now = ((price - long_entry_price) * remaining_long_size) + \
                          ((short_entry_price - price) * remaining_short_size)
                if pnl_now >= profit_target:
                    label = 1
                    break
                if pnl_now <= stop_loss:
                    label = -1
                    break
            
            if label == 0:
                if pnl_now > pnl_at_action:
                    label = 1
                elif pnl_now < pnl_at_action:
                    label = -1

            if label != 0:
                event = {'adx': adx, 'label': label}
                model1_events.append(event)

    if not model1_events:
        print("분석할 이벤트가 없습니다.")
        return

    events_df = pd.DataFrame(model1_events)
    print(f"총 {len(events_df)}개의 이벤트를 분석합니다.")

    # ===================================
    # 3. ADX 구간별 성공/실패 분석
    # ===================================
    print("\n--- ADX 값에 따른 성공률 분석 ---\n")

    # ADX 값 구간 정의
    bins = [0, 10, 15, 20, 25, 30, 40, 50, 100]
    labels = ["0-10", "10-15", "15-20", "20-25", "25-30", "30-40", "40-50", "50+"]
    events_df['adx_bin'] = pd.cut(events_df['adx'], bins=bins, labels=labels, right=False)

    # 각 구간별 통계 계산
    analysis_result = []
    for bin_label in labels:
        subset = events_df[events_df['adx_bin'] == bin_label]
        total_count = len(subset)
        if total_count == 0:
            continue
        
        success_count = subset[subset['label'] == 1].shape[0]
        failure_count = subset[subset['label'] == -1].shape[0]
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        analysis_result.append({
            "ADX Range": bin_label,
            "Total Events": total_count,
            "Success": success_count,
            "Failure": failure_count,
            "Success Rate (%)": f"{success_rate:.2f}"
        })

    # 결과 출력
    result_df = pd.DataFrame(analysis_result)
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    analyze_conditions()

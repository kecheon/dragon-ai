import pandas as pd
import numpy as np
import pandas_ta as ta
import config # 설정 파일 임포트

def analyze_pnl_impact():
    """
    액션 시점의 손익 상태(수익/손실)가 성공률에 미치는 영향을 분석합니다.
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

    # 이벤트 생성
    model1_events = []
    for i in range(config.LOOKBACK_PERIOD + config.WINDOW, len(data) - config.EVALUATION_WINDOW):
        base_price = data['Close'].iloc[i - config.LOOKBACK_PERIOD]
        long_entry_price = base_price * (1 + config.FIXED_SPREAD / 2)
        short_entry_price = base_price * (1 - config.FIXED_SPREAD / 2)
        current_price = data['Close'].iloc[i]
        unrealized_long_pnl = (current_price - long_entry_price)
        unrealized_short_pnl = (short_entry_price - current_price)
        adx = data['adx'].iloc[i]
        plus_di = data['dmp'].iloc[i]
        minus_di = data['dmn'].iloc[i]

        params = {
            'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di,
            'unrealized_long_pnl': unrealized_long_pnl, 'unrealized_short_pnl': unrealized_short_pnl,
            'ADX_TREND_THRESHOLD': config.ADX_TREND_THRESHOLD
        }
        action = config.get_action_decision(params)

        if action:
            pnl_at_action = unrealized_long_pnl + unrealized_short_pnl
            if action == "PARTIAL_CLOSE_LONG":
                remaining_long_size, remaining_short_size = 1.0 - config.PARTIAL_CLOSE_RATIO, 1.0
            else:
                remaining_long_size, remaining_short_size = 1.0, 1.0 - config.PARTIAL_CLOSE_RATIO
            
            label = 0
            profit_target = pnl_at_action + (base_price * config.PROFIT_TAKE_PCT)
            stop_loss = pnl_at_action - (base_price * config.STOP_LOSS_PCT)
            future_prices = data['Close'].iloc[i + 1 : i + 1 + config.EVALUATION_WINDOW]
            pnl_now = pnl_at_action
            for price in future_prices:
                pnl_before_fee = ((price - long_entry_price) * remaining_long_size) + ((short_entry_price - price) * remaining_short_size)
                fee = price * config.PARTIAL_CLOSE_RATIO * config.TRANSACTION_FEE_PCT
                pnl_now = pnl_before_fee - fee
                if pnl_now >= profit_target: label = 1; break
                if pnl_now <= stop_loss: label = -1; break
            if label == 0:
                if pnl_now > pnl_at_action: label = 1
                elif pnl_now < pnl_at_action: label = -1

            if label != 0:
                event = {'pnl_at_action': pnl_at_action, 'label': label}
                model1_events.append(event)

    if not model1_events:
        print("분석할 이벤트가 없습니다.")
        return

    events_df = pd.DataFrame(model1_events)
    print(f"총 {len(events_df)}개의 이벤트를 분석합니다.")

    # ===================================
    # 2. 손익 상태별 성공률 분석
    # ===================================
    print("\n--- 액션 시점의 손익 상태에 따른 성공률 분석 ---\n")

    profitable_events = events_df[events_df['pnl_at_action'] > 0]
    unprofitable_events = events_df[events_df['pnl_at_action'] <= 0]

    groups = {
        "수익 상태에서 액션 (pnl > 0)": profitable_events,
        "손실 상태에서 액션 (pnl <= 0)": unprofitable_events
    }

    analysis_result = []
    for name, df_group in groups.items():
        total_count = len(df_group)
        if total_count == 0:
            continue
        
        success_count = df_group[df_group['label'] == 1].shape[0]
        failure_count = df_group[df_group['label'] == -1].shape[0]
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        analysis_result.append({
            "분류": name,
            "Total Events": total_count,
            "Success": success_count,
            "Failure": failure_count,
            "Success Rate (%)": f"{success_rate:.2f}"
        })

    result_df = pd.DataFrame(analysis_result)
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    analyze_pnl_impact()

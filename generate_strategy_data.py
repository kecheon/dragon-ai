import pandas as pd
import numpy as np
import pandas_ta as ta
import config
from tqdm import tqdm

def generate_data():
    """
    '존버', '재진입', '전체 손절' 세 가지 액션의 결과를 시뮬레이션하여, 
    최적의 행동을 레이블링하는 학습 데이터를 생성합니다.
    """
    # ===================================
    # 1. 데이터 로드 및 특성 생성
    # ===================================
    print("데이터 로드를 시작합니다...")
    try:
        data = pd.read_csv("BTCUSDT_5m_raw_data.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("오류: BTCUSDT_5m_raw_data.csv 파일을 찾을 수 없습니다.")
        return

    print("특성 생성을 시작합니다...")
    data['volatility'] = data['Close'].pct_change().rolling(config.WINDOW).std()
    ema_short = ta.ema(data['Close'], length=20)
    ema_long = ta.ema(data['Close'], length=100)
    data['price_vs_ema_short'] = data['Close'] / ema_short
    data['price_vs_ema_long'] = data['Close'] / ema_long
    data['ema_cross'] = ema_short / ema_long
    returns = data['Close'].pct_change()
    mean_returns = returns.rolling(config.WINDOW).mean()
    std_returns = returns.rolling(config.WINDOW).std()
    data['z_score'] = (returns - mean_returns) / std_returns
    dmi_df = ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=config.WINDOW)
    data = data.join(dmi_df)
    data.rename(columns={f'ADX_{config.WINDOW}': 'adx', f'DMP_{config.WINDOW}': 'dmp', f'DMN_{config.WINDOW}': 'dmn'}, inplace=True)
    data.dropna(inplace=True)
    print("데이터 로드 및 특성 생성 완료.")

    # ===================================
    # 2. 시뮬레이션 및 레이블링
    # ===================================
    print("\n3가지 액션 시뮬레이션 및 레이블링을 시작합니다... (시간이 매우 오래 소요됩니다)")
    strategy_events = []
    iterator = tqdm(range(config.LOOKBACK_PERIOD + config.WINDOW, len(data) - config.MAX_EVALUATION_WINDOW), desc="Simulating Actions")
    for i in iterator:
        # --- 1. 불균형 상태 정의 ---
        base_price = data['Close'].iloc[i - config.LOOKBACK_PERIOD]
        long_entry_price = base_price * (1 + config.FIXED_SPREAD / 2)
        short_entry_price = base_price * (1 - config.FIXED_SPREAD / 2)
        current_price = data['Close'].iloc[i]
        pnl_long_full = (current_price - long_entry_price)
        pnl_short_full = (short_entry_price - current_price)
        pnl_at_action = pnl_long_full + pnl_short_full

        plus_di = data['dmp'].iloc[i]
        minus_di = data['dmn'].iloc[i]

        imbalance_type = None
        if pnl_at_action <= 0 and plus_di < minus_di:
            imbalance_type = "LONG_IMBALANCE"
        elif pnl_at_action <= 0 and plus_di > minus_di:
            imbalance_type = "SHORT_IMBALANCE"
        
        if not imbalance_type:
            continue

        # --- 2. 각 액션의 미래 결과 시뮬레이션 (최대 기간까지 추적) ---
        future_indices = range(i + 1, min(i + 1 + config.MAX_EVALUATION_WINDOW, len(data)))
        global_stop_loss_pnl = pnl_at_action - (base_price * 0.02)
        liquidation_pnl_level = pnl_at_action + (base_price * config.LIQUIDATION_THRESHOLD_PCT)

        # --- 액션 1: 존버(Hold) 시뮬레이션 ---
        outcome_hold = 0
        if imbalance_type == "LONG_IMBALANCE":
            hold_long_size, hold_short_size = 1.0 - config.PARTIAL_CLOSE_RATIO, 1.0
        else: # SHORT_IMBALANCE
            hold_long_size, hold_short_size = 1.0, 1.0 - config.PARTIAL_CLOSE_RATIO
        
        for j in future_indices:
            price = data['Close'].iloc[j]
            pnl_now = (price - long_entry_price) * hold_long_size + (short_entry_price - price) * hold_short_size
            if pnl_now >= 0:
                outcome_hold = 1; break
            if pnl_now <= liquidation_pnl_level: # 청산 레벨을 먼저 확인
                outcome_hold = -2; break
            if pnl_now <= global_stop_loss_pnl:
                outcome_hold = -1; break

        # --- 액션 2: 재진입(Re-lock) 시뮬레이션 ---
        outcome_relock = 0
        if imbalance_type == "LONG_IMBALANCE":
            new_long_entry_price = current_price
            for j in future_indices:
                price = data['Close'].iloc[j]
                pnl_now = ((price - long_entry_price) * (1.0 - config.PARTIAL_CLOSE_RATIO)) + \
                          ((price - new_long_entry_price) * config.PARTIAL_CLOSE_RATIO) + \
                          ((short_entry_price - price) * 1.0)
                if pnl_now >= 0:
                    outcome_relock = 1; break
                if pnl_now <= liquidation_pnl_level: # 청산 레벨을 먼저 확인
                    outcome_relock = -2; break
                if pnl_now <= global_stop_loss_pnl:
                    outcome_relock = -1; break
        else: # SHORT_IMBALANCE
            new_short_entry_price = current_price
            for j in future_indices:
                price = data['Close'].iloc[j]
                pnl_now = ((price - long_entry_price) * 1.0) + \
                          ((short_entry_price - price) * (1.0 - config.PARTIAL_CLOSE_RATIO)) + \
                          ((new_short_entry_price - price) * config.PARTIAL_CLOSE_RATIO)
                if pnl_now >= 0:
                    outcome_relock = 1; break
                if pnl_now <= liquidation_pnl_level: # 청산 레벨을 먼저 확인
                    outcome_relock = -2; break
                if pnl_now <= global_stop_loss_pnl:
                    outcome_relock = -1; break

        # --- 액션 3: 전체 손절(Cut Loss) ---
        outcome_cutloss = -1

        # --- 3. 최적 액션 레이블링 (청산 케이스 추가) ---
        outcomes = {"HOLD": outcome_hold, "RE_LOCK": outcome_relock, "CUT_LOSS": outcome_cutloss}
        
        # HOLD와 RE_LOCK 모두 청산으로 이어지면, 해당 상황을 "LIQUIDATION"으로 레이블링
        if outcome_hold == -2 and outcome_relock == -2:
            best_action = "LIQUIDATION"
        else:
            # 그 외의 경우, 가장 결과가 좋은 액션을 선택
            best_action = max(outcomes, key=outcomes.get)

        # --- 4. 학습 데이터 저장 ---
        event_features = data.loc[data.index[i], :].to_dict()
        event_features['label'] = best_action
        event_features['pnl_at_action'] = pnl_at_action
        strategy_events.append(event_features)

    # ===================================
    # 3. 최종 데이터셋 저장
    # ===================================
    if not strategy_events:
        print("생성된 이벤트가 없습니다.")
        return

    final_df = pd.DataFrame(strategy_events)
    file_path = "strategy_dataset_v3.csv"
    print(f"\n총 {len(final_df)}개의 학습 데이터를 생성했습니다. '{file_path}' 파일로 저장합니다.")
    final_df.to_csv(file_path)
    print("완료.")

if __name__ == "__main__":
    generate_data()
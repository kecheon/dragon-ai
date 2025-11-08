import pandas as pd
import numpy as np
import pandas_ta as ta

# ===================================
# === 1. 파라미터 설정 ===
# ===================================
# --- 전략 파라미터 ---
WINDOW = 14
ADX_TREND_THRESHOLD = 15
ZSCORE_LOOKBACK_PERIOD = 14
ZSCORE_THRESHOLD = 1.5
EMA_LONG_PERIOD = 100

# --- 백테스트 파라미터 ---
DATA_FILE = "data.csv"
INITIAL_BALANCE_USDT = 100000
LEVERAGE = 10
BASE_MARGIN_QUANTITY_BTC = 0.01 # 첫 진입 시 증거금 계산 기준 BTC 수량

# --- 손익 및 수수료 파라미터 ---
PROFIT_TAKE_PCT = 0.02 # 단일 포지션 익절
# STOP_LOSS_PCT는 이 전략에서 사용되지 않음
TRANSACTION_FEE_PCT = 0.0004

# ===================================
# === 2. 액션 결정 함수 ===
# ===================================
def get_action_decision(params):
    adx = params['adx']
    plus_di = params['plus_di']
    minus_di = params['minus_di']
    z_score = params['z_score']
    close_price = params['close']
    long_ema = params['long_ema']
    adx_threshold = params['ADX_TREND_THRESHOLD']
    z_score_threshold = params['ZSCORE_THRESHOLD']
    action = None
    is_trending = adx > adx_threshold
    is_bullish_momentum = plus_di > minus_di and z_score > z_score_threshold
    is_bearish_momentum = minus_di > plus_di and z_score < -z_score_threshold
    is_above_long_ema = close_price > long_ema
    is_below_long_ema = close_price < long_ema

    if is_trending and is_bullish_momentum and is_above_long_ema:
        action = "ENTER_LONG"
    elif is_trending and is_bearish_momentum and is_below_long_ema:
        action = "ENTER_SHORT"
    return action

# ===================================
# === 3. 데이터 준비 및 지표 계산 ===
# ===================================
print("--- 데이터 로딩 및 지표 계산 시작 ---")
try:
    df = pd.read_csv(DATA_FILE, index_col="Timestamp", parse_dates=True)
    print(f"데이터 로딩 완료: {len(df)} 행")
except FileNotFoundError:
    print(f"오류: 데이터 파일 '{DATA_FILE}'을 찾을 수 없습니다.")
    exit()

adx_df = ta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=WINDOW)
adx_df.rename(columns={f'ADX_{WINDOW}': 'adx', f'DMP_{WINDOW}': 'plus_di', f'DMN_{WINDOW}': 'minus_di'}, inplace=True)
df = pd.concat([df, adx_df], axis=1)
df['returns'] = df['Close'].pct_change()
df['ret_x_vol'] = df['returns'] * df['Volume']
sum_ret_x_vol = df['ret_x_vol'].rolling(window=ZSCORE_LOOKBACK_PERIOD).sum()
sum_vol = df['Volume'].rolling(window=ZSCORE_LOOKBACK_PERIOD).sum()
weighted_mean = sum_ret_x_vol / sum_vol
dev_sq = (df['returns'] - weighted_mean)**2
dev_sq_x_vol = dev_sq * df['Volume']
sum_dev_sq_x_vol = dev_sq_x_vol.rolling(window=ZSCORE_LOOKBACK_PERIOD).sum()
weighted_variance = sum_dev_sq_x_vol / sum_vol
weighted_std = np.sqrt(weighted_variance)
df['volume_zscore'] = (df['returns'] - weighted_mean) / weighted_std
df[f'ema_{EMA_LONG_PERIOD}'] = ta.ema(df["Close"], length=EMA_LONG_PERIOD)
df.dropna(inplace=True)
print("모든 지표 계산 완료.")

# ===================================
# === 4. 백테스팅 시뮬레이션 실행 ===
# ===================================
print("--- 백테스팅 시뮬레이션 시작 ---")

usdt_balance = INITIAL_BALANCE_USDT
open_positions = []
trade_groups = []
equity_history = []

for i in range(1, len(df)):
    prev_row = df.iloc[i-1]
    current_row = df.iloc[i]

    # --- 1. 포지션 종료 로직 ---
    if open_positions:
        # --- 1A. 강제 청산 확인 (최우선) ---
        total_margin_at_risk = sum(p['margin'] for p in open_positions)
        total_unrealized_pnl = sum(((current_row['Close'] - p['entry_price']) * p['quantity'] if p['type'] == 'LONG' else (p['entry_price'] - current_row['Close']) * p['quantity']) for p in open_positions)
        current_trade_equity = total_margin_at_risk + total_unrealized_pnl

        if current_trade_equity <= 0:
            usdt_balance -= total_margin_at_risk
            trade_groups.append({'positions': open_positions.copy(), 'exit_time': current_row.name, 'pnl_usdt': -total_margin_at_risk, 'exit_reason': 'LIQUIDATION'})
            open_positions.clear()
            equity_history.append(usdt_balance)
            continue

        # --- 1B. 시나리오별 종료 조건 확인 ---
        # 시나리오 1: 단일 포지션 -> 익절(TP)만 적용. 손절 없음.
        if len(open_positions) == 1:
            pos = open_positions[0]
            exit_price = 0
            exit_reason = ""
            if pos['type'] == 'LONG':
                if current_row["High"] >= pos['entry_price'] * (1 + PROFIT_TAKE_PCT):
                    exit_price, exit_reason = pos['entry_price'] * (1 + PROFIT_TAKE_PCT), "TAKE_PROFIT"
            else: # SHORT
                if current_row["Low"] <= pos['entry_price'] * (1 - PROFIT_TAKE_PCT):
                    exit_price, exit_reason = pos['entry_price'] * (1 - PROFIT_TAKE_PCT), "TAKE_PROFIT"
            
            if exit_price > 0:
                pnl = (exit_price - pos['entry_price']) * pos['quantity'] if pos['type'] == 'LONG' else (pos['entry_price'] - exit_price) * pos['quantity']
                fee = exit_price * pos['quantity'] * TRANSACTION_FEE_PCT
                usdt_balance += pnl - fee
                trade_groups.append({'positions': open_positions.copy(), 'exit_time': current_row.name, 'pnl_usdt': pnl - fee, 'exit_reason': exit_reason})
                open_positions.clear()
                equity_history.append(usdt_balance)
                continue

        # 시나리오 2: 다중(헤징) 포지션 -> 전체 수익 전환 시 동시 청산
        elif len(open_positions) > 1:
            if total_unrealized_pnl > 0:
                final_pnl = 0
                for pos in open_positions:
                    exit_price = current_row['Close']
                    pnl = (exit_price - pos['entry_price']) * pos['quantity'] if pos['type'] == 'LONG' else (pos['entry_price'] - exit_price) * pos['quantity']
                    fee = exit_price * pos['quantity'] * TRANSACTION_FEE_PCT
                    final_pnl += pnl - fee
                
                usdt_balance += final_pnl
                trade_groups.append({'positions': open_positions.copy(), 'exit_time': current_row.name, 'pnl_usdt': final_pnl, 'exit_reason': 'HEDGE_PROFIT_EXIT'})
                open_positions.clear()
                equity_history.append(usdt_balance)
                continue

    # --- 2. 신규 진입 / 헤징 진입 로직 ---
    params = {
        'adx': prev_row['adx'], 'plus_di': prev_row['plus_di'], 'minus_di': prev_row['minus_di'],
        'z_score': prev_row['volume_zscore'], 'close': prev_row['Close'], f'long_ema': prev_row[f'ema_{EMA_LONG_PERIOD}'],
        'ADX_TREND_THRESHOLD': ADX_TREND_THRESHOLD, 'ZSCORE_THRESHOLD': ZSCORE_THRESHOLD
    }
    action = get_action_decision(params)

    if action is not None:
        new_pos_type = action.split('_')[1]
        entry_price = current_row["Open"]
        if not open_positions:
            base_quantity = BASE_MARGIN_QUANTITY_BTC
            trade_quantity = base_quantity * LEVERAGE
            margin_needed = entry_price * base_quantity
            if usdt_balance > margin_needed:
                fee = entry_price * trade_quantity * TRANSACTION_FEE_PCT
                usdt_balance -= fee
                open_positions.append({'type': new_pos_type, 'entry_price': entry_price, 'quantity': trade_quantity, 'margin': margin_needed})
        else:
            last_pos = open_positions[-1]
            if new_pos_type != last_pos['type']:
                trade_quantity = last_pos['quantity'] * 2
                base_quantity = trade_quantity / LEVERAGE
                margin_needed = entry_price * base_quantity
                if usdt_balance > margin_needed + sum(p['margin'] for p in open_positions):
                    fee = entry_price * trade_quantity * TRANSACTION_FEE_PCT
                    usdt_balance -= fee
                    open_positions.append({'type': new_pos_type, 'entry_price': entry_price, 'quantity': trade_quantity, 'margin': margin_needed})

    # --- 3. 매 캔들마다 총 자산(Equity) 기록 ---
    current_equity = usdt_balance
    if open_positions:
        unrealized_pnl = sum(((current_row['Close'] - p['entry_price']) * p['quantity'] if p['type'] == 'LONG' else (p['entry_price'] - current_row['Close']) * p['quantity']) for p in open_positions)
        current_equity += unrealized_pnl
    equity_history.append(current_equity)

print("--- 백테스팅 시뮬레이션 종료 ---")

# ===================================
# === 5. 결과 분석 및 리포트 ===
# ===================================
print("\n--- 시뮬레이션 결과 분석 ---")
if not trade_groups:
    print("시뮬레이션 기간 동안 거래가 발생하지 않았습니다.")
else:
    groups_df = pd.DataFrame(trade_groups)
    equity_series = pd.Series(equity_history, index=df.index[1:])
    cumulative_max_equity = equity_series.cummax()
    drawdown = (equity_series - cumulative_max_equity) / cumulative_max_equity
    max_drawdown = drawdown.min()
    total_pnl_usdt = usdt_balance - INITIAL_BALANCE_USDT
    wins = groups_df[groups_df["pnl_usdt"] > 0]
    losses = groups_df[groups_df["pnl_usdt"] <= 0]
    liquidations = groups_df[groups_df["exit_reason"] == "LIQUIDATION"]
    win_rate = len(wins) / len(groups_df) if len(groups_df) > 0 else 0

    print(f"초기 자본 (USDT): {INITIAL_BALANCE_USDT:,.2f}")
    print(f"최종 자본 (USDT): {usdt_balance:,.2f}")
    print(f"총 손익 (USDT): {total_pnl_usdt:,.2f} ({total_pnl_usdt/INITIAL_BALANCE_USDT:.2%})")
    print(f"최대 자본 하락폭 (MDD): {max_drawdown:.2%}")
    print("-" * 30)
    print(f"총 거래 그룹 수: {len(groups_df)}")
    print(f"수익 그룹 수: {len(wins)}")
    print(f"손실 그룹 수: {len(losses)}")
    print(f"청산 그룹 수: {len(liquidations)}")
    print(f"그룹 승률: {win_rate:.2%}")
    print("-" * 30)
    print(f"그룹 평균 수익 (USDT): {wins['pnl_usdt'].mean():,.2f}" if not wins.empty else "N/A")
    print(f"그룹 평균 손실 (USDT): {losses['pnl_usdt'].mean():,.2f}" if not losses.empty else "N/A")

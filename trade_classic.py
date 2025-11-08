import pandas as pd
import numpy as np
import pandas_ta as ta

# ===================================
# === 1. 파라미터 설정 ===
# ===================================
# --- 전략 파라미터 ---
WINDOW = 14
ADX_TREND_THRESHOLD = 25
ZSCORE_LOOKBACK_PERIOD = 20
ZSCORE_THRESHOLD = 2.0
EMA_LONG_PERIOD = 100

# --- 백테스트 파라미터 ---
DATA_FILE = "data.csv"
INITIAL_BALANCE_USDT = 100000
LEVERAGE = 10
BASE_MARGIN_QUANTITY_BTC = 0.01 # 첫 진입 시 증거금 계산 기준 BTC 수량

# --- 손익 및 수수료 파라미터 ---
PROFIT_TAKE_PCT = 0.01 # 단일 포지션 익절
STOP_LOSS_PCT = 0.01 # 단일 포지션 손절
TRANSACTION_FEE_PCT = 0.0004

# ===================================
# === 2. 액션 결정 함수 ===
# ===================================
def get_action_decision(params):
    adx = params['adx']
    plus_di = params['plus_di']
    minus_di = params['minus_di']
    plus_di2 = params['plus_di2']
    minus_di2 = params['minus_di2']
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

    if is_trending and is_bullish_momentum:
        action = "ENTER_SHORT"
    elif is_trending and is_bearish_momentum:
        action = "ENTER_LONG"
    # if plus_di2 > minus_di2 and plus_di < minus_di and is_trending and is_below_long_ema:
    #     action = "ENTER_SHORT"
    # elif plus_di2 < minus_di2 and plus_di > minus_di and is_trending and is_above_long_ema:
    #     action = "ENTER_LONG"

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
open_position = None
trades = []
equity_history = []

for i in range(1, len(df)):
    pprev_row = df.iloc[i-10]
    prev_row = df.iloc[i-1]
    current_row = df.iloc[i]

    # --- 1. 포지션 종료 로직 ---
    if open_position:
        pos = open_position
        exit_price = 0
        exit_reason = ""

        # --- 1A. 강제 청산 확인 ---
        unrealized_pnl = (current_row['Close'] - pos['entry_price']) * pos['quantity'] if pos['type'] == 'LONG' else (pos['entry_price'] - current_row['Close']) * pos['quantity']
        current_equity = pos['margin'] + unrealized_pnl
        if current_equity <= 0:
            usdt_balance -= pos['margin']
            trades.append({'position': pos, 'exit_time': current_row.name, 'pnl_usdt': -pos['margin'], 'exit_reason': 'LIQUIDATION'})
            open_position = None
            equity_history.append(usdt_balance)
            continue

        # --- 1B. 익절 또는 손절 조건 확인 ---
        if pos['type'] == 'LONG':
            if current_row["High"] >= pos['entry_price'] * (1 + PROFIT_TAKE_PCT):
                exit_price, exit_reason = pos['entry_price'] * (1 + PROFIT_TAKE_PCT), "TAKE_PROFIT"
            elif current_row["Low"] <= pos['entry_price'] * (1 - STOP_LOSS_PCT):
                exit_price, exit_reason = pos['entry_price'] * (1 - STOP_LOSS_PCT), "STOP_LOSS"
        else: # SHORT
            if current_row["Low"] <= pos['entry_price'] * (1 - PROFIT_TAKE_PCT):
                exit_price, exit_reason = pos['entry_price'] * (1 - PROFIT_TAKE_PCT), "TAKE_PROFIT"
            elif current_row["High"] >= pos['entry_price'] * (1 + STOP_LOSS_PCT):
                exit_price, exit_reason = pos['entry_price'] * (1 + STOP_LOSS_PCT), "STOP_LOSS"
        
        if exit_price > 0:
            pnl = (exit_price - pos['entry_price']) * pos['quantity'] if pos['type'] == 'LONG' else (pos['entry_price'] - exit_price) * pos['quantity']
            fee = exit_price * pos['quantity'] * TRANSACTION_FEE_PCT
            usdt_balance += pnl - fee
            trades.append({'position': pos, 'exit_time': current_row.name, 'pnl_usdt': pnl - fee, 'exit_reason': exit_reason})
            open_position = None
            equity_history.append(usdt_balance)
            continue

    # --- 2. 신규 진입 로직 ---
    if not open_position:
        params = {
            'adx': prev_row['adx'], 'plus_di': prev_row['plus_di'], 'minus_di': prev_row['minus_di'],
            'adx2': pprev_row['adx'], 'plus_di2': pprev_row['plus_di'], 'minus_di2': pprev_row['minus_di'],
            'z_score': prev_row['volume_zscore'], 'close': prev_row['Close'], f'long_ema': prev_row[f'ema_{EMA_LONG_PERIOD}'],
            'ADX_TREND_THRESHOLD': ADX_TREND_THRESHOLD, 'ZSCORE_THRESHOLD': ZSCORE_THRESHOLD
        }
        action = get_action_decision(params)

        if action is not None:
            new_pos_type = action.split('_')[1]
            entry_price = current_row["Open"]
            base_quantity = BASE_MARGIN_QUANTITY_BTC
            trade_quantity = base_quantity * LEVERAGE
            margin_needed = entry_price * base_quantity
            if usdt_balance > margin_needed:
                fee = entry_price * trade_quantity * TRANSACTION_FEE_PCT
                usdt_balance -= fee
                open_position = {'type': new_pos_type, 'entry_price': entry_price, 'quantity': trade_quantity, 'margin': margin_needed}

    # --- 3. 매 캔들마다 총 자산(Equity) 기록 ---
    current_equity = usdt_balance
    if open_position:
        unrealized_pnl = ((current_row['Close'] - open_position['entry_price']) * open_position['quantity'] if open_position['type'] == 'LONG' else (open_position['entry_price'] - current_row['Close']) * open_position['quantity'])
        current_equity += unrealized_pnl
    equity_history.append(current_equity)

print("--- 백테스팅 시뮬레이션 종료 ---")

# ===================================
# === 5. 결과 분석 및 리포트 ===
# ===================================
print("\n--- 시뮬레이션 결과 분석 ---")
if not trades:
    print("시뮬레이션 기간 동안 거래가 발생하지 않았습니다.")
else:
    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity_history, index=df.index[1:])
    cumulative_max_equity = equity_series.cummax()
    drawdown = (equity_series - cumulative_max_equity) / cumulative_max_equity
    max_drawdown = drawdown.min()
    total_pnl_usdt = usdt_balance - INITIAL_BALANCE_USDT
    wins = trades_df[trades_df["pnl_usdt"] > 0]
    losses = trades_df[trades_df["pnl_usdt"] <= 0]
    liquidations = trades_df[trades_df["exit_reason"] == "LIQUIDATION"]
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0

    print(f"초기 자본 (USDT): {INITIAL_BALANCE_USDT:,.2f}")
    print(f"최종 자본 (USDT): {usdt_balance:,.2f}")
    print(f"총 손익 (USDT): {total_pnl_usdt:,.2f} ({total_pnl_usdt/INITIAL_BALANCE_USDT:.2%})")
    print(f"최대 자본 하락폭 (MDD): {max_drawdown:.2%}")
    print("-" * 30)
    print(f"총 거래 수: {len(trades_df)}")
    print(f"수익 거래 수: {len(wins)}")
    print(f"손실 거래 수: {len(losses)}")
    print(f"청산 거래 수: {len(liquidations)}")
    print(f"승률: {win_rate:.2%}")
    print("-" * 30)
    print(f"평균 수익 (USDT): {wins['pnl_usdt'].mean():,.2f}" if not wins.empty else "N/A")
    print(f"평균 손실 (USDT): {losses['pnl_usdt'].mean():,.2f}" if not losses.empty else "N/A")

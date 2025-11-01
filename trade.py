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
INITIAL_BALANCE_USDT = 10000
LEVERAGE = 10
BASE_MARGIN_QUANTITY_BTC = 0.001 # 증거금 계산 기준이 되는 BTC 수량

# --- 손익 및 수수료 파라미터 ---
PROFIT_TAKE_PCT = 0.01
STOP_LOSS_PCT = 0.01
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

position = None
entry_price = 0
usdt_balance = INITIAL_BALANCE_USDT
trades = []
equity_history = []
margin_used_usdt = 0
effective_trade_quantity_btc = 0

for i in range(1, len(df)):
    prev_row = df.iloc[i-1]
    current_row = df.iloc[i]

    # --- 1. 포지션 종료(Exit) 로직 ---
    if position is not None:
        exit_price = 0
        liquidation_price = 0
        exit_reason = ""

        # 청산 가격 계산
        if position == "LONG":
            liquidation_price = entry_price * (1 - 1/LEVERAGE)
            if current_row["Low"] <= liquidation_price:
                exit_price = liquidation_price
                exit_reason = "LIQUIDATION"
                print(f"청산 가격: {liquidation_price}")
                print(f"진입 가격: {entry_price}")
        elif position == "SHORT":
            liquidation_price = entry_price * (1 + 1/LEVERAGE)
            if current_row["High"] >= liquidation_price:
                exit_price = liquidation_price
                exit_reason = "LIQUIDATION"
                print(f"청산 가격: {liquidation_price}")
                print(f"진입 가격: {entry_price}")
        
        # TP/SL 가격 계산 (청산되지 않은 경우)
        if exit_price == 0:
            if position == "LONG":
                if current_row["Low"] <= entry_price * (1 - STOP_LOSS_PCT):
                    exit_price = entry_price * (1 - STOP_LOSS_PCT)
                    exit_reason = "STOP_LOSS"
                elif current_row["High"] >= entry_price * (1 + PROFIT_TAKE_PCT):
                    exit_price = entry_price * (1 + PROFIT_TAKE_PCT)
                    exit_reason = "TAKE_PROFIT"
            elif position == "SHORT":
                if current_row["High"] >= entry_price * (1 + STOP_LOSS_PCT):
                    exit_price = entry_price * (1 + STOP_LOSS_PCT)
                    exit_reason = "STOP_LOSS"
                elif current_row["Low"] <= entry_price * (1 - PROFIT_TAKE_PCT):
                    exit_price = entry_price * (1 - PROFIT_TAKE_PCT)
                    exit_reason = "TAKE_PROFIT"

        # 포지션 종료 처리
        if exit_price > 0:
            if exit_reason == "LIQUIDATION":
                pnl_usdt = -margin_used_usdt
            else:
                pnl_usdt = (exit_price - entry_price) * effective_trade_quantity_btc if position == "LONG" else (entry_price - exit_price) * effective_trade_quantity_btc
                exit_fee = exit_price * effective_trade_quantity_btc * TRANSACTION_FEE_PCT
                pnl_usdt -= exit_fee

            usdt_balance += pnl_usdt
            trades.append({
                "type": position, "entry_time": df.index[i-1], "exit_time": current_row.name,
                "entry_price": entry_price, "exit_price": exit_price, "pnl_usdt": pnl_usdt, "exit_reason": exit_reason
            })
            position = None
            entry_price = 0
            margin_used_usdt = 0
            effective_trade_quantity_btc = 0

    # --- 2. 포지션 진입(Entry) 로직 ---
    if position is None:
        params = {
            'adx': prev_row['adx'], 'plus_di': prev_row['plus_di'], 'minus_di': prev_row['minus_di'],
            'z_score': prev_row['volume_zscore'], 'close': prev_row['Close'], f'long_ema': prev_row[f'ema_{EMA_LONG_PERIOD}'],
            'ADX_TREND_THRESHOLD': ADX_TREND_THRESHOLD, 'ZSCORE_THRESHOLD': ZSCORE_THRESHOLD
        }
        action = get_action_decision(params)

        if action is not None:
            margin_used_usdt = current_row["Open"] * BASE_MARGIN_QUANTITY_BTC
            if usdt_balance > margin_used_usdt:
                position = action.split("_")[1]
                entry_price = current_row["Open"]
                effective_trade_quantity_btc = BASE_MARGIN_QUANTITY_BTC * LEVERAGE
                entry_fee = entry_price * effective_trade_quantity_btc * TRANSACTION_FEE_PCT
                usdt_balance -= entry_fee

    # --- 3. 매 캔들마다 총 자산(Equity) 기록 ---
    current_equity = usdt_balance
    if position is not None:
        unrealized_pnl = (current_row["Close"] - entry_price) * effective_trade_quantity_btc if position == "LONG" else (entry_price - current_row["Close"]) * effective_trade_quantity_btc
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
    losses = trades_df[trades_df["pnl_usdt"] < 0]
    liquidations = trades_df[trades_df["exit_reason"] == "LIQUIDATION"]
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
    total_profit = wins["pnl_usdt"].sum()
    total_loss = abs(losses["pnl_usdt"].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

    print(f"초기 자본 (USDT): {INITIAL_BALANCE_USDT:,.2f}")
    print(f"최종 자본 (USDT): {usdt_balance:,.2f}")
    print(f"총 손익 (USDT): {total_pnl_usdt:,.2f} ({total_pnl_usdt/INITIAL_BALANCE_USDT:.2%})")
    print("-" * 30)
    print(f"총 거래 횟수: {len(trades_df)}")
    print(f"승리 횟수: {len(wins)}")
    print(f"패배 횟수: {len(losses)}")
    print(f"청산 횟수: {len(liquidations)}")
    print(f"승률 (청산 제외): {win_rate:.2%}")
    print("-" * 30)
    print(f"평균 수익 (USDT): {wins['pnl_usdt'].mean():,.2f}" if not wins.empty else "N/A")
    print(f"평균 손실 (USDT): {losses['pnl_usdt'].mean():,.2f}" if not losses.empty else "N/A")
    print(f"수익 팩터 (Profit Factor): {profit_factor:.2f}")
    print(f"최대 자본 하락폭 (MDD): {max_drawdown:.2%}")

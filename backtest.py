import pandas as pd
import numpy as np
import xgboost as xgb
import config
import pandas_ta as ta

# ===================================
# === 1. 설정 (Configuration) ===
# ===================================
# --- 계좌 설정 ---
INITIAL_BALANCE = 10000.0
POSITION_QUANTITY = 0.01
TRANSACTION_FEE_PCT = 0.0004 # 거래 수수료 (0.04%)

# --- 파일 및 모델 경로 ---
DATA_FILE = 'BTCUSDT_5m_raw_data.csv'
MODEL_FILE = 'model_x.json'
PREDICTION_THRESHOLD = 0.8 # 모델 예측 확신도 임계값

# ===================================
# === 2. 데이터 및 모델 로딩 ===
# ===================================
print("--- Loading Data and Model ---")
# 데이터 로딩
try:
    data = pd.read_csv(DATA_FILE, index_col='Timestamp', parse_dates=True)
    print(f"Data loaded successfully: {len(data)} rows")
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found.")
    exit()

# 모델 로딩
model_x = xgb.XGBClassifier()
try:
    model_x.load_model(MODEL_FILE)
    print("Model 'model_x.json' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ===================================
# === 3. 특성 생성 함수 ===
# ===================================
def calculate_features(df):
    """주어진 데이터프레임에 모델 학습에 사용된 특성들을 계산하여 추가합니다."""
    df['volatility'] = df['Close'].pct_change().rolling(config.WINDOW).std()
    
    ema_short = ta.ema(df['Close'], length=20)
    ema_long = ta.ema(df['Close'], length=100)
    df['price_vs_ema_short'] = df['Close'] / ema_short
    df['price_vs_ema_long'] = df['Close'] / ema_long
    df['ema_cross'] = ema_short / ema_long
    
    returns = df['Close'].pct_change()
    mean_returns = returns.rolling(config.WINDOW).mean()
    std_returns = returns.rolling(config.WINDOW).std()
    df['z_score'] = (returns - mean_returns) / std_returns
    
    dmi_df = ta.adx(high=df['High'], low=df['Low'], close=df['Close'], length=config.WINDOW)
    df = df.join(dmi_df)
    df.rename(columns={
        f'ADX_{config.WINDOW}': 'adx',
        f'DMP_{config.WINDOW}': 'dmp',
        f'DMN_{config.WINDOW}': 'dmn'
    }, inplace=True)
    
    # train_x.py에서 추가된 특성
    # 백테스터에서는 spread_before_action을 동적으로 계산해야 함
    df['spread_before_action'] = 0 # 임시 값
    
    return df

print("--- Calculating Features ---")
data = calculate_features(data)
data.dropna(inplace=True)
print("Features calculated.")


# ===================================
# === 4. 백테스팅 시뮬레이션 ===
# ===================================
print("-- Starting Backtest Simulation --")

# --- 계좌 및 포지션 변수 초기화 ---
balance = INITIAL_BALANCE
equity_curve = [] # 빈 리스트로 시작
trade_log = []

# 포지션 정보
long_pos_size = 0.0
avg_long_price = 0.0
short_pos_size = 0.0
avg_short_price = 0.0

# --- 시뮬레이션 루프 ---
for i in range(1, len(data)):
    current_price = data['Close'].iloc[i]
    current_time = data.index[i]

    # 포지션 진입/관리 로직
    if long_pos_size == 0 and short_pos_size == 0:
        # 초기 진입
        avg_long_price = current_price * (1 + config.FIXED_SPREAD / 2)
        long_pos_size = POSITION_QUANTITY
        balance -= avg_long_price * long_pos_size * TRANSACTION_FEE_PCT
        
        avg_short_price = current_price * (1 - config.FIXED_SPREAD / 2)
        short_pos_size = POSITION_QUANTITY
        balance -= avg_short_price * short_pos_size * TRANSACTION_FEE_PCT
        
        trade_log.append(f"{current_time}: Initial Grid Entry -> LONG {long_pos_size} @ {avg_long_price:.2f}, SHORT {short_pos_size} @ {avg_short_price:.2f}")
    
    else: # 포지션 보유 중
        unrealized_pnl = (current_price - avg_long_price) * long_pos_size + (avg_short_price - current_price) * short_pos_size
        
        if unrealized_pnl > 0:
            # 전체 포지션 익절
            balance += unrealized_pnl
            balance -= (current_price * long_pos_size * TRANSACTION_FEE_PCT)
            balance -= (current_price * short_pos_size * TRANSACTION_FEE_PCT)
            trade_log.append(f"{current_time}: Global Profit Take -> Closed all for profit {unrealized_pnl:.2f}")
            long_pos_size, avg_long_price, short_pos_size, avg_short_price = 0.0, 0.0, 0.0, 0.0
        
        else:
            # 모델 기반 헤징 롤링
            # 새로운 방향 결정 규칙: "규칙 A: 손실이 더 큰 포지션을 롤링한다."
            unrealized_long_pnl_val = (current_price - avg_long_price) * long_pos_size
            unrealized_short_pnl_val = (avg_short_price - current_price) * short_pos_size
            
            action = None
            if unrealized_long_pnl_val < unrealized_short_pnl_val:
                action = "PARTIAL_CLOSE_LONG"
            else:
                action = "PARTIAL_CLOSE_SHORT"

            # 이제 모델에게 이 action을 실행할지 말지 물어봄
            features_df = data.iloc[[i]].copy()
            features_df['spread_before_action'] = abs(avg_long_price - avg_short_price)
            
            # 신호 확인용 PNL 계산 (학습 때와 동일한 기준 수량으로)
            unrealized_long_pnl_for_signal = (current_price - avg_long_price) * POSITION_QUANTITY
            unrealized_short_pnl_for_signal = (avg_short_price - current_price) * POSITION_QUANTITY
            features_df['unrealized_long_pnl'] = unrealized_long_pnl_for_signal
            features_df['unrealized_short_pnl'] = unrealized_short_pnl_for_signal
            features_df['total_unrealized_pnl'] = unrealized_long_pnl_for_signal + unrealized_short_pnl_for_signal

            feature_columns = [
                'unrealized_long_pnl', 'unrealized_short_pnl', 'total_unrealized_pnl',
                'spread_before_action', 'volatility', 'adx', 'dmp', 'dmn',
                'price_vs_ema_short', 'price_vs_ema_long', 'ema_cross', 'z_score'
            ]
            X_live = features_df[feature_columns].values

            # 모델 예측 (확률 기반 필터링)
            probabilities = model_x.predict_proba(X_live)[0]
            
            # '성공' 클래스에 대한 확신도가 임계값을 넘을 때만 액션 실행
            if probabilities[1] > PREDICTION_THRESHOLD:
                    trade_log.append(f"{current_time}: Model predicted SUCCESS for {action} (Prob: {probabilities[1]:.2f}). Executing roll.")
                    closed_amount = POSITION_QUANTITY * config.PARTIAL_CLOSE_RATIO
                    
                    if action == "PARTIAL_CLOSE_LONG" and long_pos_size >= closed_amount:
                        realized_pnl_on_close = (current_price - avg_long_price) * closed_amount
                        balance += realized_pnl_on_close
                        balance -= (current_price * closed_amount * TRANSACTION_FEE_PCT)
                        long_pos_size -= closed_amount
                        
                        avg_short_price = ((avg_short_price * short_pos_size) + (current_price * closed_amount)) / (short_pos_size + closed_amount)
                        short_pos_size += closed_amount
                        balance -= (current_price * closed_amount * TRANSACTION_FEE_PCT)

                    elif action == "PARTIAL_CLOSE_SHORT" and short_pos_size >= closed_amount:
                        realized_pnl_on_close = (avg_short_price - current_price) * closed_amount
                        balance += realized_pnl_on_close
                        balance -= (current_price * closed_amount * TRANSACTION_FEE_PCT)
                        short_pos_size -= closed_amount

                        avg_long_price = ((avg_long_price * long_pos_size) + (current_price * closed_amount)) / (long_pos_size + closed_amount)
                        long_pos_size += closed_amount
                        balance -= (current_price * closed_amount * TRANSACTION_FEE_PCT)
                    else:
                        # Log a more informative message
                        if action: # Only log if there was a signal to begin with
                            trade_log.append(f"{current_time}: Model predicted FAILURE for {action} (Prob: {probabilities[1]:.2f}). Action AVOIDED.")

    # 매 루프의 끝에서 현재 자산 상태를 한 번만 기록
    final_unrealized_pnl = (current_price - avg_long_price) * long_pos_size + (avg_short_price - current_price) * short_pos_size
    equity = balance + final_unrealized_pnl
    equity_curve.append(equity)

print("--- Backtest Simulation Finished ---")

# ===================================
# === 5. 결과 리포팅 ===
# ===================================
print("--- Generating Performance Report ---")

# 최종 PnL 계산
if not equity_curve:
    final_balance = balance
else:
    final_balance = equity_curve[-1]
final_pnl = final_balance - INITIAL_BALANCE
total_return_pct = (final_pnl / INITIAL_BALANCE) * 100

# Equity Curve 데이터프레임 생성
if len(equity_curve) > 0:
    equity_df = pd.DataFrame({'equity': equity_curve}, index=data.index[1:])

    # 최대 낙폭 (Max Drawdown) 계산
    peak = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - peak) / peak
    max_drawdown = drawdown.min()

    # 샤프 비율 (Sharpe Ratio) 계산
    daily_returns = equity_df['equity'].resample('D').last().pct_change().dropna()
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
    else:
        sharpe_ratio = 0.0

    print(f"Backtest Period: {equity_df.index[0].date()} to {equity_df.index[-1].date()}")
else:
    max_drawdown = 0.0
    sharpe_ratio = 0.0
    print("Backtest Period: Not enough data to calculate.")

# 거래 횟수
total_trades = len([log for log in trade_log if "Executing roll" in log or "Global Profit Take" in log])

print("\n" + "="*30)
print("PERFORMANCE METRICS")
print("="*30)
print(f"Initial Balance:    {INITIAL_BALANCE:12.2f}")
print(f"Final Balance:        {final_balance:12.2f}")
print(f"Total PnL:            {final_pnl:12.2f} ({total_return_pct:.2f}%)")
print(f"Max Drawdown:         {max_drawdown:12.2%}")
print(f"Sharpe Ratio:         {sharpe_ratio:12.2f}")
print(f"Total Trades:         {total_trades:12d}")
print("="*30)

# 마지막 거래 로그 몇 개 출력
print("\n--- Last 20 Trades ---")
for log in trade_log[-20:]:
    print(log)


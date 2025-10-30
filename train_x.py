import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report

import os
import random
import config
random.seed(42)

# ====================================
# === 1. 데이터 로딩 (캐시 우선)  ===
# ====================================
SYMBOL = 'BTCUSDT'
TIMEFRAME = '5m'
START_DATE = '2025-01-01T00:00:00Z'
RAW_DATA_FILE = f"{SYMBOL}_{TIMEFRAME}_raw_data.csv"

# 캐시 파일이 존재하면 파일에서 로드, 없으면 CCXT로 가져와서 저장
if os.path.exists(RAW_DATA_FILE):
    print(f"'{RAW_DATA_FILE}' 파일에서 데이터를 로드합니다...")
    data = pd.read_csv(RAW_DATA_FILE, index_col='Timestamp', parse_dates=True)
    print(f"Total candles loaded from file: {len(data)}")
else:
    print(f"'{RAW_DATA_FILE}' 파일이 없습니다. CCXT에서 데이터를 가져옵니다...")
    exchange = ccxt.binanceus({'options': {'defaultType': 'future'}})
    exchange.load_markets()

    since = exchange.parse8601(START_DATE)
    all_ohlcv = []

    print(f"Fetching {TIMEFRAME} candles for {SYMBOL} from {START_DATE}...")
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv:
                break
            last_ts = ohlcv[-1][0]
            all_ohlcv.extend(ohlcv)
            since = last_ts + 1
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    data = pd.DataFrame(all_ohlcv, columns=['Timestamp','Open','High','Low','Close','Volume'])
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
    data.set_index('Timestamp', inplace=True)
    print(f"Total candles fetched: {len(data)}")
    
    print(f"데이터를 '{RAW_DATA_FILE}' 파일로 저장합니다...")
    data.to_csv(RAW_DATA_FILE)



# ===================================
# === 3. 특성(Feature) 생성       ===
# ===================================

# Volatility
data['volatility'] = data['Close'].pct_change().rolling(config.WINDOW).std()

# EMA (Exponential Moving Average)
ema_short = ta.ema(data['Close'], length=20)
ema_long = ta.ema(data['Close'], length=100)
data['price_vs_ema_short'] = data['Close'] / ema_short
data['price_vs_ema_long'] = data['Close'] / ema_long
data['ema_cross'] = ema_short / ema_long

# Z-score of returns
returns = data['Close'].pct_change()
mean_returns = returns.rolling(config.WINDOW).mean()
std_returns = returns.rolling(config.WINDOW).std()
data['z_score'] = (returns - mean_returns) / std_returns

# DMI (Directional Movement Index)
dmi_df = ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=config.WINDOW)
data = data.join(dmi_df)
data.rename(columns={
    f'ADX_{config.WINDOW}': 'adx',
    f'DMP_{config.WINDOW}': 'dmp',
    f'DMN_{config.WINDOW}': 'dmn'
}, inplace=True)
# 특성 계산 후 초기 NaN 값이 있는 행들을 제거
data.dropna(inplace=True)

print("특성 생성 완료")


# ===================================================
# === 4. Model X (Hedging Rolling) 데이터 생성 ===
# ===================================================
print("\n--- Generating Data for Model X: Hedging Rolling ---")

model_x_events = []

# 전체 데이터를 스캔하여 헤징 롤링 이벤트 탐색
for i in range(config.LOOKBACK_PERIOD + config.WINDOW, len(data) - config.EVALUATION_WINDOW):
    # --- 1. 가상 초기 진입 상태 설정 ---
    base_price = data['Close'].iloc[i - config.LOOKBACK_PERIOD]
    
    # 초기 포지션: Long 1, Short 1
    initial_long_price = base_price * (1 + config.FIXED_SPREAD / 2)
    initial_short_price = base_price * (1 - config.FIXED_SPREAD / 2)
    
    long_pos_size = 1.0
    short_pos_size = 1.0
    avg_long_price = initial_long_price
    avg_short_price = initial_short_price

    # --- 2. 현재 상태 계산 (액션 판단 기준) ---
    current_price = data['Close'].iloc[i]
    unrealized_long_pnl = (current_price - avg_long_price) * long_pos_size
    unrealized_short_pnl = (avg_short_price - current_price) * short_pos_size

    # --- 3. 액션 조건 확인 ---
    adx = data['adx'].iloc[i]
    plus_di = data['dmp'].iloc[i]
    minus_di = data['dmn'].iloc[i]

    params = {
        'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di,
        'unrealized_long_pnl': unrealized_long_pnl, 'unrealized_short_pnl': unrealized_short_pnl,
        'ADX_TREND_THRESHOLD': config.ADX_TREND_THRESHOLD
    }
    action = config.get_action_decision(params)

    # --- 4. 이벤트 발생 시, 액션 시뮬레이션 및 레이블 계산 ---
    if action:
        pnl_at_action = unrealized_long_pnl + unrealized_short_pnl
        spread_before_action = abs(avg_long_price - avg_short_price)
        
        # 액션 시뮬레이션: 한쪽은 줄이고 반대쪽은 늘림
        closed_amount = config.PARTIAL_CLOSE_RATIO # 가정: 수량은 1.0 기준
        
        if action == "PARTIAL_CLOSE_LONG":
            # Long 포지션 부분 청산, Short 포지션 물타기
            long_pos_size -= closed_amount
            
            # Short 포지션 평균단가 재계산
            avg_short_price = ((avg_short_price * short_pos_size) + (current_price * closed_amount)) / (short_pos_size + closed_amount)
            short_pos_size += closed_amount

        else: # PARTIAL_CLOSE_SHORT
            # Short 포지션 부분 청산, Long 포지션 물타기
            short_pos_size -= closed_amount

            # Long 포지션 평균단가 재계산
            avg_long_price = ((avg_long_price * long_pos_size) + (current_price * closed_amount)) / (long_pos_size + closed_amount)
            long_pos_size += closed_amount

        # --- 5. 레이블링 (개선된 방식) --- 
        # 두 가지 시나리오의 최종 손익을 비교하여 레이블 결정
        
        label = 0
        future_window = data['Close'].iloc[i + 1 : i + 1 + config.EVALUATION_WINDOW]

        # 시나리오 A: '헤징 롤링' 액션을 취한 경우의 최종 손익
        final_pnl_with_action = 0
        for price in future_window:
            pnl = ((price - avg_long_price) * long_pos_size) + \
                  ((avg_short_price - price) * short_pos_size)
            if pnl > 0: # 전체 익절 조건 달성
                final_pnl_with_action = pnl # 익절 시점의 pnl을 최종 결과로
                break
            final_pnl_with_action = pnl # 마지막 캔들까지의 pnl 업데이트

        # 시나리오 B: 아무 액션도 취하지 않은 경우의 최종 손익
        final_pnl_without_action = 0
        for price in future_window:
            pnl = ((price - initial_long_price) * 1.0) + \
                  ((initial_short_price - price) * 1.0)
            if pnl > 0: # 전체 익절 조건 달성
                final_pnl_without_action = pnl
                break
            final_pnl_without_action = pnl

        # 최종 비교
        if final_pnl_with_action > final_pnl_without_action:
            label = 1  # 액션을 취한 것이 더 나은 결과를 가져옴
        else:
            label = -1 # 액션을 취한 것이 더 나쁘거나 같은 결과를 가져옴

        if label != 0:
            event = {
                'unrealized_long_pnl': unrealized_long_pnl,
                'unrealized_short_pnl': unrealized_short_pnl,
                'total_unrealized_pnl': pnl_at_action,
                'spread_before_action': spread_before_action,
                'volatility': data['volatility'].iloc[i],
                'adx': adx,
                'dmp': plus_di,
                'dmn': minus_di,
                'price_vs_ema_short': data['price_vs_ema_short'].iloc[i],
                'price_vs_ema_long': data['price_vs_ema_long'].iloc[i],
                'ema_cross': data['ema_cross'].iloc[i],
                'z_score': data['z_score'].iloc[i],
                'label': label
            }
            model_x_events.append(event)

model_x_df = pd.DataFrame(model_x_events)
print(f"Found {len(model_x_df)} 'Hedging Rolling' events.")


# ===================================
# === 5. 모델 학습 (Model X) ===
# ===================================
print("\n--- Training Model X: Hedging Rolling Model ---")
if not model_x_df.empty:
    model_x_df['label_binary'] = model_x_df['label'].replace({-1: 0, 1: 1})
    
    print(f"{len(model_x_df)}개의 데이터로 모델 X를 학습합니다.")

    features = [
        'unrealized_long_pnl', 
        'unrealized_short_pnl',
        'total_unrealized_pnl',
        'spread_before_action',
        'volatility', 
        'adx', 
        'dmp', 
        'dmn',
        'price_vs_ema_short',
        'price_vs_ema_long',
        'ema_cross',
        'z_score'
    ]
    X = model_x_df[features].values
    y = model_x_df['label_binary'].values

    if len(np.unique(y)) < 2:
        print("모델 X를 학습하기에 레이블 종류가 충분하지 않습니다 (성공/실패 중 하나만 존재).")
    else:
        # 시계열 데이터 분할
        split_index = int(len(X) * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        print("Model X Train label distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
        print("Model X Validation label distribution:", dict(zip(*np.unique(y_val, return_counts=True))))

        # 최종 모델 학습 (최적 scale_pos_weight 사용)
        final_weight = 0.3
        print(f"\n{'='*20} Training final model with scale_pos_weight: {final_weight} {'='*20}")

        # XGBoost 모델 학습
        model_x = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss',
            n_estimators=150,
            learning_rate=0.05,
            max_depth=7,
            gamma=0.2,
            subsample=0.8,
            use_label_encoder=False,
            scale_pos_weight=final_weight
        )
        model_x.fit(X_train, y_train)

        y_pred = model_x.predict(X_val)
        
        print("--- Final Model Classification Report ---")
        print(classification_report(y_val, y_pred, target_names=['Failure (0)', 'Success (1)']))

        # 최종 모델 저장
        model_x_file = "model_x.json"
        model_x.save_model(model_x_file)
        print(f"\nFinal model saved: {model_x_file}")
else:
    print("모델 X를 학습할 데이터가 없습니다.")


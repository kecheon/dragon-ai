import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import xgboost as xgb

import os
import random
import config
random.seed(42)

# ====================================
# === 1. 데이터 로딩 (캐시 우선)  ===
# ====================================
SYMBOL = 'BTCUSDT'
TIMEFRAME = '5m'
START_DATE = '2022-01-01T00:00:00Z'
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
# === 4. 모델 1 (부분 청산) 데이터 생성 (이벤트 스캐너 방식) ===
# ===================================================
print("\n--- Generating Data for Model 1: Partial Exit ---")

model1_events = []

# 전체 데이터를 스캔하여 부분 청산 이벤트 탐색
for i in range(config.LOOKBACK_PERIOD + config.WINDOW, len(data) - config.EVALUATION_WINDOW):
    # --- 1. 가상 진입 설정 ---
    # 1. 과거 특정 시점의 가격을 기준으로 삼습니다.
    base_price = data['Close'].iloc[i - config.LOOKBACK_PERIOD]

    long_entry_price = base_price * (1 + config.FIXED_SPREAD / 2)
    short_entry_price = base_price * (1 - config.FIXED_SPREAD / 2)

    # --- 2. 현재 상태 계산 ---
    current_price = data['Close'].iloc[i]
    unrealized_long_pnl = (current_price - long_entry_price) # 가치 기준
    unrealized_short_pnl = (short_entry_price - current_price) # 가치 기준

    # --- 3. 부분 청산 조건 확인 ---
    adx = data['adx'].iloc[i]
    plus_di = data['dmp'].iloc[i]
    minus_di = data['dmn'].iloc[i]

    # config.py의 함수를 사용하여 액션 결정
    params = {
        'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di,
        'unrealized_long_pnl': unrealized_long_pnl, 'unrealized_short_pnl': unrealized_short_pnl,
        'ADX_TREND_THRESHOLD': config.ADX_TREND_THRESHOLD
    }
    action = config.get_action_decision(params)

    # --- 4. 이벤트 발생 시, 레이블 계산 및 데이터 기록 ---
    if action:
        pnl_at_action = unrealized_long_pnl + unrealized_short_pnl
        
        if action == "PARTIAL_CLOSE_LONG":
            remaining_long_size = 1.0 - config.PARTIAL_CLOSE_RATIO
            remaining_short_size = 1.0
        else: # PARTIAL_CLOSE_SHORT
            remaining_long_size = 1.0
            remaining_short_size = 1.0 - config.PARTIAL_CLOSE_RATIO

        # --- 동적 레이블링 (Triple-Barrier Method) ---
        label = 0  # 0: 시간 종료(Timeout) 
        
        # PNL 변화량을 기준으로 익절/손절 라인 설정
        profit_target = pnl_at_action + (base_price * config.PROFIT_TAKE_PCT)
        stop_loss = pnl_at_action - (base_price * config.STOP_LOSS_PCT)

        future_prices = data['Close'].iloc[i + 1 : i + 1 + config.EVALUATION_WINDOW]
        
        pnl_now = pnl_at_action # 시간 종료 시 비교를 위한 초기값
        for price in future_prices:
            pnl_before_fee = ((price - long_entry_price) * remaining_long_size) + \
                             ((short_entry_price - price) * remaining_short_size)
            fee = price * config.PARTIAL_CLOSE_RATIO * config.TRANSACTION_FEE_PCT
            pnl_now = pnl_before_fee - fee
            
            if pnl_now >= profit_target:
                label = 1  # 익절
                break
            if pnl_now <= stop_loss:
                label = -1  # 손절
                break
        
        # 시간 종료 시 (익절/손절 라인 미도달), 마지막 가격으로 판단
        if label == 0:
            if pnl_now > pnl_at_action:
                label = 1
            elif pnl_now < pnl_at_action:
                label = -1

        if label != 0: # 중립이 아닌 경우만 데이터로 사용
            event = {
                'unrealized_long_pnl': unrealized_long_pnl,
                'unrealized_short_pnl': unrealized_short_pnl,
                'total_unrealized_pnl': pnl_at_action,
                'total_position_size': 2.0,
                'volatility': data['volatility'].iloc[i],
                'adx': adx,
                'dmp': plus_di,
                'dmn': minus_di,
                # EMA features added
                'price_vs_ema_short': data['price_vs_ema_short'].iloc[i],
                'price_vs_ema_long': data['price_vs_ema_long'].iloc[i],
                'ema_cross': data['ema_cross'].iloc[i],
                'label': label
            }
            model1_events.append(event)

model1_df = pd.DataFrame(model1_events)
print(f"Found {len(model1_df)} 'PARTIAL_CLOSE' events.")

# ===================================================
# === 5. 모델 2 (상태 복구) 데이터 생성 (이벤트 스캐너 방식, 고급 특성 추가) ===
# ===================================================
print("\n--- Generating Data for Model 2: State Restoration (with advanced features) ---")

# 테스트할 불균형 지속 기간 (캔들 수)
IMBALANCE_DURATIONS = [12, 24, 48, 96] # 1시간, 2시간, 4시간, 8시간

model2_events = []

# 각 불균형 지속 기간에 대해 루프 실행
for time_in_imbalance in IMBALANCE_DURATIONS:
    
    # 전체 데이터를 스캔하여 재진입 이벤트 탐색
    start_index = config.LOOKBACK_PERIOD + time_in_imbalance + config.WINDOW
    for i in range(start_index, len(data) - config.EVALUATION_WINDOW):
        
        # --- 1. 가상 과거 상태 설정 ---
        initial_entry_price = data['Close'].iloc[i - config.LOOKBACK_PERIOD - time_in_imbalance]
        long_entry_price = initial_entry_price * (1 + config.FIXED_SPREAD / 2)
        short_entry_price = initial_entry_price * (1 - config.FIXED_SPREAD / 2)
        
        # --- 2. 현재 시점에서 '재진입' 조건 확인 ---
        # (우선 숏 포지션이 더 많은 경우만 가정)
        long_size_imbalanced = 1.0 - config.PARTIAL_CLOSE_RATIO
        short_size_imbalanced = 1.0

        adx = data['adx'].iloc[i]
        plus_di = data['dmp'].iloc[i]
        minus_di = data['dmn'].iloc[i]

        action = None
        # 숏 포지션이 더 많은 상태에서, 상승 추세(숏에 불리)가 나타나면 RE_LOCK 시도
        if adx > config.ADX_TREND_THRESHOLD and plus_di > minus_di:
            action = "RE_LOCK"

        # --- 3. 이벤트 발생 시, 레이블 계산 및 데이터 기록 ---
        if action:
            current_price = data['Close'].iloc[i]
            
            # 'RE_LOCK'에 대한 레이블 계산 (가상 시나리오 비교)
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
                event = {
                    # 기존 특성
                    'unrealized_long_pnl': pnl_smaller_pos_at_relock,
                    'unrealized_short_pnl': pnl_larger_pos_at_relock,
                    'total_unrealized_pnl': actual_outcome,
                    'total_position_size': long_size_imbalanced + short_size_imbalanced,
                    'volatility': data['volatility'].iloc[i],
                    'adx': adx,
                    'dmp': plus_di,
                    'dmn': minus_di,
                    # 신규 특성
                    'time_in_imbalance': time_in_imbalance,
                    'pnl_of_larger_position': pnl_larger_pos_at_relock,
                    # 레이블
                    'label': label
                }
                model2_events.append(event)

model2_df = pd.DataFrame(model2_events)
print(f"Found {len(model2_df)} 'RE_LOCK' events across {len(IMBALANCE_DURATIONS)} scenarios.")


# ===================================
# === 6. 모델 학습 (이진 분류) ===
# ===================================

# --- 모델 1 (부분 청산 결정 모델) 학습 ---
print("\n--- Training Model 1: Partial Exit Model ---")
if not model1_df.empty:
    model1_df['label_binary'] = model1_df['label'].replace({-1: 0, 1: 1})
    
    print(f"{len(model1_df)}개의 데이터로 모델 1을 학습합니다.")

    features = [
        'unrealized_long_pnl', 
        'unrealized_short_pnl',
        'total_unrealized_pnl',
        'total_position_size',
        'volatility', 
        'adx', 
        'dmp', 
        'dmn',
        # EMA features
        'price_vs_ema_short',
        'price_vs_ema_long',
        'ema_cross'
    ]
    X1 = model1_df[features].values
    y1 = model1_df['label_binary'].values

    if len(np.unique(y1)) < 2:
        print("모델 1을 학습하기에 레이블 종류가 충분하지 않습니다 (성공/실패 중 하나만 존재).")
    else:
        # 시계열 데이터 분할: 시간 순서를 유지하기 위해 마지막 20%를 검증 세트로 사용
        split_index1 = int(len(X1) * 0.8)
        X1_train, X1_val = X1[:split_index1], X1[split_index1:]
        y1_train, y1_val = y1[:split_index1], y1[split_index1:]

        # stratify=y1 옵션이 없어졌으므로, 분할 후 레이블 분포를 확인하는 것이 중요합니다.
        print("Model 1 Train label distribution:", dict(zip(*np.unique(y1_train, return_counts=True))))
        print("Model 1 Validation label distribution:", dict(zip(*np.unique(y1_val, return_counts=True))))

        # 클래스 불균형 처리를 위해 scale_pos_weight 계산
        scale_pos_weight1 = np.sum(y1_train == 0) / np.sum(y1_train == 1)
        print(f"Model 1 scale_pos_weight: {scale_pos_weight1:.4f}")

        # 최적 하이퍼파라미터 적용
        model1 = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss',
            n_estimators=100,         # Tuned
            learning_rate=0.1,        # Tuned
            max_depth=9,              # Tuned
            gamma=0.3,                # Tuned
            subsample=0.8,            # Kept original
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight1
        )
        model1.fit(X1_train, y1_train)

        accuracy1 = model1.score(X1_val, y1_val)
        print(f"Model 1 Validation Accuracy: {accuracy1:.4f}")

        model1_file = "strategic_exit_model.json"
        model1.save_model(model1_file)
        print(f"Model 1 saved: {model1_file}")
else:
    print("모델 1을 학습할 데이터가 없습니다.")

# --- 모델 2 (상태 복구 모델) 학습 ---
print("\n--- Training Model 2: State Restoration Model ---")
if not model2_df.empty:
    model2_df['label_binary'] = model2_df['label'].replace({-1: 0, 1: 1})

    print(f"{len(model2_df)}개의 'RE_LOCK' 액션 데이터로 모델 2를 학습합니다.")

    features_m2 = [
        'unrealized_long_pnl', 
        'unrealized_short_pnl',
        'total_unrealized_pnl',
        'total_position_size',
        'volatility', 
        'adx', 
        'dmp', 
        'dmn',
        'time_in_imbalance',
        'pnl_of_larger_position'
    ]
    X2 = model2_df[features_m2].values
    y2 = model2_df['label_binary'].values

    if len(np.unique(y2)) < 2:
        print("모델 2를 학습하기에 레이블 종류가 충분하지 않습니다 (성공/실패 중 하나만 존재).")
    else:
        # 시계열 데이터 분할: 시간 순서를 유지하기 위해 마지막 20%를 검증 세트로 사용
        split_index2 = int(len(X2) * 0.8)
        X2_train, X2_val = X2[:split_index2], X2[split_index2:]
        y2_train, y2_val = y2[:split_index2], y2[split_index2:]

        # stratify=y2 옵션이 없어졌으므로, 분할 후 레이블 분포를 확인하는 것이 중요합니다.
        print("Model 2 Train label distribution:", dict(zip(*np.unique(y2_train, return_counts=True))))
        print("Model 2 Validation label distribution:", dict(zip(*np.unique(y2_val, return_counts=True))))

        model2 = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=200,
            learning_rate=0.15,
            max_depth=9,
            gamma=0.1,
            subsample=0.8,
            use_label_encoder=False
        )
        model2.fit(X2_train, y2_train)

        accuracy2 = model2.score(X2_val, y2_val)
        print(f"Model 2 Validation Accuracy: {accuracy2:.4f}")

        model2_file = "state_restoration_model.json"
        model2.save_model(model2_file)
        print(f"Model 2 saved: {model2_file}")
else:
    print("모델 2를 학습할 데이터가 없습니다.")


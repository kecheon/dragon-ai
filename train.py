import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os

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
# === 2. 파라미터 설정 ===
# ===================================
WINDOW = 14
PARTIAL_CLOSE_RATIO = 0.5
EVALUATION_WINDOW = 12
initial_spread_pct = 0.015
LOOKBACK_PERIOD = 24
ADX_TREND_THRESHOLD = 20

# ===================================
# === 3. 특성(Feature) 생성       ===
# ===================================

# Volatility
data['volatility'] = data['Close'].pct_change().rolling(WINDOW).std().fillna(0)

# DMI (Directional Movement Index)
dmi_df = ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=WINDOW)
data = data.join(dmi_df)
data.rename(columns={
    f'ADX_{WINDOW}': 'adx',
    f'DMP_{WINDOW}': 'dmp',
    f'DMN_{WINDOW}': 'dmn'
}, inplace=True)
data.fillna({
    'adx': 25,
    'dmp': data['dmp'].mean(),
    'dmn': data['dmn'].mean()
}, inplace=True)

print("특성 생성 완료")


# ===================================================
# === 4. 모델 1 (부분 청산) 데이터 생성 (이벤트 스캐너 방식) ===
# ===================================================
print("\n--- Generating Data for Model 1: Partial Exit ---")

model1_events = []

# 전체 데이터를 스캔하여 부분 청산 이벤트 탐색
for i in range(LOOKBACK_PERIOD + WINDOW, len(data) - EVALUATION_WINDOW):
    # --- 1. 가상 진입 설정 ---
    entry_price = data['Close'].iloc[i - LOOKBACK_PERIOD]
    long_entry_price = entry_price * (1 + initial_spread_pct / 2)
    short_entry_price = entry_price * (1 - initial_spread_pct / 2)

    # --- 2. 현재 상태 계산 ---
    current_price = data['Close'].iloc[i]
    unrealized_long_pnl = (current_price - long_entry_price) # 가치 기준
    unrealized_short_pnl = (short_entry_price - current_price) # 가치 기준

    # --- 3. 부분 청산 조건 확인 ---
    adx = data['adx'].iloc[i]
    plus_di = data['dmp'].iloc[i]
    minus_di = data['dmn'].iloc[i]

    action = None
    if unrealized_long_pnl > 0 and adx > ADX_TREND_THRESHOLD and plus_di < minus_di:
        action = "PARTIAL_CLOSE_LONG"
    elif unrealized_short_pnl > 0 and adx > ADX_TREND_THRESHOLD and plus_di > minus_di:
        action = "PARTIAL_CLOSE_SHORT"

    # --- 4. 이벤트 발생 시, 레이블 계산 및 데이터 기록 ---
    if action:
        pnl_at_action = unrealized_long_pnl + unrealized_short_pnl
        
        if action == "PARTIAL_CLOSE_LONG":
            remaining_long_size = 1.0 - PARTIAL_CLOSE_RATIO
            remaining_short_size = 1.0
        else: # PARTIAL_CLOSE_SHORT
            remaining_long_size = 1.0
            remaining_short_size = 1.0 - PARTIAL_CLOSE_RATIO

        price_at_eval = data['Close'].iloc[i + EVALUATION_WINDOW]
        pnl_at_eval = ((price_at_eval - long_entry_price) * remaining_long_size) + \
                      ((short_entry_price - price_at_eval) * remaining_short_size)
        
        pnl_change = pnl_at_eval - pnl_at_action
        
        label = 0
        if pnl_change > 0: label = 1
        elif pnl_change < 0: label = -1

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
                'label': label
            }
            model1_events.append(event)

model1_df = pd.DataFrame(model1_events)
print(f"Found {len(model1_df)} 'PARTIAL_CLOSE' events.")

# ===================================================
# === 5. 모델 2 (상태 복구) 데이터 생성 (이벤트 스캐너 방식) ===
# ===================================================
print("\n--- Generating Data for Model 2: State Restoration ---")

# 하이퍼파라미터: 부분 청산이 일어났다고 가정할 과거 시점
PARTIAL_CLOSE_LOOKBACK = 24 

model2_events = []

# 전체 데이터를 스캔하여 재진입 이벤트 탐색
for i in range(LOOKBACK_PERIOD + PARTIAL_CLOSE_LOOKBACK + WINDOW, len(data) - EVALUATION_WINDOW):
    
    # --- 1. 가상 과거 상태 설정 ---
    initial_entry_price = data['Close'].iloc[i - LOOKBACK_PERIOD - PARTIAL_CLOSE_LOOKBACK]
    long_entry_price = initial_entry_price * (1 + initial_spread_pct / 2)
    short_entry_price = initial_entry_price * (1 - initial_spread_pct / 2)
    
    # --- 2. 현재 시점에서 '재진입' 조건 확인 ---
    # 두 가지 불균형 시나리오(롱이 많거나, 숏이 많거나)를 모두 고려해야 하나, 우선 숏이 더 많은 경우만 가정하여 로직 구현
    long_size_imbalanced = 1.0 - PARTIAL_CLOSE_RATIO
    short_size_imbalanced = 1.0

    adx = data['adx'].iloc[i]
    plus_di = data['dmp'].iloc[i]
    minus_di = data['dmn'].iloc[i]

    action = None
    # 숏 포지션이 더 많은 상태에서, 상승 추세(숏에 불리)가 나타나면 RE_LOCK 시도
    if adx > ADX_TREND_THRESHOLD and plus_di > minus_di:
        action = "RE_LOCK"

    # --- 3. 이벤트 발생 시, 레이블 계산 및 데이터 기록 ---
    if action:
        current_price = data['Close'].iloc[i]
        
        # 'RE_LOCK'에 대한 레이블 계산 (가상 시나리오 비교)
        unrealized_long_pnl_at_relock = (current_price - long_entry_price) * long_size_imbalanced
        unrealized_short_pnl_at_relock = (short_entry_price - current_price) * short_size_imbalanced
        actual_outcome = unrealized_long_pnl_at_relock + unrealized_short_pnl_at_relock

        price_at_eval = data['Close'].iloc[i + EVALUATION_WINDOW]
        hypothetical_unrealized_long_pnl = (price_at_eval - long_entry_price) * long_size_imbalanced
        hypothetical_unrealized_short_pnl = (short_entry_price - price_at_eval) * short_size_imbalanced
        hypothetical_outcome = hypothetical_unrealized_long_pnl + hypothetical_unrealized_short_pnl
        
        pnl_difference = actual_outcome - hypothetical_outcome
        
        label = 0
        if pnl_difference > 0: label = 1
        elif pnl_difference < 0: label = -1

        if label != 0:
            event = {
                'unrealized_long_pnl': unrealized_long_pnl_at_relock,
                'unrealized_short_pnl': unrealized_short_pnl_at_relock,
                'total_unrealized_pnl': actual_outcome,
                'total_position_size': long_size_imbalanced + short_size_imbalanced,
                'volatility': data['volatility'].iloc[i],
                'adx': adx,
                'dmp': plus_di,
                'dmn': minus_di,
                'label': label
            }
            model2_events.append(event)

model2_df = pd.DataFrame(model2_events)
print(f"Found {len(model2_df)} 'RE_LOCK' events.")


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
        'dmn'
    ]
    X1 = model1_df[features].values
    y1 = model1_df['label_binary'].values

    if len(np.unique(y1)) < 2:
        print("모델 1을 학습하기에 레이블 종류가 충분하지 않습니다 (성공/실패 중 하나만 존재).")
    else:
        X1_train, X1_val, y1_train, y1_val = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)

        print("Model 1 Train label distribution:", dict(zip(*np.unique(y1_train, return_counts=True))))
        print("Model 1 Validation label distribution:", dict(zip(*np.unique(y1_val, return_counts=True))))

        model1 = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            gamma=0.1,
            subsample=0.8,
            use_label_encoder=False
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
        'dmn'
    ]
    X2 = model2_df[features_m2].values
    y2 = model2_df['label_binary'].values

    if len(np.unique(y2)) < 2:
        print("모델 2를 학습하기에 레이블 종류가 충분하지 않습니다 (성공/실패 중 하나만 존재).")
    else:
        X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

        print("Model 2 Train label distribution:", dict(zip(*np.unique(y2_train, return_counts=True))))
        print("Model 2 Validation label distribution:", dict(zip(*np.unique(y2_val, return_counts=True))))

        model2 = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
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


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
# === 2. 시뮬레이션 환경 설정      ===
# ===================================
# --- 시뮬레이션 파라미터 ---
WINDOW = 14                 # 기술적 지표 계산을 위한 윈도우 크기
PARTIAL_CLOSE_RATIO = 0.5   # 부분 청산 비율 (50%)
EVALUATION_WINDOW = 12      # 청산 성과 평가 기간 (5분 * 12 = 1시간)
SUCCESS_THRESHOLD_PCT = 0.0 # 성공적인 Exit으로 판단하는 임계값 (0보다 크면 성공)
FAILURE_THRESHOLD_PCT = 0.0 # 실패한 Exit으로 판단하는 임계값 (0보다 작으면 실패)

# --- 초기 잠금 모드 설정 ---
# 초기 진입가 스프레드를 1.5%로 고정
initial_spread_pct = 0.015
data['LongEntry'] = data['Close'].iloc[0] * (1 + initial_spread_pct / 2)
data['ShortEntry'] = data['Close'].iloc[0] * (1 - initial_spread_pct / 2)
data['Spread'] = data['LongEntry'] - data['ShortEntry']

# --- 시뮬레이션 상태를 기록할 열 추가 ---
data['unrealized_long_pnl'] = (data['Close'] - data['LongEntry']) / data['LongEntry']
data['unrealized_short_pnl'] = (data['ShortEntry'] - data['Close']) / data['ShortEntry']
data['total_unrealized_pnl'] = data['unrealized_long_pnl'] + data['unrealized_short_pnl']

data['realized_pnl'] = 0.0
data['long_position_size'] = 1.0  # 초기 포지션 크기를 1로 가정
data['short_position_size'] = 1.0
data['total_position_size'] = data['long_position_size'] + data['short_position_size']

data['action'] = "HOLD"  # 각 시점의 행동 (HOLD, PARTIAL_CLOSE_LONG, PARTIAL_CLOSE_SHORT)
data['label'] = 0      # 최종 학습 레이블 (+1, 0, -1)

print("시뮬레이션 환경 설정 완료")

# ===================================
# === 3. 특성(Feature) 생성       ===
# ===================================

# Volatility
data['volatility'] = data['Close'].pct_change().rolling(WINDOW).std().fillna(0)

# DMI (Directional Movement Index)
# ta.dmi가 아닌 ta.adx 함수를 사용하여 DMI 관련 지표(ADX, DMP, DMN)를 계산합니다.
dmi_df = ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=WINDOW)

# DMI 결과를 기존 데이터프레임에 병합
data = data.join(dmi_df)

# 열 이름 변경 및 결측치 처리
data.rename(columns={
    f'ADX_{WINDOW}': 'adx',
    f'DMP_{WINDOW}': 'dmp',
    f'DMN_{WINDOW}': 'dmn'
}, inplace=True)
# ChainedAssignmentError 경고를 해결하기 위해 fillna 작업을 하나로 통합
data.fillna({
    'adx': 25,
    'dmp': data['dmp'].mean(),
    'dmn': data['dmn'].mean()
}, inplace=True)

print("특성 생성 완료")

# ===================================================
# === 4. 시뮬레이션 및 레이블링 (Simulation & Labeling) ===
# ===================================================

print("부분 청산 시뮬레이션을 시작합니다...")

# ADX 임계값. 이 값보다 낮으면 트렌드가 약하다고 판단.
ADX_TREND_THRESHOLD = 25

# 루프를 돌면서 상태를 업데이트하기 위한 임시 리스트 생성
actions = data['action'].tolist()
realized_pnls = data['realized_pnl'].tolist()
long_sizes = data['long_position_size'].tolist()
short_sizes = data['short_position_size'].tolist()

# WINDOW 기간 이후부터 시뮬레이션 시작
for i in range(WINDOW, len(data)):
    # 이전 스텝의 상태를 가져옴
    current_long_size = long_sizes[i-1]
    current_short_size = short_sizes[i-1]
    
    # 현재 스텝의 PnL과 ADX 값
    long_pnl = data['unrealized_long_pnl'].iloc[i]
    short_pnl = data['unrealized_short_pnl'].iloc[i]
    adx = data['adx'].iloc[i]
    adx_prev = data['adx'].iloc[i-1]
    plus_di = data['dmp'].iloc[i]
    minus_di = data['dmn'].iloc[i]

    # 다음 스텝으로 상태 이전
    realized_pnls[i] = realized_pnls[i-1]
    long_sizes[i] = current_long_size
    short_sizes[i] = current_short_size

    # --- 부분 청산 조건 확인 ---
    action_taken = "HOLD"

    # 1. 롱 포지션이 수익 중이고, 추세가 약해졌을 때
    if long_pnl > 0 and adx > ADX_TREND_THRESHOLD and plus_di < minus_di and current_long_size > 0:
        action_taken = "PARTIAL_CLOSE_LONG"
        
        # 청산할 규모
        close_amount = current_long_size * PARTIAL_CLOSE_RATIO
        
        # 실현 손익 업데이트
        realized_pnls[i] += long_pnl * close_amount
        
        # 포지션 사이즈 업데이트
        long_sizes[i] -= close_amount

    # 2. 숏 포지션이 수익 중이고, 추세가 약해졌을 때
    elif short_pnl > 0 and adx > ADX_TREND_THRESHOLD and plus_di > minus_di and current_short_size > 0:
        action_taken = "PARTIAL_CLOSE_SHORT"
        
        # 청산할 규모
        close_amount = current_short_size * PARTIAL_CLOSE_RATIO
        
        # 실현 손익 업데이트
        realized_pnls[i] += short_pnl * close_amount
        
        # 포지션 사이즈 업데이트
        short_sizes[i] -= close_amount
        
    actions[i] = action_taken

# 시뮬레이션 결과를 데이터프레임에 다시 할당
data['action'] = actions
data['realized_pnl'] = realized_pnls
data['long_position_size'] = long_sizes
data['short_position_size'] = short_sizes
data['total_position_size'] = data['long_position_size'] + data['short_position_size']

print("시뮬레이션 완료. 'action' 열에 결과 기록.")

print("부분 청산 결과에 대한 레이블링을 시작합니다...")

# 전체 PnL 계산 (미실현 + 실현)
data['total_pnl'] = data['total_unrealized_pnl'] + data['realized_pnl']
labels = data['label'].tolist()

# action이 발생한 지점의 정수 위치 인덱스를 찾음
action_indices = data[data['action'] != 'HOLD'].index
action_integer_indices = [data.index.get_loc(idx) for idx in action_indices]

for i_loc in action_integer_indices:
    # 평가 기간이 데이터 범위를 벗어나는지 확인
    if i_loc + EVALUATION_WINDOW >= len(data):
        continue

    # .iloc를 사용하여 정수 위치로 데이터에 접근
    pnl_at_action = data['total_pnl'].iloc[i_loc]
    pnl_at_evaluation = data['total_pnl'].iloc[i_loc + EVALUATION_WINDOW]
    
    pnl_change = pnl_at_evaluation - pnl_at_action
    
    # 성과 판단 및 레이블링
    if pnl_change > SUCCESS_THRESHOLD_PCT:
        labels[i_loc] = 1  # 성공
    elif pnl_change < FAILURE_THRESHOLD_PCT:
        labels[i_loc] = -1 # 실패
    else:
        labels[i_loc] = 0  # 중립

data['label'] = labels

print("레이블링 완료. 'label' 열에 결과 기록.")

# ===================================
# === 5. XGBoost 모델 학습         ===
# ===================================

# 학습 데이터 필터링: 'HOLD'가 아닌 action만 학습에 사용
train_df = data[data['action'] != 'HOLD'].copy()

# 레이블을 -1, 0, 1에서 0, 1, 2로 변경 (XGBoost는 0부터 시작하는 레이블을 선호)
train_df['label_shifted'] = train_df['label'] + 1

print(f"총 {len(data)}개의 데이터 중, {len(train_df)}개의 액션 데이터를 학습에 사용합니다.")

# CSV 저장
csv_file = f"{SYMBOL}_{TIMEFRAME}_strategic_exit.csv"
train_df.to_csv(csv_file)
print(f"CSV saved: {csv_file}")

# 특성(X)과 레이블(y) 정의
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
X = train_df[features].values
y = train_df['label_shifted'].values

# 데이터가 하나도 없는 경우를 대비
if len(X) == 0:
    print("학습할 데이터가 없습니다. 시뮬레이션 조건을 확인해주세요.")
else:
    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Train label distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Validation label distribution:", dict(zip(*np.unique(y_val, return_counts=True))))

    # 모델 학습
    model = xgb.XGBClassifier(
        objective='multi:softmax', 
        num_class=3, 
        eval_metric='mlogloss',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        gamma=0.1,
        subsample=0.8,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_val, y_val)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # 모델 저장
    model_file = "strategic_exit_model.json"
    model.save_model(model_file)
    print(f"XGBoost model saved: {model_file}")

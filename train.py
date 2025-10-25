import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ================================
# === 1. 데이터 로딩 (CCXT)  ===
# ================================
SYMBOL = 'SOLUSDT'
TIMEFRAME = '5m'
START_DATE = '2025-07-01T00:00:00Z'

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

# ===================================
# === 2. Feature / Label 생성       ===
# ===================================
WINDOW = 14

# Volatility
data['Volatility'] = data['Close'].pct_change().rolling(WINDOW).std().fillna(0)

# Spread 시뮬레이션

np.random.seed(42)

# 스프레드 범위: 1% ~ 2%
spread_pct = np.random.uniform(0.01, 0.02, len(data))

# 롱과 숏 진입가 계산
data['LongEntry'] = data['Close'] * (1 + spread_pct/2)
data['ShortEntry'] = data['Close'] * (1 - spread_pct/2)

# 실제 스프레드
data['Spread'] = data['LongEntry'] - data['ShortEntry']

# LongPnL / ShortPnL
data['LongPnL'] = (data['Close'] - data['LongEntry']) / data['LongEntry']
data['ShortPnL'] = (data['ShortEntry'] - data['Close']) / data['ShortEntry']

# TimeLocked
data['TimeLocked'] = np.random.randint(1,10,len(data))

# Label
def label_defensive_exit(row):
    if row['LongPnL'] < -0.005:
        return 0  # CloseLong
    elif row['ShortPnL'] < -0.005:
        return 1  # CloseShort
    else:
        return 2  # HoldBoth

data['Label'] = data.apply(label_defensive_exit, axis=1)

# CSV 저장
csv_file = f"{SYMBOL}_{TIMEFRAME}_defensive_exit.csv"
data[['LongPnL','ShortPnL','Volatility','Spread','TimeLocked','Label']].to_csv(csv_file)
print(f"CSV saved: {csv_file}")

# ===================================
# === 3. XGBoost 모델 학습         ===
# ===================================
X = data[['LongPnL','ShortPnL','Volatility','Spread','TimeLocked']].values
y = data['Label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train label distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Validation label distribution:", dict(zip(*np.unique(y_val, return_counts=True))))


model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

accuracy = model.score(X_val, y_val)
print(f"Validation Accuracy: {accuracy:.4f}")

# 모델 저장
model_file = "defensive_exit_model.json"
model.save_model(model_file)
print(f"XGBoost model saved: {model_file}")
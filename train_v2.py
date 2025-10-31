import pandas as pd
import numpy as np
import xgboost as xgb
import pandas_ta as ta
import config
from sklearn.model_selection import train_test_split

# ===================================
# === 1. 설정 (Configuration) ===
# ===================================
# --- 파일 및 모델 경로 ---
DATA_FILE = "BTCUSDT_5m_raw_data.csv"
MODEL_FILE = "trend_model.json"

# --- 새로운 학습 파라미터 ---
# 트리플 배리어 라벨링 설정
PROFIT_TAKE_PCT = 0.01
STOP_LOSS_PCT = 0.01
TIME_BARRIER = 60

# ===================================
# === 2. 데이터 및 특성 준비 ===
# ===================================
print("--- Loading Data and Calculating Features ---")
try:
    data = pd.read_csv(DATA_FILE, index_col="Timestamp", parse_dates=True)
    print(f"Data loaded successfully: {len(data)} rows")
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found.")
    exit()

# 특성 생성 함수 (모멘텀 특성 추가)
def calculate_features(df):
    # 기존 특성
    df["volatility"] = df["Close"].pct_change().rolling(config.WINDOW).std()
    ema_short = ta.ema(df["Close"], length=20)
    ema_long = ta.ema(df["Close"], length=100)
    df["price_vs_ema_short"] = df["Close"] / ema_short
    df["price_vs_ema_long"] = df["Close"] / ema_long
    df["ema_cross"] = ema_short / ema_long
    returns = df["Close"].pct_change()
    mean_returns = returns.rolling(config.WINDOW).mean()
    std_returns = returns.rolling(config.WINDOW).std()
    df["z_score"] = (returns - mean_returns) / std_returns
    dmi_df = ta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=config.WINDOW)
    df = df.join(dmi_df)
    df.rename(columns={f"ADX_{config.WINDOW}": "adx", f"DMP_{config.WINDOW}": "dmp", f"DMN_{config.WINDOW}": "dmn"}, inplace=True)

    # === 새로운 동적/모멘텀 특성 추가 ===
    MOMENTUM_PERIOD = 10
    df["price_momentum"] = df["Close"].pct_change(periods=MOMENTUM_PERIOD)
    df["volatility_momentum"] = df["volatility"].pct_change(periods=MOMENTUM_PERIOD)
    df["adx_momentum"] = df["adx"].pct_change(periods=MOMENTUM_PERIOD)
    
    return df

data = calculate_features(data)
print("Features calculated.")

# =====================================================
# === 3. 학습 데이터 생성 (Triple-Barrier Labeling) ===
# =====================================================
print("--- Generating Labels using Triple-Barrier Method ---")
labels = []
for i in range(len(data) - TIME_BARRIER):
    entry_price = data["Close"].iloc[i]
    upper_barrier = entry_price * (1 + PROFIT_TAKE_PCT)
    lower_barrier = entry_price * (1 - STOP_LOSS_PCT)
    label = 0
    for j in range(1, TIME_BARRIER + 1):
        future_high = data["High"].iloc[i + j]
        future_low = data["Low"].iloc[i + j]
        if future_high >= upper_barrier:
            label = 1
            break
        elif future_low <= lower_barrier:
            label = -1
            break
    labels.append(label)

features_df = data.iloc[:-TIME_BARRIER].copy()
features_df["label"] = labels

# 무한대(inf) 값을 NaN으로 변환 후, 모든 결측치(NaN) 제거
features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
features_df.dropna(inplace=True)

print("Label distribution:", features_df["label"].value_counts())

# ===================================
# === 4. 모델 학습 (Multi-class) ===
# ===================================
print("--- Training New Trend Model (Multi-class) ---")

# 특성과 라벨 분리
feature_columns = [
    "volatility", "adx", "dmp", "dmn", 
    "price_vs_ema_short", "price_vs_ema_long", "ema_cross", "z_score",
    "price_momentum", "volatility_momentum", "adx_momentum"
]
X = features_df[feature_columns]
y = features_df["label"].replace({-1: 2})

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# XGBoost 다중 클래스 분류 모델 학습
model_v2 = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
)

model_v2.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=True,
)

# ===================================
# === 5. 모델 저장 ===
# ===================================
print(f"--- Saving model to {MODEL_FILE} ---")
model_v2.save_model(MODEL_FILE)
print("New trend model saved successfully.")
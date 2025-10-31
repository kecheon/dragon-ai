import pandas as pd
import numpy as np
import xgboost as xgb
import pandas_ta as ta
import config
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint

# ===================================
# === 1. 설정 (Configuration) ===
# ===================================
DATA_FILE = "BTCUSDT_5m_raw_data.csv"
PROFIT_TAKE_PCT = 0.01
STOP_LOSS_PCT = 0.01
TIME_BARRIER = 60

# ===================================
# === 2. 데이터 및 특성 준비 ===
# ===================================
print("--- Loading Data and Calculating Features for Tuning ---")
try:
    data = pd.read_csv(DATA_FILE, index_col="Timestamp", parse_dates=True)
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found.")
    exit()

def calculate_features(df):
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
    MOMENTUM_PERIOD = 10
    df["price_momentum"] = df["Close"].pct_change(periods=MOMENTUM_PERIOD)
    df["volatility_momentum"] = df["volatility"].pct_change(periods=MOMENTUM_PERIOD)
    df["adx_momentum"] = df["adx"].pct_change(periods=MOMENTUM_PERIOD)
    return df

data = calculate_features(data)
print("Features calculated.")

# ===================================
# === 3. 학습 데이터 생성 ===
# ===================================
print("--- Generating Labels ---")
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
features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
features_df.dropna(inplace=True)

# ===================================
# === 4. 하이퍼파라미터 튜닝 ===
# ===================================
print("--- Starting Hyperparameter Tuning ---")
feature_columns = [
    "volatility", "adx", "dmp", "dmn", "price_vs_ema_short", "price_vs_ema_long",
    "ema_cross", "z_score", "price_momentum", "volatility_momentum", "adx_momentum"
]
X = features_df[feature_columns]
y = features_df["label"].replace({-1: 2})

# 데이터의 일부만 사용하여 튜닝 시간 단축 (예: 25%)
_, X_sample, _, y_sample = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 탐색할 하이퍼파라미터 공간 정의
param_dist = {
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(100, 400),
    'max_depth': randint(3, 8),
    'subsample': uniform(0.7, 0.3), # 0.7 ~ 1.0
    'colsample_bytree': uniform(0.7, 0.3), # 0.7 ~ 1.0
    'gamma': uniform(0, 0.5)
}

# XGBoost 모델 초기화
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

# RandomizedSearchCV 설정
# n_iter: 시도할 조합 수, cv: 교차 검증 폴드 수
random_search = RandomizedSearchCV(
    model, param_distributions=param_dist, n_iter=25, cv=3,
    scoring='f1_weighted', n_jobs=-1, random_state=42, verbose=3
)

# 튜닝 실행
random_search.fit(X_sample, y_sample)

print("\n--- Tuning Complete ---")
print("Best parameters found: ", random_search.best_params_)
print("Best weighted F1-score: ", random_search.best_score_)

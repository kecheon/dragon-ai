import pandas as pd
import numpy as np
import xgboost as xgb
import pandas_ta as ta
import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================
# === 1. 설정 (Configuration) ===
# ===================================
DATA_FILE = "BTCUSDT_5m_raw_data.csv"
MODEL_FILE = "trend_model.json"
PROFIT_TAKE_PCT = 0.01
STOP_LOSS_PCT = 0.01
TIME_BARRIER = 60

# ===================================
# === 2. 데이터 및 특성 준비 ===
# ===================================
print("--- Loading Data and Calculating Features ---")
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

# =====================================================
# === 3. 학습 데이터 생성 ===
# =====================================================
print("--- Generating Labels for Evaluation ---")
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

# ===================================
# === 4. 모델 로딩 및 평가 준비 ===
# ===================================
print("--- Loading Model and Preparing Test Data ---")
feature_columns = [
    "volatility", "adx", "dmp", "dmn", 
    "price_vs_ema_short", "price_vs_ema_long", "ema_cross", "z_score",
    "price_momentum", "volatility_momentum", "adx_momentum"
]
X = features_df[feature_columns]
y = features_df["label"].replace({-1: 2})

_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

y_pred = model.predict(X_val)

# ===================================
# === 5. 분류 성능 평가 ===
# ===================================
print("\n--- 1. Classification Performance Metrics ---")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["Neutral (0)", "Buy (1)", "Sell (2)"]))

print("Generating Confusion Matrix plot...")
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neutral", "Buy", "Sell"], yticklabels=["Neutral", "Buy", "Sell"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Confusion Matrix saved to confusion_matrix.png")

# ===================================
# === 6. 간단한 백테스트 ===
# ===================================
print("\n--- 2. Simple Backtest Performance ---")
total_trades = 0
wins = 0
pnl = 0

for pred, actual in zip(y_pred, y_val):
    if pred != 0:
        total_trades += 1
        if pred == actual:
            wins += 1
            pnl += PROFIT_TAKE_PCT
        else:
            pnl -= STOP_LOSS_PCT

if total_trades > 0:
    win_rate = (wins / total_trades) * 100
    print(f"Total Trades (Buy/Sell signals): {total_trades}")
    print(f"Winning Trades: {wins}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total PnL (as % of capital): {pnl * 100:.2f}%")
else:
    print("No Buy/Sell signals were generated in the test period.")

print("\n--- Evaluation Complete ---")
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import xgboost as xgb
from xgboost import callback

# ----------------------------
# 1️⃣ 데이터 준비
# ----------------------------

# train.csv 파일이 있으면 로드, 없으면 샘플 생성
if os.path.exists("train.csv"):
    df = pd.read_csv("train.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
else:
    print("⚠️ train.csv not found. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )

# 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 2️⃣ DMatrix 변환
# ----------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# ----------------------------
# 3️⃣ 하이퍼파라미터 설정
# ----------------------------
params = {
    "objective": "binary:logistic",  # 이진 분류
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# ----------------------------
# 4️⃣ Early Stopping 콜백
# ----------------------------
early_stop = callback.EarlyStopping(
    rounds=20,
    metric_name="logloss",
    data_name="validation_0",
    save_best=True
)

# ----------------------------
# 5️⃣ 모델 학습
# ----------------------------
evals = [(dtrain, "train"), (dval, "validation_0")]

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    callbacks=[early_stop],
    verbose_eval=True
)

# ----------------------------
# 6️⃣ 예측 및 평가
# ----------------------------
y_pred_proba = bst.predict(dval)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\n✅ Training finished.")
print("Sample predictions:", y_pred[:10])

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def train_model_v2():
    """
    volatility_data_v2.csv를 사용하여 변동성 예측 모델 v2를 훈련하고 저장합니다.
    """
    print("--- 모델 v2 훈련 시작 ---")
    
    # --- 1. 데이터 로드 ---
    print("학습용 데이터셋(volatility_data_v2.csv)을 로드합니다...")
    try:
        df = pd.read_csv("volatility_data_v2.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("오류: volatility_data_v2.csv 파일을 찾을 수 없습니다. 데이터 생성 단계를 먼저 실행해주세요.")
        return

    # --- 2. 피처와 레이블 정의 (v2) ---
    features = ['bb_width', 'atr_normalized', 'adx', 'volatility', 'volatility_roc']
    X = df[features]
    y = df['label']

    print(f"{len(df)}개의 데이터로 모델을 학습합니다.")
    print(f"사용된 피처: {features}")

    # --- 3. 훈련/검증 데이터 분할 (시계열) ---
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    print(f"훈련 데이터: {len(X_train)}개, 검증 데이터: {len(X_val)}개")
    train_label_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    val_label_dist = dict(zip(*np.unique(y_val, return_counts=True)))
    print("Train label distribution:", train_label_dist)
    print("Validation label distribution:", val_label_dist)

    if len(train_label_dist) < 2:
        print("모델을 학습하기에 레이블 종류가 충분하지 않습니다.")
        return

    # --- 4. 모델 설정 및 훈련 ---
    # 클래스 불균형 처리를 위해 scale_pos_weight 계산
    scale_pos_weight = train_label_dist[0] / train_label_dist[1]
    print(f"계산된 scale_pos_weight: {scale_pos_weight:.4f}")

    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        n_estimators=100, learning_rate=0.1, max_depth=5, # max_depth를 약간 줄여 과적합 방지 시도
        gamma=0.3, subsample=0.8, use_label_encoder=False,
        scale_pos_weight=scale_pos_weight
    )

    print("\nXGBoost 모델 훈련을 시작합니다...")
    model.fit(X_train, y_train)
    print("모델 훈련 완료.")

    # --- 5. 모델 평가 ---
    print("\n--- 모델 성능 평가 (검증 데이터) ---")
    y_pred = model.predict(X_val)
    
    # 예측 확률값 확인 (디버깅 및 분석용)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    print(f"예측 확률(class 1)의 분포: Min={y_pred_proba.min():.4f}, Mean={y_pred_proba.mean():.4f}, Max={y_pred_proba.max():.4f}")

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)

    print(f"\n정확도 (Accuracy): {accuracy:.4f}")
    print(f"정밀도 (Precision): {precision:.4f}  <-- (모델이 '변동성 시작'이라고 예측한 것 중 실제 비율)")
    print(f"재현율 (Recall): {recall:.4f}    <-- (실제 '변동성 시작' 중 모델이 잡아낸 비율)")
    
    print("\n혼동 행렬 (Confusion Matrix):")
    print("         [ 예측: 0 | 예측: 1 ]")
    print(f"실제: 0  [[{conf_matrix[0][0]:>6} | {conf_matrix[0][1]:>6} ]]")
    print(f"실제: 1  [[{conf_matrix[1][0]:>6} | {conf_matrix[1][1]:>6} ]]")


    # --- 6. 모델 저장 ---
    model_filename = "volatility_predictor_v2.json"
    model.save_model(model_filename)
    print(f"\n훈련된 모델 v2를 '{model_filename}' 파일로 저장했습니다.")

if __name__ == "__main__":
    train_model_v2()

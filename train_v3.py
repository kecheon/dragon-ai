import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def train_model_v3():
    """
    volatility_data_v3.csv를 사용하여 변동성 예측 모델 v3를 훈련하고 저장합니다.
    """
    print("--- 모델 v3 훈련 시작 (고급 피처) ---")
    
    # --- 1. 데이터 로드 ---
    print("학습용 데이터셋(volatility_data_v3.csv)을 로드합니다...")
    try:
        df = pd.read_csv("volatility_data_v3.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("오류: volatility_data_v3.csv 파일을 찾을 수 없습니다.")
        return

    # --- 2. 피처와 레이블 정의 (v3) ---
    features = ['squeeze_on', 'consolidation_count', 'adx', 'bb_width']
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
    
    if len(train_label_dist) < 2:
        print("모델을 학습하기에 레이블 종류가 충분하지 않습니다.")
        return

    # --- 4. 모델 설정 및 훈련 ---
    scale_pos_weight = train_label_dist[0] / train_label_dist[1]
    print(f"계산된 scale_pos_weight: {scale_pos_weight:.4f}")

    # 이전 튜닝에서 찾은 최적 파라미터를 사용
    best_params = {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}
    print(f"적용된 하이퍼파라미터: {best_params}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        gamma=0.3,
        subsample=0.8,
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        **best_params
    )

    print("\nXGBoost 모델 훈련을 시작합니다...")
    model.fit(X_train, y_train)
    print("모델 훈련 완료.")

    # --- 5. 모델 평가 ---
    print("\n--- 모델 성능 평가 (검증 데이터) ---")
    y_pred = model.predict(X_val)
    
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_val, y_pred)

    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall): {recall:.4f}")
    print("\n혼동 행렬 (Confusion Matrix):")
    print(conf_matrix)

    # --- 6. 모델 저장 ---
    model_filename = "volatility_predictor_v3.json"
    model.save_model(model_filename)
    print(f"\n훈련된 모델 v3를 '{model_filename}' 파일로 저장했습니다.")

if __name__ == "__main__":
    train_model_v3()
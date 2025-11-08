import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def tune_model_v2():
    """
    GridSearchCV를 사용하여 XGBoost 모델의 최적 하이퍼파라미터를 찾습니다.
    (목표: 정밀도(Precision) 극대화)
    """
    print("--- 하이퍼파라미터 튜닝 시작 (시간이 소요될 수 있습니다) ---")

    # --- 1. 데이터 로드 ---
    try:
        df = pd.read_csv("volatility_data_v2.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("오류: volatility_data_v2.csv 파일을 찾을 수 없습니다.")
        return

    # --- 2. 훈련/검증 데이터 준비 ---
    features = ['bb_width', 'atr_normalized', 'adx', 'volatility', 'volatility_roc']
    split_index = int(len(df) * 0.8)
    X_train = df[features][:split_index]
    y_train = df['label'][:split_index]
    X_val = df[features][split_index:]
    y_val = df['label'][split_index:]

    # --- 3. GridSearchCV 설정 ---
    # 클래스 불균형 처리를 위한 scale_pos_weight 계산
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    print(f"계산된 scale_pos_weight: {scale_pos_weight:.4f}")

    # 테스트할 하이퍼파라미터 그리드 정의
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1]
    }

    # 기본 모델 초기화
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        gamma=0.3,
        subsample=0.8,
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight
    )

    # GridSearchCV 객체 생성 (평가 지표를 'precision'으로 설정)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='precision',
        cv=3, # 3-fold cross-validation
        verbose=1, # 진행 과정 출력
        n_jobs=-1 # 모든 CPU 코어 사용
    )

    # --- 4. 튜닝 실행 ---
    print("\nGridSearchCV를 사용하여 최적 파라미터를 탐색합니다...")
    grid_search.fit(X_train, y_train)

    print("\n튜닝 완료!")
    print("최적 하이퍼파라미터:", grid_search.best_params_)
    print("최고 정밀도 점수 (교차 검증):", grid_search.best_score_)

    # --- 5. 최적 모델로 재평가 ---
    print("\n--- 최적 모델 성능 평가 (검증 데이터) ---")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)

    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall): {recall:.4f}")
    print("\n혼동 행렬 (Confusion Matrix):")
    print(conf_matrix)
    
    # --- 6. 최적 모델 저장 ---
    model_filename = "volatility_predictor_v2_tuned.json"
    best_model.save_model(model_filename)
    print(f"\n튜닝된 최적 모델을 '{model_filename}' 파일로 저장했습니다.")


if __name__ == "__main__":
    tune_model_v2()
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

def tune_final_model_hyperparameters():
    """
    GridSearchCV와 TimeSeriesSplit을 사용하여
    strategy_dataset_v3.csv에 대한 최종 다중 클래스 모델의 최적 하이퍼파라미터를 찾습니다.
    """
    # ===================================
    # 1. 데이터 로드 및 전처리 (train_v3.py와 동일)
    # ===================================
    file_path = "strategy_dataset_v3.csv"
    print(f"'{file_path}' 파일에서 튜닝용 데이터를 로드합니다...")
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        print("먼저 generate_strategy_data.py를 실행하여 데이터를 생성해야 합니다.")
        return

    print("데이터 전처리를 시작합니다...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    feature_columns = [col for col in df.columns if col not in ['label', 'pnl_at_action']]
    X = df[feature_columns]
    y_str = df['label']

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    
    num_classes = len(le.classes_)
    print(f"레이블 인코딩 완료. 총 {num_classes}개의 클래스: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")


    # ===================================
    # 2. 하이퍼파라미터 튜닝 설정 및 실행
    # ===================================
    print("하이퍼파라미터 튜닝을 시작합니다... (시간이 다소 소요될 수 있습니다)")

    # 기본 모델 정의 (다중 클래스)
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric='mlogloss' # 다중 클래스용 평가 지표
    )

    # 탐색할 파라미터 그리드 정의
    # train_v3.py의 기본값을 중심으로 탐색 범위를 설정합니다.
    param_grid = {
        'max_depth': [7, 9, 11],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'gamma': [0.1, 0.3],
        'subsample': [0.8, 1.0]
    }

    # 시계열 교차 검증 설정
    time_series_split = TimeSeriesSplit(n_splits=5)

    # GridSearchCV 설정
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy', # 다중 클래스에서는 accuracy가 직관적입니다.
        cv=time_series_split,
        verbose=2,
        n_jobs=-1
    )

    # 튜닝 실행
    grid_search.fit(X, y)

    # ===================================
    # 3. 튜닝 결과 출력
    # ===================================
    print("\n--- 최종 모델 튜닝 결과 ---")
    print(f"최적의 파라미터: {grid_search.best_params_}")
    print(f"최고 교차 검증 점수 (Accuracy): {grid_search.best_score_:.4f}")
    print("\n이제 이 파라미터들을 train_v3.py에 적용하여 최종 모델을 학습할 수 있습니다.")


if __name__ == "__main__":
    tune_final_model_hyperparameters()

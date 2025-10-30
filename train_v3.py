import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def train_final_model():
    """
    생성된 strategy_dataset_v3.csv를 사용하여 최종 다중 클래스 모델을 학습하고 평가합니다.
    """
    # ===================================
    # 1. 데이터 로드 및 전처리
    # ===================================
    file_path = "strategy_dataset_v3.csv"
    print(f"'{file_path}' 파일에서 최종 학습 데이터를 로드합니다...")
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        print("먼저 generate_strategy_data.py를 실행하여 데이터를 생성해야 합니다.")
        return

    print("데이터 전처리를 시작합니다...")
    
    # 무한대 값이나 매우 큰 값들을 NaN으로 변환 후 제거
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 특성과 레이블 정의
    # pnl_at_action은 미래 정보이므로 제외, label은 문자열이므로 제외
    feature_columns = [col for col in df.columns if col not in ['label', 'pnl_at_action']]
    X = df[feature_columns]
    y_str = df['label']

    # 문자열 레이블을 숫자로 변환 ("CUT_LOSS":0, "HOLD":1, "RE_LOCK":2)
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    print(f"레이블 인코딩: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ===================================
    # 2. 데이터 분할 및 모델 학습
    # ===================================
    print("\n모델 학습을 시작합니다...")

    # 시계열 데이터 분할 (셔플 없이 마지막 20%를 검증 세트로 사용)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"훈련 데이터: {len(X_train)}개, 검증 데이터: {len(X_val)}개")
    
    # 모델 정의 (튜닝된 파라미터 사용)
    model = xgb.XGBClassifier(
        objective='multi:softmax',    # 다중 클래스 분류
        num_class=len(le.classes_), # 클래스의 수 (4)
        n_estimators=100,           # 튜닝된 값
        learning_rate=0.1,          # 튜닝된 값
        max_depth=7,                # 튜닝된 값
        gamma=0.1,                  # 튜닝된 값
        subsample=0.8,              # 튜닝된 값
        use_label_encoder=False
    )

    # 모델 학습
    model.fit(X_train, y_train)

    # ===================================
    # 3. 모델 평가
    # ===================================
    print("\n--- 최종 모델 평가 결과 ---")

    # 예측
    y_pred = model.predict(X_val)

    # 정확도
    accuracy = np.mean(y_val == y_pred)
    print(f"전체 검증 정확도 (Accuracy): {accuracy:.4f}")

    # 상세 분류 리포트
    print("\n상세 분류 리포트:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    train_final_model()

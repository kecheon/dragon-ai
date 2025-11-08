import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, recall_score

def analyze_threshold_v3():
    """
    훈련된 v3 모델의 예측 확률 임계값을 조정하며 정밀도와 재현율의 변화를 분석합니다.
    """
    print("--- v3 모델 예측 임계값 분석 시작 ---")
    
    # --- 1. 데이터 및 모델 로드 ---
    try:
        df = pd.read_csv("volatility_data_v3.csv", index_col='Timestamp', parse_dates=True)
        model = xgb.XGBClassifier()
        model.load_model("volatility_predictor_v3.json")
        print("v3 모델(volatility_predictor_v3.json)을 성공적으로 로드했습니다.")
    except FileNotFoundError as e:
        print(f"오류: 필요한 파일({e.filename})을 찾을 수 없습니다. 이전 단계를 완료했는지 확인하세요.")
        return

    # --- 2. 검증 데이터 준비 ---
    features = ['squeeze_on', 'consolidation_count', 'adx', 'bb_width']
    split_index = int(len(df) * 0.8)
    X_val = df[features][split_index:]
    y_val = df['label'][split_index:]

    # --- 3. 예측 확률 계산 ---
    print("검증 데이터에 대한 예측 확률을 계산합니다...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # --- 4. 임계값별 성능 분석 ---
    print("\n임계값(Threshold)에 따른 정밀도(Precision) 및 재현율(Recall) 변화")
    print("="*60)
    print(f"{ '임계값':<10} | { '정밀도':<10} | { '재현율':<10} | { '예측된 신호 수':<15}")
    print("-"*60)

    thresholds = np.arange(0.5, 1.0, 0.05)
    
    for thr in thresholds:
        y_pred_custom = (y_pred_proba >= thr).astype(int)
        
        precision = precision_score(y_val, y_pred_custom, zero_division=0)
        recall = recall_score(y_val, y_pred_custom, zero_division=0)
        num_signals = np.sum(y_pred_custom)
        
        print(f"{thr:<10.2f} | {precision:<10.4f} | {recall:<10.4f} | {num_signals:<15}")
        
    print("="*60)
    print("\n분석: 이번에는 임계값을 높임에 따라 정밀도가 의미 있게 향상되는지 주목해야 합니다.")

if __name__ == "__main__":
    analyze_threshold_v3()

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, recall_score

def analyze_threshold():
    """
    훈련된 모델의 예측 확률 임계값을 조정하며 정밀도와 재현율의 변화를 분석합니다.
    """
    print("--- 예측 임계값 분석 시작 ---")
    
    # --- 1. 데이터 및 모델 로드 ---
    try:
        df = pd.read_csv("volatility_data_v2.csv", index_col='Timestamp', parse_dates=True)
        model = xgb.XGBClassifier()
        model.load_model("volatility_predictor_v2.json")
    except FileNotFoundError as e:
        print(f"오류: 필요한 파일({e.filename})을 찾을 수 없습니다. 이전 단계를 완료했는지 확인하세요.")
        return

    # --- 2. 검증 데이터 준비 ---
    features = ['bb_width', 'atr_normalized', 'adx', 'volatility', 'volatility_roc']
    split_index = int(len(df) * 0.8)
    X_val = df[features][split_index:]
    y_val = df['label'][split_index:]

    # --- 3. 예측 확률 계산 ---
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # --- 4. 임계값별 성능 분석 ---
    print("\n임계값(Threshold)에 따른 정밀도(Precision) 및 재현율(Recall) 변화")
    print("="*60)
    print(f"{ '임계값':<10} | { '정밀도':<10} | { '재현율':<10} | { '예측된 신호 수':<15}")
    print("-"*60)

    thresholds = np.arange(0.5, 0.91, 0.05)
    
    for thr in thresholds:
        # 임계값에 따라 예측 레이블 결정 (0 또는 1)
        y_pred_custom = (y_pred_proba >= thr).astype(int)
        
        precision = precision_score(y_val, y_pred_custom, zero_division=0)
        recall = recall_score(y_val, y_pred_custom, zero_division=0)
        num_signals = np.sum(y_pred_custom)
        
        print(f"{thr:<10.2f} | {precision:<10.4f} | {recall:<10.4f} | {num_signals:<15}")
        
    print("="*60)
    print("\n분석: 임계값을 높이면 예측된 신호 수는 줄어들지만, 정밀도가 향상되는 경향을 보입니다.")
    print("우리는 이 표를 보고 '정밀도'와 '신호 수' 사이의 합리적인 타협점을 찾아야 합니다.")

if __name__ == "__main__":
    analyze_threshold()

import pandas as pd
import numpy as np
import pandas_ta as ta
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def tune_model1_hyperparameters():
    """
    GridSearchCV와 TimeSeriesSplit을 사용하여 모델 1의 최적 하이퍼파라미터를 찾습니다.
    """
    # ===================================
    # 1. train.py에서 데이터 준비 로직 복사
    # ===================================
    print("train.py와 동일한 조건으로 데이터를 준비합니다...")
    try:
        data = pd.read_csv("BTCUSDT_5m_raw_data.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("오류: BTCUSDT_5m_raw_data.csv 파일을 찾을 수 없습니다.")
        return

    # 파라미터 (train.py와 동일)
    WINDOW = 14
    PARTIAL_CLOSE_RATIO = 0.5
    EVALUATION_WINDOW = 24
    FIXED_SPREAD = 0.02
    LOOKBACK_PERIOD = 48
    ADX_TREND_THRESHOLD = 20
    PROFIT_TAKE_PCT = 0.0046
    STOP_LOSS_PCT = 0.0021

    # 특성 생성 (train.py와 동일)
    print("특성 및 이벤트를 생성합니다...")
    data['volatility'] = data['Close'].pct_change().rolling(WINDOW).std()
    ema_short = ta.ema(data['Close'], length=20)
    ema_long = ta.ema(data['Close'], length=100)
    data['price_vs_ema_short'] = data['Close'] / ema_short
    data['price_vs_ema_long'] = data['Close'] / ema_long
    data['ema_cross'] = ema_short / ema_long
    dmi_df = ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=WINDOW)
    data = data.join(dmi_df)
    data.rename(columns={f'ADX_{WINDOW}': 'adx', f'DMP_{WINDOW}': 'dmp', f'DMN_{WINDOW}': 'dmn'}, inplace=True)
    data.dropna(inplace=True)

    # 모델 1 이벤트 생성 (train.py와 동일)
    model1_events = []
    for i in range(LOOKBACK_PERIOD + WINDOW, len(data) - EVALUATION_WINDOW):
        base_price = data['Close'].iloc[i - LOOKBACK_PERIOD]
        long_entry_price = base_price * (1 + FIXED_SPREAD / 2)
        short_entry_price = base_price * (1 - FIXED_SPREAD / 2)
        current_price = data['Close'].iloc[i]
        unrealized_long_pnl = (current_price - long_entry_price)
        unrealized_short_pnl = (short_entry_price - current_price)
        adx = data['adx'].iloc[i]
        plus_di = data['dmp'].iloc[i]
        minus_di = data['dmn'].iloc[i]
        action = None
        if unrealized_long_pnl > 0 and adx < ADX_TREND_THRESHOLD and plus_di < minus_di:
            action = "PARTIAL_CLOSE_LONG"
        elif unrealized_short_pnl > 0 and adx < ADX_TREND_THRESHOLD and plus_di > minus_di:
            action = "PARTIAL_CLOSE_SHORT"
        if action:
            pnl_at_action = unrealized_long_pnl + unrealized_short_pnl
            if action == "PARTIAL_CLOSE_LONG":
                remaining_long_size, remaining_short_size = 1.0 - PARTIAL_CLOSE_RATIO, 1.0
            else:
                remaining_long_size, remaining_short_size = 1.0, 1.0 - PARTIAL_CLOSE_RATIO
            label = 0
            profit_target = pnl_at_action + (base_price * PROFIT_TAKE_PCT)
            stop_loss = pnl_at_action - (base_price * STOP_LOSS_PCT)
            future_prices = data['Close'].iloc[i + 1 : i + 1 + EVALUATION_WINDOW]
            pnl_now = pnl_at_action
            for price in future_prices:
                pnl_now = ((price - long_entry_price) * remaining_long_size) + ((short_entry_price - price) * remaining_short_size)
                if pnl_now >= profit_target: label = 1; break
                if pnl_now <= stop_loss: label = -1; break
            if label == 0:
                if pnl_now > pnl_at_action: label = 1
                elif pnl_now < pnl_at_action: label = -1
            if label != 0:
                event = {
                    'unrealized_long_pnl': unrealized_long_pnl, 'unrealized_short_pnl': unrealized_short_pnl,
                    'total_unrealized_pnl': pnl_at_action, 'total_position_size': 2.0,
                    'volatility': data['volatility'].iloc[i], 'adx': adx, 'dmp': plus_di, 'dmn': minus_di,
                    'price_vs_ema_short': data['price_vs_ema_short'].iloc[i],
                    'price_vs_ema_long': data['price_vs_ema_long'].iloc[i],
                    'ema_cross': data['ema_cross'].iloc[i], 'label': label
                }
                model1_events.append(event)
    
    model1_df = pd.DataFrame(model1_events)
    model1_df['label_binary'] = model1_df['label'].replace({-1: 0, 1: 1})
    features = ['unrealized_long_pnl', 'unrealized_short_pnl', 'total_unrealized_pnl', 'total_position_size', 'volatility', 'adx', 'dmp', 'dmn', 'price_vs_ema_short', 'price_vs_ema_long', 'ema_cross']
    X1 = model1_df[features].values
    y1 = model1_df['label_binary'].values

    # ===================================
    # 2. 하이퍼파라미터 튜닝 설정 및 실행
    # ===================================
    print("\n하이퍼파라미터 튜닝을 시작합니다... (시간이 다소 소요될 수 있습니다)")

    # 클래스 불균형 처리를 위한 scale_pos_weight 계산
    scale_pos_weight = np.sum(y1 == 0) / np.sum(y1 == 1)

    # 기본 모델 정의
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight
    )

    # 탐색할 파라미터 그리드 정의
    param_grid = {
        'max_depth': [5, 7, 9],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'gamma': [0.1, 0.3]
    }

    # 시계열 교차 검증 설정
    time_series_split = TimeSeriesSplit(n_splits=5)

    # GridSearchCV 설정
    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=time_series_split, 
        verbose=2, 
        n_jobs=-1
    )

    # 튜닝 실행
    grid_search.fit(X1, y1)

    # ===================================
    # 3. 튜닝 결과 출력
    # ===================================
    print("\n--- 튜닝 결과 ---")
    print(f"최적의 파라미터: {grid_search.best_params_}")
    print(f"최고 교차 검증 점수 (Accuracy): {grid_search.best_score_:.4f}")

if __name__ == "__main__":
    tune_model1_hyperparameters()

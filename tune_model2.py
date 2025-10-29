import pandas as pd
import numpy as np
import pandas_ta as ta
import xgboost as xgb
import config # 설정 파일 임포트
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def tune_model2_hyperparameters():
    """
    GridSearchCV와 TimeSeriesSplit을 사용하여 모델 2의 최적 하이퍼파라미터를 찾습니다.
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

    # 특성 생성
    print("특성 및 이벤트를 생성합니다...")
    data['volatility'] = data['Close'].pct_change().rolling(config.WINDOW).std()
    ema_short = ta.ema(data['Close'], length=20)
    ema_long = ta.ema(data['Close'], length=100)
    data['price_vs_ema_short'] = data['Close'] / ema_short
    data['price_vs_ema_long'] = data['Close'] / ema_long
    data['ema_cross'] = ema_short / ema_long
    dmi_df = ta.adx(high=data['High'], low=data['Low'], close=data['Close'], length=config.WINDOW)
    data = data.join(dmi_df)
    data.rename(columns={f'ADX_{config.WINDOW}': 'adx', f'DMP_{config.WINDOW}': 'dmp', f'DMN_{config.WINDOW}': 'dmn'}, inplace=True)
    data.dropna(inplace=True)

    # 모델 2 이벤트 생성 (train.py와 동일)
    IMBALANCE_DURATIONS = [12, 24, 48, 96]
    model2_events = []
    for time_in_imbalance in IMBALANCE_DURATIONS:
        start_index = config.LOOKBACK_PERIOD + time_in_imbalance + config.WINDOW
        for i in range(start_index, len(data) - config.EVALUATION_WINDOW):
            initial_entry_price = data['Close'].iloc[i - config.LOOKBACK_PERIOD - time_in_imbalance]
            long_entry_price = initial_entry_price * (1 + config.FIXED_SPREAD / 2)
            short_entry_price = initial_entry_price * (1 - config.FIXED_SPREAD / 2)
            adx = data['adx'].iloc[i]
            plus_di = data['dmp'].iloc[i]
            minus_di = data['dmn'].iloc[i]
            params = {
                'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di,
                'ADX_TREND_THRESHOLD': config.ADX_TREND_THRESHOLD
            }
            action = config.get_model2_action_decision(params)
            if action is not None:
                if action == "RE_LOCK_FOR_SHORT":
                    long_size_imbalanced, short_size_imbalanced = 1.0 - config.PARTIAL_CLOSE_RATIO, 1.0
                elif action == "RE_LOCK_FOR_LONG":
                    long_size_imbalanced, short_size_imbalanced = 1.0, 1.0 - config.PARTIAL_CLOSE_RATIO
                current_price = data['Close'].iloc[i]
                pnl_long = (current_price - long_entry_price) * long_size_imbalanced
                pnl_short = (short_entry_price - current_price) * short_size_imbalanced
                actual_outcome = pnl_long + pnl_short
                price_at_eval = data['Close'].iloc[i + config.EVALUATION_WINDOW]
                hypo_pnl_long = (price_at_eval - long_entry_price) * long_size_imbalanced
                hypo_pnl_short = (short_entry_price - price_at_eval) * short_size_imbalanced
                hypothetical_outcome = hypo_pnl_long + hypo_pnl_short
                pnl_difference = actual_outcome - hypothetical_outcome
                label = 0
                if pnl_difference > 0: label = 1
                elif pnl_difference < 0: label = -1
                if label != 0:
                    event = {
                        'unrealized_long_pnl': pnl_long, 'unrealized_short_pnl': pnl_short,
                        'total_unrealized_pnl': actual_outcome,
                        'total_position_size': long_size_imbalanced + short_size_imbalanced,
                        'volatility': data['volatility'].iloc[i], 'adx': adx, 'dmp': plus_di, 'dmn': minus_di,
                        'price_vs_ema_short': data['price_vs_ema_short'].iloc[i],
                        'price_vs_ema_long': data['price_vs_ema_long'].iloc[i],
                        'ema_cross': data['ema_cross'].iloc[i],
                        'time_in_imbalance': time_in_imbalance,
                        'label': label
                    }
                    model2_events.append(event)
    
    model2_df = pd.DataFrame(model2_events)
    model2_df['label_binary'] = model2_df['label'].replace({-1: 0, 1: 1})
    features_m2 = ['unrealized_long_pnl', 'unrealized_short_pnl', 'total_unrealized_pnl', 'total_position_size', 'volatility', 'adx', 'dmp', 'dmn', 'price_vs_ema_short', 'price_vs_ema_long', 'ema_cross', 'time_in_imbalance']
    X2 = model2_df[features_m2].values
    y2 = model2_df['label_binary'].values

    # ===================================
    # 2. 하이퍼파라미터 튜닝 설정 및 실행
    # ===================================
    print("\n하이퍼파라미터 튜닝을 시작합니다... (시간이 매우 오래 소요될 수 있습니다)")

    # 클래스 불균형 처리를 위한 scale_pos_weight 계산
    scale_pos_weight = np.sum(y2 == 0) / np.sum(y2 == 1) if np.sum(y2 == 1) > 0 else 1

    # 기본 모델 정의
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight)

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
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=time_series_split, verbose=2, n_jobs=-1)

    # 튜닝 실행
    grid_search.fit(X2, y2)

    # ===================================
    # 3. 튜닝 결과 출력
    # ===================================
    print("\n--- 모델 2 튜닝 결과 ---")
    print(f"최적의 파라미터: {grid_search.best_params_}")
    print(f"최고 교차 검증 점수 (Accuracy): {grid_search.best_score_:.4f}")

if __name__ == "__main__":
    tune_model2_hyperparameters()

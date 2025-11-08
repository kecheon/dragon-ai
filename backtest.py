import pandas as pd
import xgboost as xgb
import numpy as np

def run_backtest():
    """
    v3 모델과 정의된 거래 전략을 사용하여 백테스팅을 수행합니다.
    """
    print("--- 최종 전략 백테스팅 시작 ---")

    # --- 1. 데이터 및 모델 로드 ---
    try:
        model_df = pd.read_csv("volatility_data_v3.csv", index_col='Timestamp', parse_dates=True)
        # 백테스팅에는 실제 가격 데이터가 필요하므로 원본 데이터를 로드합니다.
        # 사용자가 3년치 데이터를 합쳤다고 가정하고, 해당 파일 이름을 'BTCUSDT_5m_raw_data.csv'로 예상합니다.
        price_df = pd.read_csv("BTCUSDT_5m_raw_data.csv.2022", index_col='Timestamp', parse_dates=True)
        
        model = xgb.XGBClassifier()
        model.load_model("volatility_predictor_v3.json")
        print("데이터 및 v3 모델을 성공적으로 로드했습니다.")
    except FileNotFoundError as e:
        print(f"오류: 필요한 파일({e.filename})을 찾을 수 없습니다. 이전 단계를 완료했는지 확인하세요.")
        print("3년치 데이터 파일의 이름이 'BTCUSDT_5m_raw_data.csv'가 맞는지 확인해주세요.")
        return

    # --- 2. 신호 생성 ---
    features = ['squeeze_on', 'consolidation_count', 'adx', 'bb_width']
    X = model_df[features]
    
    print("예측 확률을 계산합니다...")
    pred_proba = model.predict_proba(X)[:, 1]
    
    # 최종 신호: 임계값 0.90 적용
    THRESHOLD = 0.90
    signals = (pred_proba >= THRESHOLD).astype(int)
    model_df['signal'] = signals
    
    # 백테스팅을 위해 신호 데이터를 원본 가격 데이터와 합침
    df = price_df.join(model_df['signal'], how='inner')

    # --- 3. 백테스팅 시뮬레이션 ---
    print("백테스팅 시뮬레이션을 시작합니다...")
    
    # 전략 파라미터
    TAKE_PROFIT_PCT = 0.02
    STOP_LOSS_PCT = 0.01
    TIME_LIMIT_CANDLES = 120 # 4시간
    FEE_PCT = 0.0005

    in_position = False
    position = {}
    trade_history = []

    for i in range(1, len(df)):
        # 포지션 종료 조건 확인
        if in_position:
            position['duration'] += 1
            pnl = 0
            exit_reason = None

            # Stop Loss 체크
            if position['direction'] == 'long' and df['Low'].iloc[i] <= position['stop_loss_price']:
                pnl = (position['stop_loss_price'] - position['entry_price']) / position['entry_price']
                exit_reason = 'Stop Loss'
            elif position['direction'] == 'short' and df['High'].iloc[i] >= position['stop_loss_price']:
                pnl = (position['entry_price'] - position['stop_loss_price']) / position['entry_price']
                exit_reason = 'Stop Loss'

            # Take Profit 체크
            if exit_reason is None:
                if position['direction'] == 'long' and df['High'].iloc[i] >= position['take_profit_price']:
                    pnl = TAKE_PROFIT_PCT
                    exit_reason = 'Take Profit'
                elif position['direction'] == 'short' and df['Low'].iloc[i] <= position['take_profit_price']:
                    pnl = TAKE_PROFIT_PCT
                    exit_reason = 'Take Profit'

            # Time Limit 체크
            if exit_reason is None and position['duration'] >= TIME_LIMIT_CANDLES:
                exit_price = df['Close'].iloc[i]
                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry_price']) / position['entry_price']
                else:
                    pnl = (position['entry_price'] - exit_price) / position['entry_price']
                exit_reason = 'Time Limit'

            if exit_reason:
                net_pnl = pnl - (2 * FEE_PCT) # 진입/청산 수수료
                position['pnl'] = net_pnl
                position['exit_reason'] = exit_reason
                trade_history.append(position)
                in_position = False
                position = {}

        # 포지션 진입 조건 확인
        if not in_position and df['signal'].iloc[i-1] == 1:
            entry_price = df['Open'].iloc[i]
            direction = 'long' if df['Close'].iloc[i-1] > df['Open'].iloc[i-1] else 'short'
            
            if direction == 'long':
                take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            else: # short
                take_profit_price = entry_price * (1 - TAKE_PROFIT_PCT)
                stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)

            position = {
                'entry_time': df.index[i],
                'entry_price': entry_price,
                'direction': direction,
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'duration': 0
            }
            in_position = True

    # --- 4. 결과 리포트 ---
    if not trade_history:
        print("\n백테스팅 기간 동안 거래가 발생하지 않았습니다.")
        return
        
    report_df = pd.DataFrame(trade_history)
    
    total_trades = len(report_df)
    wins = report_df[report_df['pnl'] > 0]
    losses = report_df[report_df['pnl'] <= 0]
    
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    
    total_pnl = report_df['pnl'].sum()
    
    avg_profit = wins['pnl'].mean()
    avg_loss = losses['pnl'].mean()
    
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if abs(losses['pnl'].sum()) > 0 else 0
    
    print("\n--- 백테스팅 결과 리포트 ---")
    print("="*40)
    print(f"총 거래 횟수: {total_trades}")
    print(f"총 순익 (PNL): {total_pnl:.4f} (초기 자본의 {total_pnl*100:.2f}%)")
    print(f"승률 (Win Rate): {win_rate:.2%}")
    print(f"수익 거래 수: {len(wins)}")
    print(f"손실 거래 수: {len(losses)}")
    print(f"평균 익절률: {avg_profit:.4f}")
    print(f"평균 손절률: {avg_loss:.4f}")
    print(f"손익비 (Profit Factor): {profit_factor:.2f}")
    print("="*40)
    
    print("\n거래 종료 사유 분포:")
    print(report_df['exit_reason'].value_counts(normalize=True).apply('{:.2%}'.format))


if __name__ == "__main__":
    run_backtest()

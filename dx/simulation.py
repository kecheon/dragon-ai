import argparse
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

# 우리가 만든 모듈들
from data_loader import load_price_data
from defensive_strategy import StrategyConfig, Position, AccountState, DynamicHedgeStrategy

# 시뮬레이션을 위한 글로벌 설정 (트리거 조건 추가)
INITIAL_POSITION_SIZE = 1.0
FIXED_SPREAD = 0.02
LOOKBACK_PERIOD = 96
SIMULATION_WINDOW = 288
# 새로운 시장 상황 트리거 임계값 (DMI 기반)
ADX_TRIGGER_THRESHOLD = 25.0 # ADX가 이 값 이상일 때 추세 강하다고 판단
DMI_CROSSOVER_THRESHOLD = 0 # +DI와 -DI의 교차 또는 우위 판단 기준 (0이면 단순 교차)

def run_simulation(symbol: str):
    """
    시장 상황 기반 트리거를 사용하는 최종 방어 전략 시뮬레이션을 실행합니다.
    """
    # 1. 데이터 로드 및 특성 계산 (ADX 추가)
    print(f"'{symbol}'에 대한 데이터를 로드하고 특성(ATR, ADX)을 계산합니다...")
    data = load_price_data(symbol)
    if data.empty:
        print("데이터가 없어 시뮬레이션을 중단합니다.")
        return

    # ADX/DMI 및 변동성(ATR) 지표 계산
    dmi_df = data.ta.adx(length=14)
    data = data.join(dmi_df)
    data.rename(columns={'ADX_14': 'adx', 'DMP_14': 'plus_di', 'DMN_14': 'minus_di'}, inplace=True)

    data.ta.atr(length=14, append=True)
    data.rename(columns={'ATRr_14': 'atr'}, inplace=True)
    data['atr_sma'] = data['atr'].rolling(window=50).mean() # ATR의 50주기 이동평균


    data.dropna(inplace=True)
    
    config = StrategyConfig()
    strategy = DynamicHedgeStrategy(config, logger=print)
    
    results = []

    print("시뮬레이션을 시작합니다...")
    for i in tqdm(range(LOOKBACK_PERIOD, len(data) - SIMULATION_WINDOW)):
        base_price = data['Close'].iloc[i - LOOKBACK_PERIOD]
        
        long_pos = Position(
            side="LONG", entry_price=base_price * (1 + FIXED_SPREAD / 2),
            size=INITIAL_POSITION_SIZE, initial_size=INITIAL_POSITION_SIZE
        )
        short_pos = Position(
            side="SHORT", entry_price=base_price * (1 - FIXED_SPREAD / 2),
            size=INITIAL_POSITION_SIZE, initial_size=INITIAL_POSITION_SIZE
        )
        
        initial_long_entry = long_pos.entry_price
        initial_short_entry = short_pos.entry_price
        
        final_pnl_strategy = None
        status = "HOLD"

        for j in range(SIMULATION_WINDOW):
            current_step = i + j
            market_price = data['Close'].iloc[current_step]
            
            # 1. 방어 로직 트리거 조건 (DMI + 변동성 필터)
            adx_now = data['adx'].iloc[current_step]
            plus_di_now = data['plus_di'].iloc[current_step]
            minus_di_now = data['minus_di'].iloc[current_step]
            atr_now = data['atr'].iloc[current_step]
            atr_sma_now = data['atr_sma'].iloc[current_step]

            # 추세 조건: ADX가 임계값 이상이고, +DI와 -DI가 교차한 상태
            is_trending = False
            if adx_now > ADX_TRIGGER_THRESHOLD:
                if (plus_di_now > minus_di_now) or (minus_di_now > plus_di_now):
                    is_trending = True
            
            # 변동성 조건: 현재 ATR이 ATR의 이동평균보다 큰 상태
            is_volatile = atr_now > atr_sma_now

            trigger_activated = is_trending and is_volatile

            if trigger_activated:
                acct_state = AccountState(
                    u_loss=0, margin_usage=0.5
                )
                status = strategy.determine_next_action(long_pos, short_pos, acct_state, market_price,
                                                      plus_di_now, minus_di_now, adx_now)
            else:
                status = "HOLD"

            # 2. 조기 종료 처리
            if "EXIT" in status:
                final_pnl_strategy = (market_price - long_pos.entry_price) * long_pos.size + \
                                     (short_pos.entry_price - market_price) * short_pos.size
                break
        
        # 3. 결과 기록
        if final_pnl_strategy is None:
            final_market_price = data['Close'].iloc[i + SIMULATION_WINDOW - 1]
            final_pnl_strategy = ((final_market_price - long_pos.entry_price) * long_pos.size) + \
                                 ((short_pos.entry_price - final_market_price) * short_pos.size)

        baseline_pnl = ((data['Close'].iloc[i + SIMULATION_WINDOW - 1] - initial_long_entry) * INITIAL_POSITION_SIZE) + \
                       ((initial_short_entry - data['Close'].iloc[i + SIMULATION_WINDOW - 1]) * INITIAL_POSITION_SIZE)
                       
        label = 1 if final_pnl_strategy > baseline_pnl else -1
        
        results.append({
            'timestamp': data.index[i], 'baseline_pnl': baseline_pnl,
            'strategy_pnl': final_pnl_strategy, 'label': label, 'final_status': status
        })

    # 4. 결과 저장 및 출력
    results_df = pd.DataFrame(results)
    output_filename = f"defensive_simulation_results_{symbol}_final_v3.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"시뮬레이션 완료. 결과가 '{output_filename}'에 저장되었습니다.")
    print(results_df.head())
    print("\n레이블 분포:\n", results_df['label'].value_counts())
    print("\n종료 상태 분포:\n", results_df['final_status'].value_counts())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run final defensive exit strategy simulation.')
    parser.add_argument('--symbol', type=str, default='ETHUSDT',
                        help="The trading symbol to use (e.g., 'BTCUSDT', 'ETHUSDT')")
    args = parser.parse_args()
    run_simulation(args.symbol)
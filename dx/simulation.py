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
# 새로운 시장 상황 트리거 임계값
ATR_SPIKE_MULTIPLIER = 1.5 # 기준 ATR 대비 1.5배 이상 급등 시
ADX_TREND_THRESHOLD = 25.0 # ADX 25 이상일 때 강한 추세로 판단

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

    data.ta.atr(length=14, append=True)
    data.rename(columns={'ATRr_14': 'atr'}, inplace=True)
    
    # ADX/DMI 계산
    dmi_df = data.ta.adx(length=14)
    data = data.join(dmi_df)
    data.rename(columns={'ADX_14': 'adx'}, inplace=True) # 트리거에 ADX만 사용

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
            
            # 1. 방어 로직 트리거 조건 (새로운 로직)
            atr_now = data['atr'].iloc[current_step]
            atr_base = data['atr'].iloc[i]
            adx_now = data['adx'].iloc[current_step]

            trigger_activated = False
            if atr_now > atr_base * ATR_SPIKE_MULTIPLIER:
                trigger_activated = True
            elif adx_now > ADX_TREND_THRESHOLD:
                trigger_activated = True

            if trigger_activated:
                acct_state = AccountState(
                    u_loss=0, margin_usage=0.5,
                    atr_now=atr_now, atr_base=atr_base
                )
                status = strategy.determine_next_action(long_pos, short_pos, acct_state, market_price)
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
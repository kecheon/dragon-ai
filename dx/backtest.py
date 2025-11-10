import argparse
import pandas as pd
import pandas_ta as ta
import numpy as np
np.random.seed(42) # 백테스팅 재현성을 위해 난수 시드 고정
from tqdm import tqdm

# 우리가 만든 모듈들
from dx.data_loader import load_price_data
from dx.defensive_strategy import Position, AccountState, DynamicHedgeStrategy, StrategyMode
from dx.signal_generator import StrategyConfig, generate_signals

# 2. 백테스팅 설정
# 통합된 StrategyConfig 객체 생성
config = StrategyConfig(
    adx_threshold=15.0,
    atr_window=100,
    MaxBalancingAttempts=2,
    CycleStopLossRatio=-0.10,
    SpreadExitThreshold=0.1,
    LockedModePriority="DEFENSE"
)

def run_backtest(symbol: str):
    """
    연속적인 매매 사이클을 통해 동적 헤지 전략을 백테스팅합니다.
    """
    # 1. 데이터 로드 및 지표 계산
    print(f"'{symbol}'에 대한 데이터를 로드하고 기술적 지표를 계산합니다...")
    data = load_price_data(symbol)
    if data.empty:
        print("데이터가 없어 백테스팅을 중단합니다.")
        return

    # ADX/DMI 및 변동성(ATR) 지표 계산
    dmi_df = data.ta.adx(length=14)
    data = data.join(dmi_df)
    data.rename(columns={'ADX_14': 'adx', 'DMP_14': 'plus_di', 'DMN_14': 'minus_di'}, inplace=True)
    data.ta.atr(length=14, append=True)
    data.rename(columns={'ATRr_14': 'atr'}, inplace=True)
    

    
    # 신호 생성
    data = generate_signals(data, config)
    
    data.reset_index(inplace=True) # 인덱스를 리셋하여 정수 인덱스로 접근
    data.rename(columns={data.columns[0]: 'timestamp'}, inplace=True) # 첫 번째 컬럼(원래 인덱스)의 이름을 'timestamp'로 변경

    strategy = DynamicHedgeStrategy(config, logger=lambda x: None) # 백테스팅 중에는 로그 출력 끔
    
    # --- 백테스팅 주요 파라미터 ---
    FIXED_SPREAD = 0.05  # 5%
    INITIAL_POSITION_SIZE = 1.0

    # 기간 필터링
    data['timestamp'] = pd.to_datetime(data['timestamp']) # timestamp 컬럼을 datetime으로 변환
    if data.empty:
        print(f"데이터가 없어 백테스팅을 중단합니다.")
        return

    cycle_results = []
    current_step = 0
    
    print("백테스팅을 시작합니다...")
    with tqdm(total=len(data)) as pbar:
        while current_step < len(data) - 1:
            # --- 새로운 사이클 시작 ---
            
            # 1. 현실적인 진입 가격 설정
            price_variation = np.random.uniform(-0.04, 0.04)
            base_price = data['Close'].iloc[current_step] * (1 + price_variation)
            
            long_pos = Position(side="LONG", entry_price=base_price * (1 + FIXED_SPREAD / 2), size=INITIAL_POSITION_SIZE, initial_size=INITIAL_POSITION_SIZE)
            short_pos = Position(side="SHORT", entry_price=base_price * (1 - FIXED_SPREAD / 2), size=INITIAL_POSITION_SIZE, initial_size=INITIAL_POSITION_SIZE)
            
            cycle_start_step = current_step
            cycle_start_time = data['timestamp'].iloc[cycle_start_step]
            balancing_attempts = 0
            max_drawdown = 0
            peak_pnl = -np.inf

            # 사이클 최대 손실 한도 설정
            initial_value = (long_pos.entry_price * long_pos.initial_size) + (short_pos.entry_price * short_pos.initial_size)
            stop_loss_amount = initial_value * abs(config.CycleStopLossRatio)

            # 2. 단일 사이클 진행
            for j in range(current_step, len(data) - 1):
                step_in_cycle = j
                market_price = data['Close'].iloc[step_in_cycle]

                # PNL 및 MDD 추적
                current_pnl = ((market_price - long_pos.entry_price) * long_pos.size) + \
                              ((short_pos.entry_price - market_price) * short_pos.size)
                if current_pnl > peak_pnl:
                    peak_pnl = current_pnl
                drawdown = peak_pnl - current_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                # *** 최종 안전장치: 사이클 최대 손실 도달 시 즉시 종료 ***
                if current_pnl < -stop_loss_amount:
                    status = "EXIT_STOP_LOSS"
                    break
                
                # 전략 발동 조건
                trigger_activated = data['signal'].iloc[step_in_cycle]

                status = "HOLD"
                if trigger_activated:
                    mode_before = strategy._get_current_mode(long_pos, short_pos)
                    status = strategy.determine_next_action(
                        long_pos, short_pos, AccountState(0, 0), market_price,
                        data['plus_di'].iloc[step_in_cycle], data['minus_di'].iloc[step_in_cycle], 
                        data['adx'].iloc[step_in_cycle], balancing_attempts
                    )
                    mode_after = strategy._get_current_mode(long_pos, short_pos)
                    if mode_before == StrategyMode.IMBALANCED and mode_after == StrategyMode.LOCKED:
                        balancing_attempts += 1

                # 사이클 종료 조건 확인
                if "EXIT" in status:
                    break
            
            # 3. 사이클 결과 기록
            cycle_end_step = step_in_cycle
            final_pnl = current_pnl
            
            cycle_results.append({
                'start_time': cycle_start_time,
                'end_time': data['timestamp'].iloc[cycle_end_step],
                'duration_in_steps': cycle_end_step - cycle_start_step,
                'final_pnl': final_pnl,
                'max_drawdown': max_drawdown,
                'balancing_attempts': balancing_attempts,
                'exit_status': status
            })

            # 다음 사이클 시작 위치로 이동 및 pbar 업데이트
            pbar.update(cycle_end_step - current_step + 1)
            current_step = cycle_end_step + 1

    # 4. 최종 결과 분석 및 저장
    results_df = pd.DataFrame(cycle_results)
    output_filename = f"backtest_results_{symbol}.csv"
    results_df.to_csv(output_filename, index=False)

    print(f"\n백테스팅 완료. 결과가 '{output_filename}'에 저장되었습니다.")
    
    # 종합 성과 지표 출력
    total_pnl = results_df['final_pnl'].sum()
    total_cycles = len(results_df)
    winning_cycles = len(results_df[results_df['final_pnl'] > 0])
    win_rate = (winning_cycles / total_cycles) * 100 if total_cycles > 0 else 0
    avg_pnl = results_df['final_pnl'].mean()
    avg_duration = results_df['duration_in_steps'].mean()
    
    print("\n--- 백테스팅 종합 결과 ---")
    print(f"전체 기간: {data['timestamp'].iloc[0]} ~ {data['timestamp'].iloc[-1]}")
    print(f"총 매매 사이클: {total_cycles}회")
    print(f"총 손익: {total_pnl:.2f}")
    print(f"승률: {win_rate:.2f}%")
    print(f"평균 손익 (사이클 당): {avg_pnl:.4f}")
    print(f"평균 보유 기간 (5분봉 기준): {avg_duration:.2f} 스텝")
    print("--------------------------")
    print("\n종료 상태 분포:")
    print(results_df['exit_status'].value_counts())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run continuous backtest for the dynamic hedge strategy.')
    parser.add_argument('--symbol', type=str, default='ETHUSDT',
                        help="The trading symbol to use (e.g., 'BTCUSDT', 'ETHUSDT')")
    args = parser.parse_args()
    run_backtest(args.symbol)
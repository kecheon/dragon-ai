import argparse
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

# 우리가 만든 모듈들
from data_loader import load_price_data
from defensive_strategy import StrategyConfig, Position, AccountState, DynamicHedgeStrategy, StrategyMode

# 시뮬레이션을 위한 글로벌 설정 (트리거 조건 추가)
INITIAL_POSITION_SIZE = 1.0
FIXED_SPREAD = 0.02 # 왜 이값이 클수록 성과가  좋게 나오냐? 넉넉한 스프레드가 있어야 안심하고 물타기 되는 건가
# 0.01 5891/3626
# 0.02 5958/3557
# 0.03 6173/3344
# 0.04 6378/3139
# 0.05 6389/3128
# 0.06 0.04 보다 못함
LOOKBACK_PERIOD = 120
SIMULATION_WINDOW = 300
# 새로운 시장 상황 트리거 임계값 (DMI 기반)
ADX_TRIGGER_THRESHOLD = 15.0 # ADX가 이 값 이상일 때 추세 강하다고 판단
DMI_CROSSOVER_THRESHOLD = 0 # +DI와 -DI의 교차 또는 우위 판단 기준 (0이면 단순 교차)

ATR_WINDOW = 50

def run_simulation(symbol: str, adx_threshold: float, atr_window: int, fixed_spread: float, max_balancing_attempts: int):
    """
    주어진 파라미터로 단일 시뮬레이션을 실행하고 결과를 반환합니다.
    """
    # 1. 데이터 로드 및 특성 계산
    data = load_price_data(symbol)
    if data.empty:
        return None

    dmi_df = data.ta.adx(length=14)
    data = data.join(dmi_df)
    data.rename(columns={'ADX_14': 'adx', 'DMP_14': 'plus_di', 'DMN_14': 'minus_di'}, inplace=True)
    data.ta.atr(length=14, append=True)
    data.rename(columns={'ATRr_14': 'atr'}, inplace=True)
    data['atr_sma'] = data['atr'].rolling(window=atr_window).mean()
    data.dropna(inplace=True)
    
    config = StrategyConfig(MaxBalancingAttempts=max_balancing_attempts)
    strategy = DynamicHedgeStrategy(config, logger=lambda x: None)
    
    results = []
    lookback_period = 120
    simulation_window = 300
    initial_position_size = 1.0

    for i in range(lookback_period, len(data) - simulation_window):
        base_price = data['Close'].iloc[i - lookback_period]
        
        long_pos = Position(side="LONG", entry_price=base_price * (1 + fixed_spread / 2), size=initial_position_size, initial_size=initial_position_size)
        short_pos = Position(side="SHORT", entry_price=base_price * (1 - fixed_spread / 2), size=initial_position_size, initial_size=initial_position_size)
        
        initial_long_entry = long_pos.entry_price
        initial_short_entry = short_pos.entry_price
        
        final_pnl_strategy = None
        status = "HOLD"
        balancing_attempts = 0

        for j in range(simulation_window):
            current_step = i + j
            market_price = data['Close'].iloc[current_step]
            
            adx_now = data['adx'].iloc[current_step]
            is_trending = adx_now > adx_threshold
            is_volatile = data['atr'].iloc[current_step] > data['atr_sma'].iloc[current_step]
            trigger_activated = is_trending and is_volatile

            if trigger_activated:
                mode_before = strategy._get_current_mode(long_pos, short_pos)
                status = strategy.determine_next_action(
                    long_pos, short_pos, AccountState(0, 0), market_price,
                    data['plus_di'].iloc[current_step], data['minus_di'].iloc[current_step], adx_now, balancing_attempts
                )
                mode_after = strategy._get_current_mode(long_pos, short_pos)
                if mode_before == StrategyMode.IMBALANCED and mode_after == StrategyMode.LOCKED:
                    balancing_attempts += 1
            else:
                status = "HOLD"

            if "EXIT" in status or ("STRATEGIC_CUT" in status and "NO_ACTION" not in status):
                final_pnl_strategy = ((market_price - long_pos.entry_price) * long_pos.size) + \
                                     ((short_pos.entry_price - market_price) * short_pos.size)
                break
        
        if final_pnl_strategy is None:
            final_market_price = data['Close'].iloc[i + simulation_window - 1]
            final_pnl_strategy = ((final_market_price - long_pos.entry_price) * long_pos.size) + \
                                 ((short_pos.entry_price - final_market_price) * short_pos.size)

        baseline_pnl = ((data['Close'].iloc[i + simulation_window - 1] - initial_long_entry) * initial_position_size) + \
                       ((initial_short_entry - data['Close'].iloc[i + simulation_window - 1]) * initial_position_size)
                       
        label = 1 if final_pnl_strategy > baseline_pnl else -1
        
        results.append({'label': label})

    results_df = pd.DataFrame(results)
    label_counts = results_df['label'].value_counts()
    
    return {
        'adx_threshold': adx_threshold,
        'atr_window': atr_window,
        'fixed_spread': fixed_spread,
        'max_balancing_attempts': max_balancing_attempts,
        'win_count': label_counts.get(1, 0),
        'loss_count': label_counts.get(-1, 0),
        'total_count': len(results_df),
        'win_rate': (label_counts.get(1, 0) / len(results_df)) * 100 if len(results_df) > 0 else 0
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sensitivity analysis for the dynamic hedge strategy.')
    parser.add_argument('--symbol', type=str, default='ETH',
                        help="The trading symbol to use (e.g., 'BTC', 'ETH')")
    args = parser.parse_args()

    # --- 민감도 분석을 위한 파라미터 범위 설정 ---
    adx_thresholds = [15, 20, 25]
    atr_windows = [50, 100, 150]
    fixed_spreads = [0.02, 0.03, 0.04]
    max_balancing_attempts_list = [3, 5, 7]

    all_results = []
    
    total_combinations = len(adx_thresholds) * len(atr_windows) * len(fixed_spreads) * len(max_balancing_attempts_list)
    pbar = tqdm(total=total_combinations, desc="Sensitivity Analysis")

    print(f"'{args.symbol}'에 대한 민감도 분석을 시작합니다. 총 {total_combinations}개의 조합을 테스트합니다.")

    for adx in adx_thresholds:
        for atr in atr_windows:
            for spread in fixed_spreads:
                for attempts in max_balancing_attempts_list:
                    result = run_simulation(
                        symbol=args.symbol,
                        adx_threshold=adx,
                        atr_window=atr,
                        fixed_spread=spread,
                        max_balancing_attempts=attempts
                    )
                    if result:
                        all_results.append(result)
                    pbar.update(1)
    
    pbar.close()

    # --- 최종 결과 집계 및 출력 ---
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.sort_values(by='win_rate', ascending=False)
        
        output_filename = f"sensitivity_analysis_results_{args.symbol}.csv"
        summary_df.to_csv(output_filename, index=False)

        print(f"\n민감도 분석 완료. 결과가 '{output_filename}'에 저장되었습니다.")
        print("\n--- 상위 5개 최적 파라미터 조합 ---")
        print(summary_df.head(5).to_string())
    else:
        print("분석을 위한 시뮬레이션 결과가 없습니다.")
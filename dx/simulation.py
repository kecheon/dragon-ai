import argparse
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

# 우리가 만든 모듈들
from dx.data_loader import load_price_data
from dx.defensive_strategy import Position, AccountState, DynamicHedgeStrategy, StrategyMode
from dx.signal_generator import StrategyConfig, generate_initial_signals

def run_simulation(symbol: str, config: StrategyConfig, fixed_spread: float):
    """
    주어진 파라미터로 단일 시뮬레이션을 실행하고 결과를 반환합니다.
    """
    # 1. 데이터 로드 및 기본 지표 계산
    data = load_price_data(symbol)
    if data.empty:
        return None

    dmi_df = data.ta.adx(length=14)
    data = data.join(dmi_df)
    data.rename(columns={'ADX_14': 'adx', 'DMP_14': 'plus_di', 'DMN_14': 'minus_di'}, inplace=True)
    data.ta.atr(length=14, append=True)
    data.rename(columns={'ATRr_14': 'atr'}, inplace=True)
    
    # 2. 신호 생성
    data = generate_initial_signals(data, config)
    
    # 3. 시뮬레이션 설정
    strategy = DynamicHedgeStrategy(config, logger=lambda x: None)
    
    results = []
    lookback_period = 120
    simulation_window = 300
    initial_position_size = 1.0

    for i in range(len(data) - simulation_window):
        # lookback_period가 generate_signals에서 처리되었으므로, 인덱스를 0부터 시작
        if i < lookback_period: continue

        base_price = data['Close'].iloc[i - lookback_period]
        
        long_pos = Position(side="LONG", entry_price=base_price * (1 + fixed_spread / 2), size=initial_position_size, initial_size=initial_position_size)
        short_pos = Position(side="SHORT", entry_price=base_price * (1 - fixed_spread / 2), size=initial_position_size, initial_size=initial_position_size)
        
        initial_long_entry = long_pos.entry_price
        initial_short_entry = short_pos.entry_price
        
        final_pnl_strategy = None
        status = "HOLD"
        balancing_attempts = 0
        cycle_realized_pnl = 0.0

        for j in range(simulation_window):
            current_step = i + j
            if current_step >= len(data): break
            
            market_price = data['Close'].iloc[current_step]
            
            trigger_activated = data['trigger'].iloc[current_step]

            if trigger_activated:
                mode_before = strategy._get_current_mode(long_pos, short_pos)
                status, pnl_from_action = strategy.determine_next_action(
                    long_pos, short_pos, AccountState(u_loss=0, margin_usage=0.5), market_price,
                    data['plus_di'].iloc[current_step], data['minus_di'].iloc[current_step], 
                    data['adx'].iloc[current_step], balancing_attempts, cycle_realized_pnl
                )
                cycle_realized_pnl += pnl_from_action
                
                mode_after = strategy._get_current_mode(long_pos, short_pos)
                if mode_before == StrategyMode.IMBALANCED and mode_after == StrategyMode.LOCKED:
                    balancing_attempts += 1
            else:
                status = "HOLD"

            if "EXIT" in status:
                final_unrealized_pnl = ((market_price - long_pos.entry_price) * long_pos.size) + \
                                       ((short_pos.entry_price - market_price) * short_pos.size)
                final_pnl_strategy = final_unrealized_pnl + cycle_realized_pnl
                break
        
        if final_pnl_strategy is None:
            final_market_price = data['Close'].iloc[i + simulation_window - 1]
            final_unrealized_pnl = ((final_market_price - long_pos.entry_price) * long_pos.size) + \
                                   ((short_pos.entry_price - final_market_price) * short_pos.size)
            final_pnl_strategy = final_unrealized_pnl + cycle_realized_pnl

        baseline_pnl = ((data['Close'].iloc[i + simulation_window - 1] - initial_long_entry) * initial_position_size) + \
                       ((initial_short_entry - data['Close'].iloc[i + simulation_window - 1]) * initial_position_size)
                       
        label = 1 if final_pnl_strategy > baseline_pnl else -1
        
        results.append({'label': label})

    results_df = pd.DataFrame(results)
    label_counts = results_df['label'].value_counts()
    
    return {
        'adx_threshold': config.adx_threshold,
        'atr_window': config.atr_window,
        'fixed_spread': fixed_spread,
        'max_balancing_attempts': config.MaxBalancingAttempts,
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
    # adx_thresholds = [15, 20, 25]
    # atr_windows = [50, 100, 150]
    # fixed_spreads = [0.01, 0.03, 0.05]
    # max_balancing_attempts_list = [2, 3, 5]
    adx_thresholds = [20]
    atr_windows = [100]
    fixed_spreads = [0.03]
    max_balancing_attempts_list = [2]
   
    all_results = []
    
    param_combinations = [
        (adx, atr, spread, attempts)
        for adx in adx_thresholds
        for atr in atr_windows
        for spread in fixed_spreads
        for attempts in max_balancing_attempts_list
    ]
    
    print(f"'{args.symbol}'에 대한 민감도 분석을 시작합니다. 총 {len(param_combinations)}개의 조합을 테스트합니다.")

    for adx, atr, spread, attempts in tqdm(param_combinations, desc="Sensitivity Analysis"):
        # 통합된 StrategyConfig 객체 생성
        config = StrategyConfig(
            adx_threshold=adx,
            atr_window=atr,
            MaxBalancingAttempts=attempts
        )
        result = run_simulation(
            symbol=args.symbol,
            config=config, # 통합된 config 객체 전달
            fixed_spread=spread
        )
        if result:
            all_results.append(result)
    
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
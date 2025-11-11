import argparse
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

# 우리가 만든 모듈들
from dx.data_loader import load_price_data
from dx.defensive_strategy import (
    Position,
    AccountState,
    DynamicHedgeStrategy,
    StrategyMode,
)
from dx.signal_generator import StrategyConfig, generate_initial_signals


def create_simulation_dataset(symbol: str, config: StrategyConfig, fixed_spread: float):
    """
    주어진 파라미터로 시뮬레이션을 실행하고 머신러닝 학습용 데이터셋을 생성합니다.
    """
    # 1. 데이터 로드 및 기본 지표 계산
    data = load_price_data(symbol)
    if data.empty:
        return None

    dmi_df = data.ta.adx(length=14)
    data = data.join(dmi_df)
    data.rename(
        columns={"ADX_14": "adx", "DMP_14": "plus_di", "DMN_14": "minus_di"},
        inplace=True,
    )
    data.ta.atr(length=14, append=True)
    data.rename(columns={"ATRr_14": "atr"}, inplace=True)

    # 2. 신호 생성
    data = generate_initial_signals(data, config)

    # 3. 시뮬레이션 설정
    strategy = DynamicHedgeStrategy(config, logger=lambda x: None)

    dataset_rows = []
    lookback_period = 120  # 안정적인 지표 계산을 위해 필요한 최소 데이터 기간
    simulation_window = 300  # 단일 시뮬레이션을 진행할 기간 (5분봉 기준)
    initial_position_size = 1.0

    # tqdm을 사용하여 데이터셋 생성 진행 상황 표시
    for i in tqdm(range(len(data) - simulation_window), desc="Generating dataset"):
        if i < lookback_period:
            continue

        # --- 1. 피처 추출 ---
        # 시뮬레이션 시작 시점(의사결정 시점)의 시장 상황을 피처로 추출
        features = {
            "adx": data["adx"].iloc[i],
            "plus_di": data["plus_di"].iloc[i],
            "minus_di": data["minus_di"].iloc[i],
            "atr": data["atr"].iloc[i],
        }

        # --- 2. 단일 시뮬레이션 실행 ---
        base_price = data["Close"].iloc[i]  # 현재 가격을 기준으로 포지션 진입

        long_pos = Position(
            side="LONG",
            entry_price=base_price * (1 + fixed_spread / 2),
            size=initial_position_size,
            initial_size=initial_position_size,
        )
        short_pos = Position(
            side="SHORT",
            entry_price=base_price * (1 - fixed_spread / 2),
            size=initial_position_size,
            initial_size=initial_position_size,
        )

        initial_long_entry = long_pos.entry_price
        initial_short_entry = short_pos.entry_price

        final_pnl_strategy = None
        status = "HOLD"
        cycle_exit_status = "END_OF_WINDOW"  # 기본 종료 상태
        balancing_attempts = 0
        cycle_realized_pnl = 0.0

        for j in range(simulation_window):
            current_step = i + j
            if current_step >= len(data):
                break

            market_price = data["Close"].iloc[current_step]
            trigger_activated = data["trigger"].iloc[current_step]

            if trigger_activated:
                mode_before = strategy._get_current_mode(long_pos, short_pos)
                status, pnl_from_action = strategy.determine_next_action(
                    long_pos,
                    short_pos,
                    AccountState(u_loss=0, margin_usage=0.5),
                    market_price,
                    data["plus_di"].iloc[current_step],
                    data["minus_di"].iloc[current_step],
                    data["adx"].iloc[current_step],
                    balancing_attempts,
                    cycle_realized_pnl,
                )
                cycle_realized_pnl += pnl_from_action

                mode_after = strategy._get_current_mode(long_pos, short_pos)
                if (
                    mode_before == StrategyMode.IMBALANCED
                    and mode_after == StrategyMode.LOCKED
                ):
                    balancing_attempts += 1
            else:
                status = "HOLD"

            if "EXIT" in status:
                cycle_exit_status = status  # 실제 종료 상태로 업데이트
                final_unrealized_pnl = (
                    (market_price - long_pos.entry_price) * long_pos.size
                ) + ((short_pos.entry_price - market_price) * short_pos.size)
                final_pnl_strategy = final_unrealized_pnl + cycle_realized_pnl
                break

        if final_pnl_strategy is None:
            final_market_price = data["Close"].iloc[i + simulation_window - 1]
            final_unrealized_pnl = (
                (final_market_price - long_pos.entry_price) * long_pos.size
            ) + ((short_pos.entry_price - final_market_price) * short_pos.size)
            final_pnl_strategy = final_unrealized_pnl + cycle_realized_pnl

        # --- 3. 레이블 계산 ---
        # 단순 보유(baseline) 전략 대비 성과를 기준으로 레이블 결정
        baseline_pnl = (
            (data["Close"].iloc[i + simulation_window - 1] - initial_long_entry)
            * initial_position_size
        ) + (
            (initial_short_entry - data["Close"].iloc[i + simulation_window - 1])
            * initial_position_size
        )
        label = 1 if final_pnl_strategy > baseline_pnl else -1

        # --- 4. 피처와 레이블 결합 ---
        features["label"] = label
        features["exit_status"] = cycle_exit_status
        dataset_rows.append(features)

    return pd.DataFrame(dataset_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dataset for ML model training from simulation."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="ETHUSDT",
        help="The trading symbol to use (e.g., 'BTCUSDT', 'ETHUSDT')",
    )
    args = parser.parse_args()

    # --- 단일 실행을 위한 파라미터 설정 ---
    FIXED_SPREAD = 0.03
    config = StrategyConfig(
        adx_threshold=20.0,
        atr_window=100,
        MaxBalancingAttempts=2,
        CycleStopLossRatio=-0.10,
        SpreadExitThreshold=0.1,
        LockedModePriority="ATTACK",
        ReversalStopLossRatio=-0.1,
    )

    print(f"'{args.symbol}'에 대한 시뮬레이션을 시작하여 데이터셋을 생성합니다.")

    # --- 데이터셋 생성 함수 실행 ---
    dataset_df = create_simulation_dataset(
        symbol=args.symbol,
        config=config,
        fixed_spread=FIXED_SPREAD,
    )

    # --- 최종 결과(데이터셋) 저장 및 통계 출력 ---
    if dataset_df is not None and not dataset_df.empty:
        output_filename = f"simulation_dataset_{args.symbol}.csv"
        dataset_df.to_csv(output_filename, index=False)
        
        total_simulations = len(dataset_df)
        win_count = (dataset_df["label"] == 1).sum()
        win_rate = (win_count / total_simulations) * 100 if total_simulations > 0 else 0

        print(f"\n데이터셋 생성 완료. 결과가 '{output_filename}'에 저장되었습니다.")
        print("\n--- 시뮬레이션 종합 통계 ---")
        print(f"총 시뮬레이션 횟수: {total_simulations}회")
        print(f"승리(Label=1) 횟수: {win_count}회")
        print(f"패배(Label=-1) 횟수: {total_simulations - win_count}회")
        print(f"전체 승률: {win_rate:.2f}%")
        
        print("\n--- 종료 상태별 상세 통계 ---")
        exit_stats = dataset_df.groupby("exit_status")["label"].agg(
            count="size",
            wins=lambda x: (x == 1).sum()
        ).reset_index()
        exit_stats["win_rate( %)"] = (exit_stats["wins"] / exit_stats["count"]) * 100
        exit_stats = exit_stats.sort_values(by="count", ascending=False)
        print(exit_stats.to_string(index=False))

    else:
        print("데이터셋을 생성하지 못했습니다.")
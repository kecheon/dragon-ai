import argparse
import pandas as pd
import pandas_ta as ta

# 우리가 만든 모듈들
from data_loader import load_price_data
from defensive_strategy import StrategyConfig, Position, AccountState, DefensiveStrategy

# 시뮬레이션 설정
INITIAL_POSITION_SIZE = 1.0
FIXED_SPREAD = 0.01
LOOKBACK_PERIOD = 12
SIMULATION_WINDOW = 288
DEBUG_INDEX = 100 # 검사할 특정 시작 인덱스

def run_debug_simulation(symbol: str):
    """
    방어적 출구 전략의 단일 인스턴스를 상세 로깅과 함께 실행합니다.
    """
    # 1. 데이터 로드 및 특성 계산
    print(f"'{symbol}'에 대한 데이터를 로드하고 특성을 계산합니다...")
    data = load_price_data(symbol)
    if data.empty:
        print("데이터가 없어 시뮬레이션을 중단합니다.")
        return

    data.ta.atr(length=14, append=True)
    data.rename(columns={'ATRr_14': 'atr'}, inplace=True)
    data.dropna(inplace=True)
    
    # 2. 상세 로그를 출력하도록 전략 설정
    config = StrategyConfig()
    # print 함수를 로거로 직접 주입하여 모든 내부 메시지를 확인
    strategy = DefensiveStrategy(config, logger=print)
    
    print(f"\n--- 인덱스 {DEBUG_INDEX}에 대한 디버그 시뮬레이션 시작 ---")
    
    # 3. 가상 초기 포지션 설정
    i = LOOKBACK_PERIOD + DEBUG_INDEX
    base_price = data['Close'].iloc[i - LOOKBACK_PERIOD]
    
    long_pos = Position(side="LONG", entry_price=base_price * (1 + FIXED_SPREAD / 2), size=INITIAL_POSITION_SIZE)
    short_pos = Position(side="SHORT", entry_price=base_price * (1 - FIXED_SPREAD / 2), size=INITIAL_POSITION_SIZE)
    
    print(f"초기 설정 ({data.index[i]}):")
    print(f"  - 기준 가격: {base_price:.4f}")
    print(f"  - 롱 포지션: 진입가={long_pos.entry_price:.4f}, 크기={long_pos.size}")
    print(f"  - 숏 포지션: 진입가={short_pos.entry_price:.4f}, 크기={short_pos.size}")
    print("-" * 20)

    # 4. 단일 케이스에 대한 시뮬레이션 루프 실행
    for j in range(SIMULATION_WINDOW):
        current_step = i + j
        market_price = data['Close'].iloc[current_step]
        
        long_pnl = (market_price - long_pos.entry_price) * long_pos.size
        short_pnl = (short_pos.entry_price - market_price) * short_pos.size
        total_pnl = long_pnl + short_pnl
        
        print(f"\n[단계 {j+1}/{SIMULATION_WINDOW}] 시간: {data.index[current_step]}, 시장가: {market_price:.4f}")
        print(f"현재 PNL -> 롱: {long_pnl:.4f}, 숏: {short_pnl:.4f}, 합계: {total_pnl:.4f}")

        acct_state = AccountState(
            u_loss=-total_pnl if total_pnl < 0 else 0,
            margin_usage=0.5, # 단순화된 값, 영향 관찰용
            atr_now=data['atr'].iloc[current_step],
            atr_base=data['atr'].iloc[i]
        )
        
        # 총 PNL이 마이너스일 때만 방어 전략 호출
        if total_pnl < 0:
            status = strategy.defensive_loop_step(long_pos, short_pos, acct_state, market_price)
            print(f"전략 상태: {status}")
            print(f"업데이트된 포지션 -> 롱 진입가: {long_pos.entry_price:.4f}, 숏 진입가: {short_pos.entry_price:.4f}")
        else:
            print("전략 상태: NO_ACTION (PNL이 0 이상)")

        # PNL이 0 이상이 되면 루프 중단
        if total_pnl >= 0 and j > 0:
            print("\n--- 시뮬레이션 종료: 총 PNL이 0 이상이 되었습니다. ---")
            break
            
    print("\n--- 디버그 시뮬레이션이 끝났습니다. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a debug version of the defensive exit strategy simulation.')
    parser.add_argument('--symbol', type=str, default='SOLUSDT',
                        help="The trading symbol to use (e.g., 'BTCUSDT', 'SOLUSDT')")
    args = parser.parse_args()
    
    run_debug_simulation(args.symbol)

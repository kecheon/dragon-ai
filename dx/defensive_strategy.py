from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum, auto

# --- 전략 모드 정의 ---
class StrategyMode(Enum):
    LOCKED = auto()      # 포지션 수량이 동일한 상태 (잠금 모드)
    IMBALANCED = auto()  # 포지션 수량이 다른 상태 (불균형 모드)

# --- 최종 전략 설정 ---
@dataclass
class StrategyConfig:
    SlipTolerance: float = 0.002
    MaxAllowedΔM: float = 5
    EpsilonLoss: float = 0.002
    ATRSpikeFactor: float = 1.5
    StepCount: int = 4
    MaxUtilization: float = 0.8
    FeeRate: float = 0.0005
    MaxBalancingAttempts: int = 2 # 균형화 시도 횟수 한도
    # 신규 종료 조건 임계값
    SpreadExitThreshold: float = 0.1  # 스프레드가 이 값보다 작아지면 동시 청산

# --- 데이터 구조 ---
@dataclass
class Position:
    side: str
    entry_price: float
    size: float
    initial_size: float # 포지션 불균형 상태를 추적하기 위함

@dataclass
class AccountState:
    u_loss: float
    margin_usage: float

@dataclass
class SimulationResult:
    new_entry: float
    dS: float
    d_u_loss: float
    d_margin: float

# --- 최종 전략 클래스 ---
class DynamicHedgeStrategy:
    def __init__(self, config: StrategyConfig, logger=print):
        self.config = config
        self.log = logger

    # --- 시뮬레이션 헬퍼 함수 ---
    def propose_qs(self, pos: Position) -> List[float]:
        """물타기/불타기 수량을 현재 포지션 크기의 50%로 고정하여 제안"""
        return [pos.size * 0.5]

    def get_est_exec_price(self, side: str, market_price: float) -> float:
        """슬리피지를 고려한 예상 체결가를 계산하는 함수"""
        slip_factor = (1 + self.config.SlipTolerance) if side == "LONG" else (1 - self.config.SlipTolerance)
        return market_price * slip_factor

    # --- 핵심 로직 ---
    def simulate_averaging(self, pos: Position, q: float, exec_price: float, other_entry: float) -> SimulationResult:
        """특정 행동이 포지션에 미칠 영향을 시뮬레이션하는 함수"""
        new_entry = (pos.entry_price * pos.size + exec_price * q) / (pos.size + q)
        dS = abs(new_entry - other_entry) - abs(pos.entry_price - other_entry)
        d_u_loss = (new_entry - pos.entry_price) * pos.size if pos.side == "SHORT" else (pos.entry_price - new_entry) * pos.size
        d_margin = (exec_price * q) / 10  # 단순화된 마진 계산
        return SimulationResult(new_entry=new_entry, dS=dS, d_u_loss=d_u_loss, d_margin=d_margin)

    def meets_financial_criteria(self, sim_res: SimulationResult, acct: AccountState) -> bool:
        """행동의 재무적/변동성 조건을 검사하는 함수"""
        # if sim_res.d_u_loss > -self.config.EpsilonLoss and sim_res.d_margin > self.config.MaxAllowedΔM:
        #     self.log(f"  - 행동 거부: 재무적 영향(손실/마진)이 너무 큽니다.{sim_res.d_u_loss} {sim_res.d_margin}")
        #     return True
        # if acct.atr_now > acct.atr_base * self.config.ATRSpikeFactor:
        #     # self.log("  - 행동 거부: 변동성 급등이 감지되었습니다.")
        #     return False
        return True

    def execute_averaging(self, pos: Position, total_q: float, market_price: float) -> None:
        """시뮬레이션 상에서 포지션 상태를 업데이트하는 함수"""
        exec_price = self.get_est_exec_price(pos.side, market_price)
        new_size = pos.size + total_q
        new_entry_price = (pos.entry_price * pos.size + exec_price * total_q) / new_size
        pos.entry_price = new_entry_price
        pos.size = new_size
        # self.log(f"  => 실행: {pos.side}에 {total_q:.4f} 추가. 새 진입가: {pos.entry_price:.4f}, 새 크기: {pos.size:.4f}")

    def _get_current_mode(self, long_pos: Position, short_pos: Position) -> StrategyMode:
        """포지션 수량을 비교하여 현재 전략 모드를 반환"""
        # 부동소수점 비교를 위해 작은 허용 오차(epsilon) 사용
        if abs(long_pos.size - short_pos.size) < 1e-9:
            return StrategyMode.LOCKED
        return StrategyMode.IMBALANCED

    def execute_partial_close(self, pos: Position, q_to_close: float, market_price: float) -> float:
        """지정된 수량만큼 포지션을 부분 청산하고 실현 손익을 반환"""
        # 부분 청산을 위한 예상 체결가는 반대 방향의 슬리피지를 적용
        exec_price = self.get_est_exec_price("SHORT" if pos.side == "LONG" else "LONG", market_price)
        realized_pnl = (exec_price - pos.entry_price) * q_to_close if pos.side == "LONG" else (pos.entry_price - exec_price) * q_to_close
        pos.size -= q_to_close
        self.log(f"  => 부분 청산 실행: {pos.side} {q_to_close:.4f} 계약 청산. 실현 손익: {realized_pnl:.4f}, 남은 수량: {pos.size:.4f}")
        return realized_pnl

    def determine_next_action(self, long_pos: Position, short_pos: Position, acct: AccountState, market_price: float,
                              plus_di_now: float, minus_di_now: float, adx_now: float, balancing_attempts: int) -> str:
        """상태에 따라 다음 행동을 결정하는 메인 로직 함수"""
        total_pnl = ((market_price - long_pos.entry_price) * long_pos.size) + \
                    ((short_pos.entry_price - market_price) * short_pos.size)
        current_mode = self._get_current_mode(long_pos, short_pos)
        current_spread = abs(long_pos.entry_price - short_pos.entry_price)

        # 1. 종료 조건 (최우선)
        if current_mode == StrategyMode.IMBALANCED and total_pnl > 0:
            self.log(f"EXIT: 불균형 상태에서 수익({total_pnl:.4f}) 전환. 포지션 동시 청산.")
            return "EXIT_PROFIT_TARGET_MET"
        if current_spread < self.config.SpreadExitThreshold and total_pnl > 0:
            self.log(f"EXIT: 스프레드 목표({self.config.SpreadExitThreshold:.4f}) 달성 및 수익 발생. 포지션 동시 청산.")
            return "EXIT_SPREAD_TARGET_MET"

        # 2. 행동 결정
        # 2a. [잠금 모드]에서의 행동: 기회 포착
        if current_mode == StrategyMode.LOCKED:
            # 부분 익절 로직 (기존과 동일)
            long_pnl_per_unit = market_price - long_pos.entry_price
            short_pnl_per_unit = short_pos.entry_price - market_price
            if long_pnl_per_unit > 0 and minus_di_now > plus_di_now:
                q_to_close = long_pos.size * 0.5
                if q_to_close > 1e-9:
                    self.execute_partial_close(long_pos, q_to_close, market_price)
                    return "ACTION_PARTIAL_CLOSE_LONG"
            if short_pnl_per_unit > 0 and plus_di_now > minus_di_now:
                q_to_close = short_pos.size * 0.5
                if q_to_close > 1e-9:
                    self.execute_partial_close(short_pos, q_to_close, market_price)
                    return "ACTION_PARTIAL_CLOSE_SHORT"
            
            # 공격적 물타기 로직 (기존과 동일)
            valid_actions: List[Dict[str, Any]] = []
            if plus_di_now > minus_di_now:
                pos_to_avg, other_pos_entry, action_type = long_pos, short_pos.entry_price, "AVG_LONG_TREND"
            elif minus_di_now > plus_di_now:
                pos_to_avg, other_pos_entry, action_type = short_pos, long_pos.entry_price, "AVG_SHORT_TREND"
            else:
                return "NO_ACTION_LOCKED_NO_TREND"
            
            for q in self.propose_qs(pos_to_avg):
                sim_res = self.simulate_averaging(pos_to_avg, q, self.get_est_exec_price(pos_to_avg.side, market_price), other_pos_entry)
                if self.meets_financial_criteria(sim_res, acct):
                    valid_actions.append({'type': action_type, 'q': q, 'dS': sim_res.dS, 'pos_to_avg': pos_to_avg})
            
            if not valid_actions: return "NO_VALID_ACTION"
            best_action = min(valid_actions, key=lambda x: x['dS'])
            self.execute_averaging(best_action['pos_to_avg'], best_action['q'], market_price)
            return f"ACTION_{best_action['type']}"

        # 2b. [불균형 모드]에서의 행동: 수익 실현 또는 위험 관리
        elif current_mode == StrategyMode.IMBALANCED:
            if total_pnl > 0: # 이 조건은 함수 시작 부분에서 이미 처리되었지만, 명확성을 위해 남겨둠
                return "EXIT_PROFIT_TARGET_MET" 
            
            # PNL이 음수일 때, 균형화 시도 횟수에 따라 행동 결정
            if balancing_attempts < self.config.MaxBalancingAttempts:
                # [행동] 방어적 균형화 (기존 로직)
                if long_pos.size < short_pos.size:
                    pos_to_avg, other_pos_entry, action_type = long_pos, short_pos.entry_price, "DEFENSIVE_AVG_LONG"
                elif short_pos.size < long_pos.size:
                    pos_to_avg, other_pos_entry, action_type = short_pos, long_pos.entry_price, "DEFENSIVE_AVG_SHORT"
                else:
                    return "NO_ACTION_IMBALANCED_ALMOST_LOCKED"
                
                q = self.propose_qs(pos_to_avg)[0] # 간단하게 첫번째 제안 사용
                sim_res = self.simulate_averaging(pos_to_avg, q, self.get_est_exec_price(pos_to_avg.side, market_price), other_pos_entry)
                if sim_res.dS < 0 and self.meets_financial_criteria(sim_res, acct):
                    self.execute_averaging(pos_to_avg, q, market_price)
                    return f"ACTION_{action_type}"
                else:
                    return "NO_VALID_ACTION_DEFENSIVE_AVG"

            else:
                # [행동] Forced Cut (의도적 불균형 심화 및 손절)
                self.log(f"  - Forced Cut 발동 (시도 횟수: {balancing_attempts})")
                q_to_close = 0
                # 추세에 불리한 포지션을 손절
                if plus_di_now > minus_di_now: # 상승 추세 -> 숏 포지션이 불리
                    pos_to_cut = short_pos
                    q_to_close = pos_to_cut.size * 0.5
                    if q_to_close > 1e-9:
                        self.execute_partial_close(pos_to_cut, q_to_close, market_price)
                        return "ACTION_FORCED_CUT_SHORT"
                elif minus_di_now > plus_di_now: # 하락 추세 -> 롱 포지션이 불리
                    pos_to_cut = long_pos
                    q_to_close = pos_to_cut.size * 0.5
                    if q_to_close > 1e-9:
                        self.execute_partial_close(pos_to_cut, q_to_close, market_price)
                        return "ACTION_FORCED_CUT_LONG"
                
                return "NO_ACTION_FORCED_CUT_NO_TREND"

        return "NO_ACTION" # 모든 조건에 해당하지 않을 경우

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
    MaxBalancingAttempts: int = 5 # 균형화 시도 횟수 한도
    ForcedCutRatio: float = 0.25 # Forced Cut 시 청산할 포지션 비율
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
        return True

    def execute_averaging(self, pos: Position, total_q: float, market_price: float) -> None:
        """시뮬레이션 상에서 포지션 상태를 업데이트하는 함수"""
        exec_price = self.get_est_exec_price(pos.side, market_price)
        new_size = pos.size + total_q
        new_entry_price = (pos.entry_price * pos.size + exec_price * total_q) / new_size
        pos.entry_price = new_entry_price
        pos.size = new_size

    def _get_current_mode(self, long_pos: Position, short_pos: Position) -> StrategyMode:
        """포지션 수량을 비교하여 현재 전략 모드를 반환"""
        if abs(long_pos.size - short_pos.size) < 1e-9:
            return StrategyMode.LOCKED
        return StrategyMode.IMBALANCED

    def execute_partial_close(self, pos: Position, q_to_close: float, market_price: float) -> float:
        """지정된 수량만큼 포지션을 부분 청산하고 실현 손익을 반환"""
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
            return "EXIT_PROFIT_TARGET_MET"
        if current_spread < self.config.SpreadExitThreshold and total_pnl > 0:
            return "EXIT_SPREAD_TARGET_MET"

        # 2. 행동 결정
        # 2a. [잠금 모드]에서의 행동
        if current_mode == StrategyMode.LOCKED:
            # 부분 익절 로직
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
            
            # 공격적 물타기 로직
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

        # 2b. [불균형 모드]에서의 행동
        elif current_mode == StrategyMode.IMBALANCED:
            if total_pnl <= 0:
                if balancing_attempts < self.config.MaxBalancingAttempts:
                    # [행동] 개선된 방어적 균형화
                    pos_to_avg, other_pos_entry, action_type = (long_pos, short_pos.entry_price, "DEFENSIVE_AVG_LONG") if long_pos.size < short_pos.size else (short_pos, long_pos.entry_price, "DEFENSIVE_AVG_SHORT")
                    
                    q = self.propose_qs(pos_to_avg)[0]
                    sim_res = self.simulate_averaging(pos_to_avg, q, self.get_est_exec_price(pos_to_avg.side, market_price), other_pos_entry)
                    
                    # *** 개선된 조건: 더 유리한 가격에서만 물타기 ***
                    is_favorable_price = (pos_to_avg.side == "LONG" and market_price < pos_to_avg.entry_price) or \
                                         (pos_to_avg.side == "SHORT" and market_price > pos_to_avg.entry_price)

                    if sim_res.dS < 0 and is_favorable_price and self.meets_financial_criteria(sim_res, acct):
                        self.execute_averaging(pos_to_avg, q, market_price)
                        return f"ACTION_{action_type}"
                    else:
                        return "NO_VALID_ACTION_DEFENSIVE_AVG"
                else:
                    # [행동] 전략적 손절 (Strategic Cut) -> 강제로 Locked 모드 복귀
                    self.log(f"  - 전략적 손절 발동 (균형화 시도 횟수: {balancing_attempts})")
                    pos_to_cut, other_pos, cut_side = (short_pos, long_pos, "SHORT") if plus_di_now > minus_di_now else (long_pos, short_pos, "LONG")
                    
                    q_to_close = pos_to_cut.size - other_pos.size
                    
                    if q_to_close > 1e-9:
                        self.execute_partial_close(pos_to_cut, q_to_close, market_price)
                        return f"ACTION_STRATEGIC_CUT_TO_LOCKED_{cut_side}"
                    else:
                        return "NO_ACTION_STRATEGIC_CUT_NO_QUANTITY"

        return "NO_ACTION"


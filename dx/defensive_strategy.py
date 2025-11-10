from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from dx.signal_generator import StrategyConfig # 통합된 StrategyConfig 임포트

# 전략 모드 정의
class StrategyMode(Enum):
    LOCKED = "LOCKED"  # 롱/숏 포지션 수량이 동일한 상태
    IMBALANCED = "IMBALANCED"  # 롱/숏 포지션 수량이 다른 상태

@dataclass
class Position:
    side: str
    entry_price: float
    size: float
    initial_size: float # 초기 진입 시점의 포지션 크기 (물타기 전)

@dataclass
class AccountState:
    u_loss: float # 미실현 손실
    margin_usage: float # 증거금 사용률

@dataclass
class SimResult:
    dS: float # 스프레드 변화량 (음수면 감소)
    margin_usage: float # 예상 증거금 사용률

class DynamicHedgeStrategy:
    def __init__(self, config: StrategyConfig, logger=None):
        self.config = config
        self.logger = logger if logger is not None else print

    def log(self, message):
        self.logger(message)

    def _get_current_mode(self, long_pos: Position, short_pos: Position) -> StrategyMode:
        if long_pos.size == short_pos.size:
            return StrategyMode.LOCKED
        return StrategyMode.IMBALANCED

    def get_est_exec_price(self, side: str, market_price: float) -> float:
        """슬리피지를 고려한 예상 체결 가격"""
        if side == "LONG":
            return market_price * (1 + self.config.SlipTolerance)
        else: # SHORT
            return market_price * (1 - self.config.SlipTolerance)

    def propose_qs(self, pos: Position) -> List[float]:
        """포지션에 추가할 수량 제안 (현재는 고정 비율)"""
        return [pos.size * self.config.AveragingSizeRatio]

    def simulate_averaging(self, pos_to_avg: Position, q: float, est_exec_price: float, other_pos_entry: float) -> SimResult:
        """물타기 시뮬레이션 결과 (스프레드 변화, 증거금 사용률)"""
        new_total_size = pos_to_avg.size + q
        new_entry_price = (pos_to_avg.entry_price * pos_to_avg.size + est_exec_price * q) / new_total_size
        
        # 스프레드 변화 시뮬레이션 (dS가 음수면 스프레드 감소)
        if pos_to_avg.side == "LONG":
            dS = (new_entry_price - other_pos_entry) - (pos_to_avg.entry_price - other_pos_entry)
        else: # SHORT
            dS = (other_pos_entry - new_entry_price) - (other_pos_entry - pos_to_avg.entry_price)
        
        # 증거금 사용률 시뮬레이션 (간단화)
        margin_usage = 0.5 # 실제 계산은 더 복잡하지만, 여기서는 임시 값
        return SimResult(dS=dS, margin_usage=margin_usage)

    def meets_financial_criteria(self, sim_res: SimResult, acct: AccountState) -> bool:
        """재무적 기준 충족 여부 (예: 증거금 사용률)"""
        # 예상 증거금 사용량이 90%를 넘지 않아야 함
        if sim_res.margin_usage > 0.9:
            return False
        return True

    def execute_averaging(self, pos: Position, q: float, market_price: float):
        """포지션에 물타기 실행"""
        est_exec_price = self.get_est_exec_price(pos.side, market_price)
        pos.entry_price = (pos.entry_price * pos.size + est_exec_price * q) / (pos.size + q)
        pos.size += q
        self.log(f"  - {pos.side} 포지션 물타기 실행: {q:.2f} 추가, 새 평균 진입가: {pos.entry_price:.2f}")

    def execute_partial_close(self, pos: Position, q: float, market_price: float):
        """포지션 부분 청산 실행"""
        pos.size -= q
        self.log(f"  - {pos.side} 포지션 부분 청산 실행: {q:.2f} 감소")

    def _try_averaging(self, long_pos, short_pos, market_price, plus_di_now, minus_di_now, acct: AccountState):
        """공격적 진입(Averaging) 시도"""
        pos_to_avg, other_pos, trend = (long_pos, short_pos, "LONG") if plus_di_now > minus_di_now else (short_pos, long_pos, "SHORT")
        
        # 물타기 시도 전에 포지션 크기가 0이 아닌지 확인
        if pos_to_avg.size <= 1e-9: # 너무 작은 포지션은 물타기 안함
            return "NO_ACTION"

        q = self.propose_qs(pos_to_avg)[0]
        sim_res = self.simulate_averaging(pos_to_avg, q, self.get_est_exec_price(pos_to_avg.side, market_price), other_pos.entry_price)
        
        if sim_res.dS < 0 and self.meets_financial_criteria(sim_res, acct):
            self.execute_averaging(pos_to_avg, q, market_price)
            return f"ACTION_AVG_{trend}_TREND"
        return "NO_ACTION"

    def _try_partial_close(self, long_pos, short_pos, market_price, plus_di_now, minus_di_now):
        """수비적 진입(Partial Close) 시도"""
        # 현재 추세에 반대되는 포지션을 부분 청산
        pos_to_close, trend = (short_pos, "LONG") if plus_di_now > minus_di_now else (long_pos, "SHORT")
        
        # 청산할 포지션 크기가 0이 아닌지 확인
        if pos_to_close.size <= 1e-9: # 너무 작은 포지션은 청산 안함
            return "NO_ACTION"

        q_to_close = pos_to_close.size * self.config.PartialCloseRatio
        if q_to_close > 1e-9:
            self.execute_partial_close(pos_to_close, q_to_close, market_price)
            return f"ACTION_PARTIAL_CLOSE_{trend}"
        return "NO_ACTION"

    def determine_next_action(self, long_pos: Position, short_pos: Position, acct: AccountState, market_price: float,
                              plus_di_now: float, minus_di_now: float, adx_now: float, balancing_attempts: int) -> str:
        """
        현재 상태를 진단하고, 다음 행동을 결정하여 문자열로 반환합니다.
        """
        total_pnl = ((market_price - long_pos.entry_price) * long_pos.size) + \
                    ((short_pos.entry_price - market_price) * short_pos.size)
        
        current_mode = self._get_current_mode(long_pos, short_pos)
        current_spread = abs(long_pos.entry_price - short_pos.entry_price)

        # 1. 종료 조건 (최우선)
        if current_mode == StrategyMode.IMBALANCED and total_pnl > 0:
            return "EXIT_PROFIT_TARGET_MET"
        
        # 스프레드 목표 달성 종료 조건
        if current_spread < self.config.SpreadExitThreshold:
            return "EXIT_SPREAD_TARGET_MET"

        # 2. 행동 결정
        # 2a. [잠금 모드]에서의 행동
        if current_mode == StrategyMode.LOCKED:
            # Config에 따라 공격 우선 또는 수비 우선 순서로 행동 결정
            if self.config.LockedModePriority == "ATTACK":
                # 1순위: 공격적 진입 (Averaging)
                action = self._try_averaging(long_pos, short_pos, market_price, plus_di_now, minus_di_now, acct)
                if action != "NO_ACTION":
                    return action
                # 2순위: 수비적 진입 (Partial Close)
                action = self._try_partial_close(long_pos, short_pos, market_price, plus_di_now, minus_di_now)
                if action != "NO_ACTION":
                    return action
            else: # "DEFENSE" 우선
                # 1순위: 수비적 진입 (Partial Close)
                action = self._try_partial_close(long_pos, short_pos, market_price, plus_di_now, minus_di_now)
                if action != "NO_ACTION":
                    return action
                # 2순위: 공격적 진입 (Averaging)
                action = self._try_averaging(long_pos, short_pos, market_price, plus_di_now, minus_di_now, acct)
                if action != "NO_ACTION":
                    return action
        
        # 2b. [불균형 모드]에서의 행동
        elif current_mode == StrategyMode.IMBALANCED:
            # 손실 상태일 때만 방어 로직 실행
            if total_pnl <= 0:
                # 균형화 시도 횟수가 최대 횟수 미만일 경우 -> 방어적 균형화 (Defensive Averaging)
                if balancing_attempts < self.config.MaxBalancingAttempts:
                    # 불리한 포지션(수량이 적은 쪽)에 물타기하여 균형 맞추기 시도
                    pos_to_avg, other_pos, action_type = (long_pos, short_pos, "DEFENSIVE_AVG_LONG") if long_pos.size < short_pos.size else (short_pos, long_pos, "DEFENSIVE_AVG_SHORT")
                    other_pos_entry = other_pos.entry_price
                    
                    # 여러 제안 사이즈 중 최적의 것을 선택 (현재는 첫번째 것만 사용)
                    q = self.propose_qs(pos_to_avg)[0]
                    sim_res = self.simulate_averaging(pos_to_avg, q, self.get_est_exec_price(pos_to_avg.side, market_price), other_pos_entry)
                    
                    # 물타기 시 스프레드 감소 및 재무 조건 충족 시에만 실행
                    if sim_res.dS < 0 and self.meets_financial_criteria(sim_res, acct):
                        self.execute_averaging(pos_to_avg, q, market_price)
                        return f"ACTION_{action_type}"
                    else:
                        return "NO_VALID_ACTION_DEFENSIVE_AVG"
                # 균형화 시도 횟수 소진 시 -> 전략적 손절 (Strategic Cut)
                else:
                    # 강제로 Locked 모드로 복귀시키기 위해 불리한 포지션을 필요한 만큼만 손절
                    self.log(f"  - 전략적 손절 발동 (균형화 시도 횟수: {balancing_attempts})")
                    pos_to_cut, other_pos, cut_side = (short_pos, long_pos, "SHORT") if plus_di_now > minus_di_now else (long_pos, short_pos, "LONG")
                    
                    q_to_close = pos_to_cut.size - other_pos.size
                    
                    if q_to_close > 1e-9:
                        self.execute_partial_close(pos_to_cut, q_to_close, market_price)
                        return f"ACTION_STRATEGIC_CUT_TO_LOCKED_{cut_side}"
                    else:
                        return "NO_ACTION_STRATEGIC_CUT_NO_QUANTITY"

        return "NO_ACTION"
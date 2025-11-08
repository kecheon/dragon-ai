from dataclasses import dataclass
from typing import List, Dict, Any

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
    atr_now: float
    atr_base: float

@dataclass
class SimulationResult:
    new_entry: float
    dS: float
    d_u_loss: float
    d_margin: float

# --- 최종 전략 클래스 ---
class DefensiveStrategy:
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

    def defensive_loop_step(self, long_pos: Position, short_pos: Position, acct: AccountState, market_price: float) -> str:
        """동적 최적 행동 선택 로직이 포함된 메인 루프 함수"""
        # 1. 종료 조건 우선 확인
        current_spread = abs(long_pos.entry_price - short_pos.entry_price)
        if current_spread < self.config.SpreadExitThreshold:
            self.log(f"EXIT: 스프레드 목표({self.config.SpreadExitThreshold:.4f}) 달성 (현재: {current_spread:.4f}). 포지션 동시 청산.")
            return "EXIT_SPREAD_TARGET_MET"

        total_pnl = ((market_price - long_pos.entry_price) * long_pos.size) + \
                    ((short_pos.entry_price - market_price) * short_pos.size)

        is_imbalanced = abs(long_pos.size - short_pos.size) > 1e-9
        if is_imbalanced and total_pnl > 0:
            self.log(f"EXIT: 불균형 상태에서 수익({total_pnl:.4f}) 전환. 포지션 동시 청산.")
            return "EXIT_PROFIT_TARGET_MET"

        if total_pnl >= 0:
            return "NO_ACTION_PROFITABLE"

        # 2. 최적 행동 탐색
        loss_pos, other_pos = (long_pos, short_pos) if ((market_price - long_pos.entry_price) * long_pos.size) < ((short_pos.entry_price - market_price) * short_pos.size) else (short_pos, long_pos)
        # self.log(f"방어 로직 발동. 손실측: {loss_pos.side}, 수익측: {other_pos.side}")

        valid_actions: List[Dict[str, Any]] = []

        # 시뮬레이션 1: 손실 포지션 물타기
        for q in self.propose_qs(loss_pos):
            sim_res = self.simulate_averaging(loss_pos, q, self.get_est_exec_price(loss_pos.side, market_price), other_pos.entry_price)
            if sim_res.dS < 0 and self.meets_financial_criteria(sim_res, acct):
                valid_actions.append({'type': 'AVG_LOSER', 'q': q, 'dS': sim_res.dS, 'pos_to_avg': loss_pos})
                # self.log(f"  - 유효 행동 발견: 손실측 물타기 (q={q:.4f}, 예상 dS={sim_res.dS:.4f})")

        # 시뮬레이션 2: 수익 포지션 불타기
        for q in self.propose_qs(other_pos):
            sim_res = self.simulate_averaging(other_pos, q, self.get_est_exec_price(other_pos.side, market_price), loss_pos.entry_price)
            if sim_res.dS < 0 and self.meets_financial_criteria(sim_res, acct):
                valid_actions.append({'type': 'AVG_WINNER', 'q': q, 'dS': sim_res.dS, 'pos_to_avg': other_pos})
                # self.log(f"  - 유효 행동 발견: 수익측 불타기 (q={q:.4f}, 예상 dS={sim_res.dS:.4f})")

        # 3. 행동 선택 및 실행
        if not valid_actions:
            # self.log("  - 유효한 스프레드 감소 행동을 찾지 못했습니다.")
            return "NO_VALID_ACTION"

        best_action = min(valid_actions, key=lambda x: x['dS'])
        # self.log(f"  -> 최적 행동 선택: {best_action['type']}")

        self.execute_averaging(best_action['pos_to_avg'], best_action['q'], market_price)
        return f"ACTION_{best_action['type']}"

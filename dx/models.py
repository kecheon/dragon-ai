from dataclasses import dataclass
from enum import Enum

# 여러 모듈에서 공통으로 사용하는 데이터 구조를 정의합니다.
# 이 클래스들을 별도의 파일로 분리하여 순환 참조(Circular Import) 문제를 해결합니다.

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
class StrategyConfig:
    """
    전략의 모든 파라미터를 통합 관리하는 데이터 클래스.
    신호 생성 조건, 포지션 관리, 리스크 관리 등 모든 설정이 포함됩니다.
    """
    # --- 신호 생성 파라미터 (Signal Generation Parameters) ---
    adx_threshold: float = 15.0
    """ADX(Average Directional Index) 임계값. ADX가 이 값보다 높으면 추세가 강하다고 판단합니다."""
    atr_window: int = 100
    """ATR(Average True Range)의 이동평균을 계산하는 윈도우 크기. 변동성 판단에 사용됩니다."""
    defensive_rsi_threshold: int = 30
    """방어적 물타기 시 RSI 과매도/과매수 판단 기준. (예: 30이면 RSI < 30일 때 과매도)"""

    # --- 포지션 관리 파라미터 (Position Management Parameters) ---
    AveragingSizeRatio: float = 1.5
    """'과감한 물타기' 시 기존 포지션 대비 추가할 수량의 비율. (예: 1.5는 150% 추가)"""
    PartialCloseRatio: float = 0.5
    """부분 익절(Partial Close) 시 청산할 수량의 비율. (예: 0.5는 50% 청산)"""
    LockedModePriority: str = "ATTACK"
    """잠금(Locked) 모드에서 행동 우선순위. "ATTACK"은 공격적 진입을, "DEFENSE"는 수비적 진입을 먼저 시도합니다."""
    MaxBalancingAttempts: int = 1
    """방어적 균형화를 시도할 수 있는 최대 횟수. 이 횟수를 초과하면 지능형 방향 전환을 고려합니다."""

    # --- 리스크 관리 파라미터 (Risk Management Parameters) ---
    CycleStopLossRatio: float = -0.10
    """단일 매매 사이클에서 허용되는 최대 손실률. 이 비율을 초과하면 사이클이 즉시 종료됩니다. (예: -0.10은 10% 손실)"""
    SpreadExitThreshold: float = 0.1
    """스프레드(롱/숏 진입 가격 차이)가 이 값보다 작아지면 포지션을 동시 청산하는 임계값. 방어적 탈출 조건으로 사용됩니다."""
    SlipTolerance: float = 0.0005
    """거래 체결 시 예상 가격과 실제 체결 가격 간의 허용 가능한 오차(슬리피지) 비율. (예: 0.0005는 0.05%)"""

    # --- 방향 전환 파라미터 (Intelligent Reversal Parameters) ---
    ReversalPartialCloseRatio: float = 0.5
    """'실수 인정' 시, 잘못 베팅했던 포지션을 청산하는 비율. (예: 0.5는 50% 청산)"""
    ReversalAveragingRatio: float = 1.0
    """'실수 인정' 후, 새로운 대세 방향의 포지션에 물타기하는 비율. (예: 1.0은 100% 추가)"""
    ReversalStopLossRatio: float = -0.03
    """'지능형 방향 전환' 직후에만 적용되는 특별 손절률. 일반 손절률보다 타이트하게 설정하여 휩소(Whipsaw) 손실을 제한합니다."""

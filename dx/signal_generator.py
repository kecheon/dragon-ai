from dataclasses import dataclass
import pandas as pd

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

    # --- 포지션 관리 파라미터 (Position Management Parameters) ---
    AveragingSizeRatio: float = 0.5
    """물타기(Averaging) 시 기존 포지션 대비 추가할 수량의 비율. (예: 0.5는 기존 수량의 50% 추가)"""
    PartialCloseRatio: float = 0.5
    """부분 익절(Partial Close) 시 청산할 수량의 비율. (예: 0.5는 기존 수량의 50% 청산)"""
    LockedModePriority: str = "ATTACK"
    """잠금(Locked) 모드에서 행동 우선순위. "ATTACK"은 공격적 진입(Averaging)을, "DEFENSE"는 수비적 진입(Partial Close)을 먼저 시도합니다."""
    MaxBalancingAttempts: int = 2
    """방어적 균형화(Defensive Averaging)를 시도할 수 있는 최대 횟수. 이 횟수를 초과하면 전략적 손절을 고려합니다."""

    # --- 리스크 관리 파라미터 (Risk Management Parameters) ---
    ForcedCutRatio: float = 0.25
    """강제 손절(Forced Cut) 시 청산할 포지션의 비율. (현재는 Strategic Cut으로 대체되어 사용되지 않을 수 있음)"""
    CycleStopLossRatio: float = -0.10
    """단일 매매 사이클에서 허용되는 최대 손실률. 이 비율을 초과하면 사이클이 즉시 종료됩니다. (예: -0.10은 10% 손실)"""
    SpreadExitThreshold: float = 0.1
    """스프레드(롱/숏 진입 가격 차이)가 이 값보다 작아지면 포지션을 동시 청산하는 임계값. 방어적 탈출 조건으로 사용됩니다."""
    SlipTolerance: float = 0.0005
    """거래 체결 시 예상 가격과 실제 체결 가격 간의 허용 가능한 오차(슬리피지) 비율. (예: 0.0005는 0.05% 슬리피지)"""

def generate_signals(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    주어진 데이터프레임에 전략 진입/청산 신호를 생성합니다.

    Args:
        data (pd.DataFrame): 'adx', 'plus_di', 'minus_di', 'atr' 컬럼을 포함하는 가격 데이터.
        config (StrategyConfig): 신호 생성에 필요한 파라미터 객체.

    Returns:
        pd.DataFrame: 'signal' 컬럼이 추가된 데이터프레임.
                      'signal'은 전략을 발동시킬지 여부를 나타내는 boolean 값입니다.
    """
    # ATR 이동평균 계산
    data['atr_sma'] = data['atr'].rolling(window=config.atr_window).mean()
    
    # 결측치 제거
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 신호 생성
    signals = []
    for i in range(len(data)):
        adx_now = data['adx'].iloc[i]
        plus_di_now = data['plus_di'].iloc[i]
        minus_di_now = data['minus_di'].iloc[i]
        
        # 추세 조건: ADX가 임계값 이상이고, +DI와 -DI가 교차한 상태
        is_trending = False
        if adx_now > config.adx_threshold:
            if (plus_di_now > minus_di_now) or (minus_di_now > plus_di_now):
                is_trending = True
        
        # 변동성 조건: 현재 ATR이 ATR의 이동평균보다 큰 상태
        is_volatile = data['atr'].iloc[i] > data['atr_sma'].iloc[i]
        
        trigger_activated = is_trending and is_volatile
        signals.append(trigger_activated)
        
    data['signal'] = signals
    
    return data

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

    # --- 타점 개선 필터 (Entry Point Filter) ---
    use_ema_filter: bool = True
    """EMA 정배열/역배열을 추세 방향 필터로 사용할지 여부."""
    ema_short_period: int = 50
    """단기 EMA 계산에 사용될 기간."""
    ema_long_period: int = 200
    """장기 EMA 계산에 사용될 기간."""
    use_rsi_filter: bool = True
    """RSI를 모멘텀 필터로 사용할지 여부."""
    rsi_momentum_threshold: int = 50
    """RSI 모멘텀 판단 기준. (예: 50 이상이면 상승 모멘텀)"""

    # --- 포지션 관리 파라미터 (Position Management Parameters) ---
    AveragingSizeRatio: float = 0.5
    """'과감한 물타기' 시 기존 포지션 대비 추가할 수량의 비율. (예: 1.0은 100% 추가)"""
    MaxPositionSize: float = 5.0
    """단일 포지션이 가질 수 있는 최대 크기. 기하급수적인 포지션 증가를 방지하여 리스크를 관리합니다."""
    PartialCloseRatio: float = 0.5
    """부분 익절(Partial Close) 시 청산할 수량의 비율. (예: 0.5는 기존 수량의 50% 청산)"""
    LockedModePriority: str = "ATTACK"
    """잠금(Locked) 모드에서 행동 우선순위. "ATTACK"은 공격적 진입(Averaging)을, "DEFENSE"는 수비적 진입(Partial Close)을 먼저 시도합니다."""
    MaxBalancingAttempts: int = 2
    """방어적 균형화(Defensive Averaging)를 시도할 수 있는 최대 횟수. 이 횟수를 초과하면 전략적 손절을 고려합니다."""

    # --- 리스크 관리 파라미터 (Risk Management Parameters) ---
    ForcedCutRatio: float = 0.5
    """강제 손절(Forced Cut) 시 청산할 포지션의 비율. (현재는 Strategic Cut으로 대체되어 사용되지 않을 수 있음)"""
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
    ReversalStopLossRatio: float = -0.05
    """'지능형 방향 전환' 직후에만 적용되는 특별 손절률. 일반 손절률보다 타이트하게 설정하여 휩소(Whipsaw) 손실을 제한합니다."""


def generate_initial_signals(
    data: pd.DataFrame, config: StrategyConfig
) -> pd.DataFrame:
    """
    백테스팅 시작 전에, 전체 데이터에 대해 기본적인 지표와 진입 신호를 미리 계산합니다.
    EMA, RSI 필터를 추가하여 타점의 신뢰도를 높입니다.
    """
    # ATR 이동평균, EMA, RSI 계산
    data["atr_sma"] = data["atr"].rolling(window=config.atr_window).mean()
    data[f"ema_short"] = data.ta.ema(length=config.ema_short_period)
    data[f"ema_long"] = data.ta.ema(length=config.ema_long_period)
    data.ta.rsi(append=True)
    data.dropna(inplace=True)

    # 기본적인 진입 트리거 신호 생성
    signals = []
    for i in range(len(data)):
        # 1. 기본 조건: 추세 강도 및 변동성
        is_trending = data["adx"].iloc[i] > config.adx_threshold
        is_volatile = data["atr"].iloc[i] > data["atr_sma"].iloc[i]

        signal = False
        if is_trending and is_volatile:
            # 2. 필터 조건: 추세 방향(EMA) 및 모멘텀(RSI)
            is_bullish_ema = not config.use_ema_filter or (
                data["ema_short"].iloc[i] > data["ema_long"].iloc[i]
            )
            is_bearish_ema = not config.use_ema_filter or (
                data["ema_short"].iloc[i] < data["ema_long"].iloc[i]
            )
            is_bullish_rsi = not config.use_rsi_filter or (
                data["RSI_14"].iloc[i] > 1.2 * config.rsi_momentum_threshold
            )
            is_bearish_rsi = not config.use_rsi_filter or (
                data["RSI_14"].iloc[i] < 0.8 * config.rsi_momentum_threshold
            )

            # 3. 최종 신호 결정
            # 상승 추세 조건이 모두 맞을 때
            if (
                data["plus_di"].iloc[i] > data["minus_di"].iloc[i]
                and is_bullish_ema
                and is_bullish_rsi
            ):
                signal = True
            # 하락 추세 조건이 모두 맞을 때
            elif (
                data["minus_di"].iloc[i] > data["plus_di"].iloc[i]
                and is_bearish_ema
                and is_bearish_rsi
            ):
                signal = True

        signals.append(signal)

    data["trigger"] = signals
    data.reset_index(inplace=True)
    data.rename(columns={"Timestamp": "timestamp"}, inplace=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data

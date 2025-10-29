# ===================================
# === 파라미터 설정 ===
# ===================================
WINDOW = 14
PARTIAL_CLOSE_RATIO = 0.5
EVALUATION_WINDOW = 24
FIXED_SPREAD = 0.02
LOOKBACK_PERIOD = 48
ADX_TREND_THRESHOLD = 20

# === 동적 레이블링 파라미터 ===
PROFIT_TAKE_PCT = 0.0046 # 익절 라인 (0.46%, 95th percentile)
STOP_LOSS_PCT = 0.0021 # 손절 라인 (0.21%, 75th percentile)
TRANSACTION_FEE_PCT = 0.0004 # 거래 수수료 (0.04%)

# ===================================
# === 액션 결정 함수 ===
# ===================================
def get_action_decision(params):
    """
    주어진 조건에 따라 부분 청산 액션을 결정합니다.
    params: adx, plus_di, minus_di, unrealized_long_pnl, unrealized_short_pnl, ADX_TREND_THRESHOLD
    """
    
    # 필요한 값들을 params 딕셔너리에서 추출
    adx = params['adx']
    plus_di = params['plus_di']
    minus_di = params['minus_di']
    unrealized_long_pnl = params['unrealized_long_pnl']
    unrealized_short_pnl = params['unrealized_short_pnl']
    adx_threshold = params['ADX_TREND_THRESHOLD']

    action = None

    # --- 현재 활성화된 전략 ---
    # 설명: ADX가 임계값 '이상'이고, DMI 교차가 발생할 때 (손익 무관)
    if plus_di < minus_di:
        action = "PARTIAL_CLOSE_LONG"
    elif plus_di > minus_di:
        action = "PARTIAL_CLOSE_SHORT"
    
    # --- 비활성화된 실험 전략들 (주석 처리) ---

    # 실험 2: ADX가 임계값 '미만'일 때 (손익 무관)
    # if adx < adx_threshold and plus_di < minus_di:
    #     action = "PARTIAL_CLOSE_LONG"
    # elif adx < adx_threshold and plus_di > minus_di:
    #     action = "PARTIAL_CLOSE_SHORT"

    # 실험 3: ADX가 임계값 '미만'이고 '수익 중'일 때
    # if unrealized_long_pnl > 0 and adx < adx_threshold and plus_di < minus_di:
    #     action = "PARTIAL_CLOSE_LONG"
    # elif unrealized_short_pnl > 0 and adx < adx_threshold and plus_di > minus_di:
    #     action = "PARTIAL_CLOSE_SHORT"

    return action

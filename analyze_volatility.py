import pandas as pd
import numpy as np

def analyze_price_movement(file_path):
    """
    OHLCV CSV 파일의 가격 움직임 통계를 분석합니다.
    """
    print(f"'{file_path}' 파일에서 데이터 분석을 시작합니다...")
    
    try:
        # 데이터 로드
        df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
        print(f"총 {len(df)}개의 캔들 데이터를 로드했습니다.")

        # 1. 캔들 내 진폭 (%): (고가 - 저가) / 시가
        intracandle_amplitude = ((df['High'] - df['Low']) / df['Open']) * 100
        
        # 2. 캔들 간 진폭 (%): 종가 기준 변화율
        intercandle_amplitude = df['Close'].pct_change().abs() * 100

        print("\n--- 가격 변동성 통계 (단위: %) ---\n")
        
        percentiles = [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
        
        print("1. 캔들 내 진폭 [(고가-저가)/시가] 분포:")
        print(intracandle_amplitude.describe(percentiles=percentiles).to_string())
        
        print("\n" + "="*40 + "\n")
        
        print("2. 캔들 간 종가 변화율 분포:")
        print(intercandle_amplitude.describe(percentiles=percentiles).to_string())
        
        print("\n분석 완료. 위 통계를 바탕으로 익절/손절 파라미터를 조정할 수 있습니다.")

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"분석 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # 분석할 데이터 파일 경로
    RAW_DATA_FILE = "BTCUSDT_5m_raw_data.csv"
    analyze_price_movement(RAW_DATA_FILE)

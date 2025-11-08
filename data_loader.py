import pandas as pd
import glob
import os

def load_price_data(symbol: str, timeframe: str = '5m', data_dir: str = '.') -> pd.DataFrame:
    """
    지정된 심볼과 타임프레임에 해당하는 모든 원본 가격 데이터 CSV 파일을 찾아 로드하고 병합합니다.
    파일 이름 패턴(예: 'BTCUSDT_5m_raw_data.csv.2022')을 기반으로 파일을 찾습니다.

    Args:
        symbol (str): 로드할 암호화폐 심볼 (예: 'BTCUSDT', 'ETHUSDT').
        timeframe (str): 데이터의 타임프레임 (예: '5m'). 기본값은 '5m'입니다.
        data_dir (str): 데이터 파일이 위치한 디렉토리. 기본값은 현재 디렉토리입니다.

    Returns:
        pd.DataFrame: 모든 데이터를 시간순으로 정렬하여 병합한 데이터프레임.
                      데이터를 찾지 못한 경우 빈 데이터프레임을 반환합니다.
    """
    file_pattern = os.path.join(data_dir, f"{symbol}*.csv")
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"경고: '{file_pattern}' 패턴에 해당하는 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    df_list = []
    for file in sorted(file_list):
        try:
            print(f"'{file}' 파일에서 데이터를 로드합니다...")
            df = pd.read_csv(file, index_col='Timestamp', parse_dates=True)
            df_list.append(df)
        except Exception as e:
            print(f"오류: '{file}' 파일 로드 중 에러 발생: {e}")
            continue

    if not df_list:
        print("경고: 유효한 데이터를 로드하지 못했습니다.")
        return pd.DataFrame()

    # 모든 데이터프레임을 하나로 합치고 인덱스(Timestamp) 기준으로 정렬
    combined_df = pd.concat(df_list)
    combined_df.sort_index(inplace=True)
    
    # 중복된 인덱스 처리 (있을 경우 첫 번째 값만 남김)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    print(f"총 {len(file_list)}개 파일에서 {len(combined_df)}개의 캔들 데이터를 로드했습니다.")
    return combined_df

if __name__ == '__main__':
    # 사용 예시
    print("--- BTCUSDT 데이터 로드 테스트 ---")
    btc_data = load_price_data('BTCUSDT')
    if not btc_data.empty:
        print(btc_data.head())
        print(btc_data.tail())

    print(" --- ETHUSDT 데이터 로드 테스트 ---")
    eth_data = load_price_data('ETHUSDT')
    if not eth_data.empty:
        print(eth_data.head())

    print(" --- SOLUSDT 데이터 로드 테스트 ---")
    # train.py에서 사용하던 이름이지만, 실제 파일 목록엔 없는 케이스 테스트
    sol_data = load_price_data('SOLUSDT')
    if sol_data.empty:
        print("SOLUSDT 데이터를 찾지 못했습니다.")

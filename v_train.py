import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def plot_feature_importance():
    """
    훈련된 XGBoost 모델을 로드하여 피처 중요도를 시각화하고 이미지 파일로 저장합니다.
    """
    print("--- 모델 피처 중요도 분석 시작 ---")
    
    # 한글 폰트 설정
    # 시스템에 맞는 한글 폰트 경로를 지정해야 할 수 있습니다.
    # 예: '/System/Library/Fonts/Supplemental/AppleGothic.ttf' (macOS)
    # 예: 'C:/Windows/Fonts/malgun.ttf' (Windows)
    try:
        # 사용 가능한 한글 폰트 중 하나를 선택
        font_path = None
        for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
            if 'AppleGothic' in font or 'Malgun' in font or 'NanumGothic' in font:
                font_path = font
                break
        
        if font_path:
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rc('font', family=font_name)
            print(f"한글 폰트 '{font_name}'을(를) 설정했습니다.")
        else:
            print("경고: 시스템에서 AppleGothic, Malgun Gothic 또는 NanumGothic 폰트를 찾을 수 없습니다. 차트의 한글이 깨질 수 있습니다.")
        
        # 마이너스 부호가 깨지는 문제 해결
        plt.rcParams['axes.unicode_minus'] = False

    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}")

    try:
        # 1. 모델 로드
        model = xgb.XGBClassifier()
        model.load_model("volatility_predictor_model.json")
        print("모델(volatility_predictor_model.json)을 성공적으로 로드했습니다.")

        # 2. 피처 이름 로드 (데이터 파일에서)
        df = pd.read_csv("volatility_data.csv")
        features = [
            'volatility', 'price_vs_ema_short', 'price_vs_ema_long',
            'ema_cross', 'z_score', 'adx', 'dmp', 'dmn'
        ]
        
        # 3. 피처 중요도 시각화
        fig, ax = plt.subplots(figsize=(12, 8))
        xgb.plot_importance(model, ax=ax, importance_type='gain', show_values=False)
        plt.title('XGBoost 피처 중요도 (Gain)')
        plt.xlabel('중요도 (Gain)')
        plt.ylabel('피처')
        
        # 4. 그래프 저장
        output_filename = "feature_importance.png"
        plt.savefig(output_filename)
        print(f"피처 중요도 그래프를 '{output_filename}' 파일로 저장했습니다.")
        
        # plt.show() # 주피터 노트북 등에서 바로 보려면 이 줄의 주석을 해제하세요.

    except FileNotFoundError as e:
        print(f"오류: 필요한 파일({e.filename})을 찾을 수 없습니다. 이전 단계를 완료했는지 확인하세요.")
    except Exception as e:
        print(f"피처 중요도 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    plot_feature_importance()

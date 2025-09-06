import pandas as pd
import matplotlib.pyplot as plt


def plot_sensor_from_csv(csv_path: str, tag_name: str):
    """
    csv_path: CSV 파일 경로 (예: 'data.csv')
    tag_name: 시각화하려는 센서 태그명 (예: 'LIT101')
    """
    # 1) CSV 파일 읽기
    #    - quotechar='"'를 지정해서, "124,3135"처럼 큰따옴표로 묶인 값 안의 쉼표를 하나의 필드로 읽도록 한다.
    df = pd.read_csv(csv_path, quotechar='"')

    # 2) Timestamp 열에 불필요한 앞뒤 공백이 있을 수 있으므로 먼저 strip()으로 제거
    df['Timestamp'] = df['Timestamp'].astype(str).str.strip()

    # 3) 날짜/시간으로 파싱
    #    원본 예시: "22/12/2015 4:30:00 PM"
    #    format: "%d/%m/%Y %I:%M:%S %p"
    #    - %I는 12시간제 시각(1~12)을 파싱하므로 "4"처럼 한 자리 시각도 문제없이 인식됨
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'],
        format='%d/%m/%Y %I:%M:%S %p',
        dayfirst=True
    )

    # 4) 센서 태그명 유효성 검사
    if tag_name not in df.columns:
        available = [col for col in df.columns if col not in ('Timestamp', 'Normal/Attack')]
        raise ValueError(f"'{tag_name}' 컬럼을 찾을 수 없습니다. 사용 가능한 태그: {available}")

    # 5) 인덱스에 Timestamp 설정
    df.set_index('Timestamp', inplace=True)

    # 6) 선택한 센서(tag_name) 시리즈 추출 및 float 변환
    #    - 먼저 astype(str)으로 문자열로 바꾼 뒤, 쉼표 제거 후 float로 변환
    sensor_series = (
        df[tag_name]
        .astype(str)  # int나 float인 경우에도 str 변경
        .str.replace(',', '')  # "124,3135" → "1243135"
        .astype(float)  # float로 변환
    )

    # 7) Matplotlib으로 시각화
    plt.figure(figsize=(12, 5))
    plt.plot(sensor_series.index, sensor_series.values, label=tag_name, linewidth=1)
    plt.xlabel('Timestamp')
    plt.ylabel(f'{tag_name} 값')
    plt.title(f'{tag_name} 시계열 데이터')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    csv_file = 'SWaT_Dataset_Normal_v1.csv'       # 예: 'data.csv'
    desired_tag = 'LIT101'
    plot_sensor_from_csv(csv_file, desired_tag)

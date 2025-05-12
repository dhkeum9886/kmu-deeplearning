"""
common_utils.py

■ 역할
  • CSV 파일에서 주어진 센서 태그(tag) 데이터를 읽어와
    - 타임스탬프 열 자동 탐지
    - 문자열 정리 및 datetime 변환
    - 중복 타임스탬프 제거
    - 1초 단위로 리샘플링(결측 시 선형 보간)
    - PyTorch Tensor로 변환
  • ARIMAModel 클래스:
    - 전통적 ARIMA 시계열 모델(p, d, q)을 PyTorch nn.Module로 구현
    - 학습 가능한 파라미터(자기회귀 계수, 이동평균 계수, 상수항)
    - forward 호출 시 잔차(residual) 계산 결과 반환
  • get_device():
    - CUDA 사용 가능 여부 확인
    - 사용 가능한 GPU 개수 반환

■ 상세 설명
  • load_csv:
    -
  • ARIMAModel:
    - 초기화 시 p, d, q 차수를 설정하고 파라미터를 nn.Parameter로 선언
    - forward
      -- d 차분 수행 (차분 차수)
      -- 과거 p 스텝 관측치로 AR 부분 계산
      -- 과거 q 스텝 잔차로 MA 부분 계산
      -- 예측값 = 상수항 + AR + MA, 잔차 = 실제 - 예측
      -- 앞 p 스텝은 초기값으로 제외한 나머지 잔차 반환
  • get_device:
    - GPU가 있으면 "cuda" 디바이스와 GPU 개수 반환,
    - 없으면 "cpu"와 0 반환
"""

import unicodedata
import pandas as pd
import torch
import torch.nn as nn

# 인덱스의 타임스탬프 포맷
DATE_FMT = "%d/%m/%Y %I:%M:%S %p"


def load_csv(csv_path: str, tag: str) -> tuple[torch.Tensor, pd.DatetimeIndex]:
    """
    CSV에서 지정한 센서(tag) 데이터를 읽어와 Tensor와 Timestamp 인덱스를 반환
    """
    # 구분자 자동 감지
    with open(csv_path, 'r', encoding='utf-8') as f:
        hdr = f.readline()
    sep = ';' if hdr.count(';') > hdr.count(',') else ','

    # 모든 열을 문자열로 읽어 Timestamp와 tag 칼럼 추출
    df = pd.read_csv(csv_path, sep=sep, dtype=str, encoding='utf-8')
    cols = df.columns.tolist()
    ts_col = next(c for c in cols if "timestamp" in unicodedata.normalize("NFKC", c).lower())
    if tag not in cols:
        raise ValueError(f"'{tag}' 열을 찾을 수 없습니다. 헤더: {cols}")
    df = df[[ts_col, tag]]

    # Timestamp 전처리, 공백 제거 > datetime 변환 > 인덱스로 설정
    df[ts_col] = df[ts_col].str.strip()
    df[ts_col] = pd.to_datetime(df[ts_col], format=DATE_FMT, dayfirst=True)
    df.set_index(ts_col, inplace=True)

    # 중복 타임스탬프 제거 (중복된 타임스탬프는 첫번째 값만 유지)
    df = df[~df.index.duplicated(keep='first')]

    # sensor 값 문자열 > float32
    df[tag] = df[tag].str.replace(",", ".", regex=False).astype("float32")

    # 결측치 처리, 1초 해상도 재샘플링 + 선형 보간
    series = df[tag].asfreq("1s").interpolate("linear")

    # 텐서 변환 및 인덱스 반환
    tensor = torch.tensor(series.values, dtype=torch.float32)
    return tensor, series.index


class ARIMAModel(nn.Module):
    """
    PyTorch 기반 ARIMA(p, d, q) 모델
    GPU를 활용하여 학습을 하기 위함.
    - p (AR 차수): 과거 관측치 몇 개를 볼지
    - d (차분 차수): 몇 번 차분해 정상 시계열로 만들지
    - q (MA 차수): 과거 오차 몇 개를 볼지
    """

    def __init__(self, p: int = 2, d: int = 1, q: int = 2):
        super().__init__()
        self.p, self.d, self.q = p, d, q
        # 학습 가능한 파라미터
        self.phi = nn.Parameter(torch.zeros(p))
        self.theta = nn.Parameter(torch.zeros(q))
        self.mu = nn.Parameter(torch.zeros(1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y가 1D인 경우 배치 차원 추가
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # 배치 크기(B)와 시퀀스 길이(Tseq)를 가져옴
        B, Tseq = y.shape

        # 모델의 차수 파라미터
        d, p, q = self.d, self.p, self.q

        # 실제 예측 가능한 시퀀스 길이
        T = Tseq - d

        # 잔차를 저장할 버퍼
        eps = y.new_zeros(B, T + q)

        # 차분된 시계열
        yd = y[:, d:] - y[:, :-d]

        # t 에 따라 AR+MA 계산 → 잔차 저장
        for t in range(p, T):
            # ar, 과거 p개 시점의 관측치에 대한 가중합
            ar = (self.phi * torch.flip(y[:, d + t - p:d + t], dims=[1])).sum(dim=1)

            # ma, 과거 q개 시점의 잔차에 대한 가중합
            ma = (self.theta * torch.flip(eps[:, t - q:t], dims=[1])).sum(dim=1)

            # 예측값 계산
            pred = self.mu + ar + ma

            # 잔차 계산
            eps[:, t] = yd[:, t] - pred
        # 초기 p 스텝 제외 후 반환
        return eps[:, p:]


def get_device() -> tuple[torch.device, int]:
    """
    CUDA 사용 가능 시 ('cuda', GPU 개수), 아니면 ('cpu', 0) 반환
    """
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        [print(f"[{idx}] {torch.cuda.get_device_name(idx)}") for idx in range(count)]
        return torch.device("cuda"), torch.cuda.device_count()
    else:
        return torch.device("cpu"), 0

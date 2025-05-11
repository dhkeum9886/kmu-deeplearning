## common\_utils.py

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
    1. 파일 헤더 첫 줄에서 구분자(',', ';')를 자동 감지
    2. pandas로 Timestamp 및 tag 열을 문자열로 일단 읽기
    3. Timestamp 문자열 불필요 공백 제거 후 고정 포맷으로 datetime 변환
    4. 중복된 타임스탬프 제거하고, 1초 단위로 재샘플링
    5. 결측값은 선형 보간(interpolate)으로 채움
    6. 최종 시계열 데이터를 torch.Tensor(float32)로 변환하여 반환
  • ARIMAModel:
    - 초기화 시 p, d, q 차수를 설정하고 파라미터를 nn.Parameter로 선언
    - forward 단계에서:
      a) d 차분 수행 (차분 차수)
      b) 과거 p 스텝 관측치로 AR 부분 계산
      c) 과거 q 스텝 잔차로 MA 부분 계산
      d) 예측값 = 상수항 + AR + MA, 잔차 = 실제 - 예측
      e) 앞 p 스텝은 초기값으로 제외한 나머지 잔차 반환
  • get_device:
    - GPU가 있으면 "cuda" 디바이스와 GPU 개수 반환,
    - 없으면 "cpu"와 0 반환
"""

import unicodedata
import pandas as pd
import torch
import torch.nn as nn

DATE_FMT = "%d/%m/%Y %I:%M:%S %p"


def load_csv(csv_path: str, tag: str) -> tuple[torch.Tensor, pd.DatetimeIndex]:
    """
    CSV에서 지정한 센서(tag) 데이터를 읽어와 Tensor와 Timestamp 인덱스를 반환
    """
    # 1) 구분자 자동 감지
    with open(csv_path, 'r', encoding='utf-8') as f:
        hdr = f.readline()
    sep = ';' if hdr.count(';') > hdr.count(',') else ','

    # 2) 모든 열을 문자열로 읽어 Timestamp와 tag 칼럼만 추출
    df = pd.read_csv(csv_path, sep=sep, dtype=str, encoding='utf-8')
    cols = df.columns.tolist()
    ts_col = next(c for c in cols if "timestamp" in unicodedata.normalize("NFKC", c).lower())
    if tag not in cols:
        raise ValueError(f"'{tag}' 열을 찾을 수 없습니다. 헤더: {cols}")
    df = df[[ts_col, tag]]

    # 3) Timestamp 전처리: 공백 제거 → datetime 변환 → 인덱스 설정
    df[ts_col] = df[ts_col].str.strip()
    df[ts_col] = pd.to_datetime(df[ts_col], format=DATE_FMT, dayfirst=True)
    df.set_index(ts_col, inplace=True)

    # 4) 중복 타임스탬프 제거 (첫 값 유지)
    df = df[~df.index.duplicated(keep='first')]

    # 5) sensor 값 문자열 → float32
    df[tag] = df[tag].str.replace(",", ".", regex=False).astype("float32")

    # 6) 1초 해상도 재샘플링 + 선형 보간
    series = df[tag].asfreq("1s").interpolate("linear")

    # 7) torch.Tensor 변환 및 인덱스 반환
    tensor = torch.tensor(series.values, dtype=torch.float32)
    return tensor, series.index


class ARIMAModel(nn.Module):
    """
    PyTorch 기반 ARIMA(p, d, q) 모델
    - p: 자기회귀 차수, d: 차분 차수, q: 이동평균 차수
    """

    def __init__(self, p: int = 2, d: int = 1, q: int = 2):
        super().__init__()
        self.p, self.d, self.q = p, d, q
        # 학습 가능한 파라미터
        self.phi = nn.Parameter(torch.zeros(p))
        self.theta = nn.Parameter(torch.zeros(q))
        self.mu = nn.Parameter(torch.zeros(1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # 1D 입력인 경우 배치 차원 추가
        if y.dim() == 1:
            y = y.unsqueeze(0)
        B, Tseq = y.shape
        d, p, q = self.d, self.p, self.q
        T = Tseq - d

        # 잔차 저장 버퍼
        eps = y.new_zeros(B, T + q)
        # 차분된 시계열
        yd = y[:, d:] - y[:, :-d]

        # t 에 따라 AR+MA 계산 → 잔차 저장
        for t in range(p, T):
            ar = (self.phi * torch.flip(y[:, d + t - p:d + t], dims=[1])).sum(dim=1)
            ma = (self.theta * torch.flip(eps[:, t - q:t], dims=[1])).sum(dim=1)
            pred = self.mu + ar + ma
            eps[:, t] = yd[:, t] - pred
        # 초기 p 스텝 제외 후 반환
        return eps[:, p:]


def get_device() -> tuple[torch.device, int]:
    """
    CUDA 사용 가능 시 ('cuda', GPU 개수), 아니면 ('cpu', 0) 반환
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.device_count()
    return torch.device("cpu"), 0

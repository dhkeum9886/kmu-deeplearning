"""
common_utils.py

공통 유틸리티 모듈
──────────────────────────────────────────────────────────────────────
1) load_with_timestamp(csv_path, tag)
   • CSV 파일에서 타임스탬프 열을 찾아
     1초 간격으로 재샘플링 후 결측 보간 → PyTorch Tensor로 변환
   • 반환: (Tensor 시계열, pandas.DatetimeIndex 타임스탬프)

2) ARIMAModel(nn.Module)
   • 전통적 시계열 모델 ARIMA(p, d, q)를
     PyTorch 모델 형태로 구현
   • forward(y) 호출 시 “잔차(residual)” 계산 결과 반환

3) get_device()
   • CUDA(GPU) 사용 가능 여부와 GPU 개수를 반환
     → 학습·추론 스크립트에서 CPU/GPU 자동 전환용
──────────────────────────────────────────────────────────────────────
"""

import unicodedata
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

# SWaT CSV 타임스탬프 형식 (12시간 AM/PM 표기)
DATE_FMT = "%d/%m/%Y %I:%M:%S %p"


def load_with_timestamp(csv_path: str, tag: str) -> tuple[torch.Tensor, pd.DatetimeIndex]:
    """
    CSV 파일을 읽어 1초 해상도 시계열 Tensor와
    해당 타임스탬프 인덱스를 반환한다.

    주요 흐름:
    1) 헤더만 읽어 'Timestamp' 열 이름 자동 탐색 (대소문자·공백 무시)
    2) 실제 데이터 읽기:
       - parse_dates: 문자열→날짜(datetime)
       - date_format: 포맷 고정으로 읽기 속도↑
       - decimal=",": "124,3135" → 124.3135
    3) 숫자 변환 실패 처리:
       - 문자열로 남아 있는 경우 콤마→점 치환 후 float32 타입으로 변환
    4) asfreq("1s"): 1초 단위로 시간 보간
       - interpolate("linear"): 선형 보간으로 결측 채우기
    """
    # 1) 헤더에서 타임스탬프 열명 찾기
    raw_cols = pd.read_csv(csv_path, nrows=0).columns
    ts_col = next(
        col for col in raw_cols
        if "timestamp" in unicodedata.normalize("NFKC", col).lower()
    )

    # 2) 실제 데이터 로드 (필요한 열만)
    df = pd.read_csv(
        csv_path,
        usecols=[ts_col, tag],
        parse_dates=[ts_col],
        date_format=DATE_FMT,
        index_col=ts_col,
        decimal=",",
    )

    # 3) 숫자형 아니면 콤마→점 치환 후 float32 변환
    if not pd.api.types.is_numeric_dtype(df[tag]):
        df[tag] = (
            df[tag].astype(str)
                  .str.replace(",", ".", regex=False)
                  .astype("float32")
        )

    # 4) 1초 해상도로 재샘플 + 결측 선형 보간
    series = df[tag].asfreq("1s").interpolate("linear")

    # 5) Tensor 변환 (float32) 및 타임스탬프 인덱스 반환
    tensor = torch.tensor(series.values, dtype=torch.float32)
    return tensor, series.index


class ARIMAModel(nn.Module):
    """
    ARIMA(p, d, q) 모델을 PyTorch nn.Module로 구현

    - p: AR(자기회귀) 차수 (과거 관측치 개수)
    - d: 차분 차수 (시계열 안정화용)
    - q: MA(이동평균) 차수 (과거 오차 개수)

    forward(y):
    - y 입력: 1D Tensor(T,) 또는 2D Tensor(B, T)
    - 내부적으로 배치 차원(B)을 맞춰 처리
    - 반환: 잔차(residual) Tensor, 크기 (B, T-p)
    """
    def __init__(self, p: int = 2, d: int = 1, q: int = 2):
        super().__init__()
        self.p, self.d, self.q = p, d, q

        # 학습할 파라미터: AR 계수(phi), MA 계수(theta), 상수항(mu)
        self.phi   = nn.Parameter(torch.zeros(p))
        self.theta = nn.Parameter(torch.zeros(q))
        self.mu    = nn.Parameter(torch.zeros(1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y가 1D면 배치 차원(B=1) 추가
        if y.dim() == 1:
            y = y.unsqueeze(0)  # (1, T)
        B, Tseq = y.shape
        d, p, q = self.d, self.p, self.q

        # 차분 이후 실제 사용 가능한 길이
        T = Tseq - d

        # 과거 오차(eps)를 보관할 버퍼
        eps = y.new_zeros(B, T + q)

        # 1차 차분된 시계열 (실제값 - 이전값)
        yd = y[:, d:] - y[:, :-d]  # (B, T)

        # t마다 AR+MA 부분 예측 → 잔차 계산
        for t in range(p, T):
            # AR 부분: 과거 p개 관측치의 가중합
            ar = (self.phi * torch.flip(y[:, d+t-p:d+t], dims=[1])).sum(dim=1)
            # MA 부분: 과거 q개 잔차의 가중합
            ma = (self.theta * torch.flip(eps[:, t-q:t], dims=[1])).sum(dim=1)
            # 예측값 = 상수항(mu) + AR + MA
            pred = self.mu + ar + ma
            # 잔차 = 실제값(차분) - 예측값
            eps[:, t] = yd[:, t] - pred

        # 앞 p개 스텝은 초기값이므로 제외하고 반환
        return eps[:, p:]


def get_device() -> tuple[torch.device, int]:
    """
    • torch.cuda.is_available()로 GPU 사용 가능 확인
    • 사용 가능하면 ('cuda', GPU 개수)
    • 아니면 ('cpu', 0)
    """
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU  개수:", torch.cuda.device_count())
        print("주 GPU :", torch.cuda.get_device_name(0))
        return torch.device("cuda"), torch.cuda.device_count()
    return torch.device("cpu"), 0

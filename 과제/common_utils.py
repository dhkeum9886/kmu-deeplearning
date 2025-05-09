# common_utils.py

"""
common_utils.py

공통 유틸리티 모듈
──────────────────────────────────────────────────────────────────────
1) load_with_timestamp(csv_path, tag)
   • CSV/TSV 파일 구분자(',', ';') 자동 감지
   • Timestamp 열 값의 앞뒤 공백 제거 → pd.to_datetime 정확히 파싱
   • 중복 타임스탬프 제거 → 1초 해상도(asfreq) → 선형 보간(interpolate)
   • 최종 Tensor와 DatetimeIndex 반환

2) ARIMAModel(nn.Module)
   • PyTorch 기반 ARIMA(p, d, q) 모델 구현

3) get_device()
   • CUDA 사용 가능 여부 및 GPU 개수 반환
──────────────────────────────────────────────────────────────────────
"""

import unicodedata
import pandas as pd
import torch
import torch.nn as nn

# SWaT CSV 타임스탬프 포맷 (예: '28/12/2015 10:00:00 AM')
DATE_FMT = "%d/%m/%Y %I:%M:%S %p"


def load_with_timestamp(csv_path: str, tag: str) -> tuple[torch.Tensor, pd.DatetimeIndex]:
    """
    1) 첫 줄을 읽어 sep(',', ';') 감지
    2) usecols=[ts_col, tag] 로 데이터 로드 (문자열 상태)
    3) ts_col 값 .str.strip() → 공백 제거
       → pd.to_datetime(..., format=DATE_FMT, dayfirst=True)
    4) 중복 인덱스 제거(keep='first')
    5) tag 열에서 콤마 소수점→점, float32 변환
    6) asfreq('1s') → 1초 해상도 리샘플 + interpolate('linear')
    7) Tensor 변환 및 DatetimeIndex 반환
    """
    # 1) sep 감지
    with open(csv_path, 'r', encoding='utf-8') as f:
        hdr = f.readline()
    sep = ';' if hdr.count(';') > hdr.count(',') else ','

    # 2) Timestamp + tag 열만 문자열로 읽기
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: True,  # we'll drop unwanted cols next
        sep=sep,
        dtype=str,
        encoding='utf-8',
    )
    # 자동으로 컬럼 필터링
    cols = list(df.columns)
    ts_col = next(c for c in cols
                  if "timestamp" in unicodedata.normalize("NFKC", c).lower())
    if tag not in cols:
        raise ValueError(f"'{tag}' 열을 찾을 수 없습니다. 파일의 헤더: {cols}")
    df = df[[ts_col, tag]]

    # 3) 공백 제거 후 datetime 파싱
    df[ts_col] = df[ts_col].str.strip()
    df[ts_col] = pd.to_datetime(df[ts_col],
                                format=DATE_FMT,
                                dayfirst=True,
                                errors='raise')
    df.set_index(ts_col, inplace=True)

    # 4) 중복 타임스탬프 제거
    df = df[~df.index.duplicated(keep='first')]

    # 5) tag 열 콤마 소수점→점 변환 & float32
    df[tag] = df[tag].str.replace(",", ".", regex=False).astype("float32")

    # 6) 1초 해상도 리샘플 + 선형 보간
    series = df[tag].asfreq("1s").interpolate("linear")

    # 7) Tensor 변환 + DatetimeIndex 반환
    tensor = torch.tensor(series.values, dtype=torch.float32)
    return tensor, series.index


class ARIMAModel(nn.Module):
    """
    PyTorch 기반 ARIMA(p,d,q) 모델

    forward(y):
      y: 1D (T,) 또는 2D (B,T) Tensor
      return: 잔차 Tensor (B, T-p)
    """
    def __init__(self, p: int = 2, d: int = 1, q: int = 2):
        super().__init__()
        self.p, self.d, self.q = p, d, q
        self.phi   = nn.Parameter(torch.zeros(p))
        self.theta = nn.Parameter(torch.zeros(q))
        self.mu    = nn.Parameter(torch.zeros(1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(0)
        B, Tseq = y.shape
        d, p, q = self.d, self.p, self.q
        T = Tseq - d

        eps = y.new_zeros(B, T + q)
        yd  = y[:, d:] - y[:, :-d]

        for t in range(p, T):
            ar = (self.phi * torch.flip(y[:, d+t-p:d+t], dims=[1])).sum(dim=1)
            ma = (self.theta * torch.flip(eps[:, t-q:t], dims=[1])).sum(dim=1)
            pred = self.mu + ar + ma
            eps[:, t] = yd[:, t] - pred
        return eps[:, p:]


def get_device() -> tuple[torch.device, int]:
    """
    CUDA 사용 가능 시 ('cuda', GPU 수), 아니면 ('cpu', 0) 반환
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.device_count()
    return torch.device("cpu"), 0

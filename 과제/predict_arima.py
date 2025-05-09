"""
predict_arima.py + 시각화

■ 역할
 1. 학습된 ARIMA 파라미터(.pt)를 로드
 2. 평가용 CSV에서 동일 전처리 후 시계열 Tensor 생성
 3. 1-step ahead 잔차(residual) 계산
 4. 잔차를 Z-score 표준화 → |Z|>3인 스텝을 이상치로 판단
 5. 이상치 개수, 비율, 타임스탬프 출력
 6. Matplotlib으로 잔차 시계열을 그린 뒤 이상치 포인트 강조
"""

import argparse
import sys

import torch
from scipy.stats import zscore
import matplotlib.pyplot as plt  # ★ 추가

from common_utils import load_with_timestamp, ARIMAModel, get_device

# ── 1) 커맨드라인 인자 ────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",   default="test.csv",
                    help="평가용 CSV 파일 경로")
parser.add_argument("--tag",   default="LIT101",
                    help="예측할 센서 태그명")
parser.add_argument("--model", default="models/arima_LIT101.pt",
                    help="학습된 파라미터(.pt) 경로")
args = parser.parse_args()

# ── 2) 디바이스 선택 ───────────────────────────────────────────
device, _ = get_device()
print(f"▶ 추론 디바이스: {device}")

# ── 3) 데이터 로드 → Tensor, 타임스탬프 인덱스 반환 ───────────
series, ts_index = load_with_timestamp(args.csv, args.tag)
series = series.to(device)   # shape (T,)

# ── 4) 모델 초기화 & 파라미터 로드 ───────────────────────────
model = ARIMAModel().to(device)
state = torch.load(args.model, map_location=device)
model.load_state_dict(state)
model.eval()

# ── 5) 잔차(residual) 계산 ─────────────────────────────────────
with torch.no_grad():
    residuals = model(series).squeeze(0)  # shape (res_len,)

res_len = residuals.numel()
if res_len == 0:
    print("⚠ 입력 시계열이 너무 짧아 잔차 계산 불가")
    sys.exit(1)

# ── 6) Z-score > 3로 이상치 판정 ───────────────────────────────
z_scores = zscore(residuals.cpu().numpy(), nan_policy="omit")
anoms    = abs(z_scores) > 3

# ── 7) 시계열 끝에서 res_len개만큼 슬라이스하여 정렬 ────────────
aligned_ts = ts_index[-res_len:]

# ── 8) 결과 출력 ───────────────────────────────────────────────
count = int(anoms.sum())
ratio = count / res_len * 100
print(f"전체 {res_len} 스텝 중 이상치 {count}개 ({ratio:.2f}%)")
print("앞 10개 이상치 타임스탬프:")
for t in aligned_ts[anoms][:10]:
    print(" •", t)

# ── 9) Matplotlib 시각화 ─────────────────────────────────────
#    • 잔차 시계열(Residual) 전체 : 파란선
#    • 이상치 포인트만          : 빨간색 점
plt.figure(figsize=(12, 4))
plt.plot(aligned_ts, residuals.cpu().numpy(), label="Residuals")
plt.scatter(
    aligned_ts[anoms],
    residuals.cpu().numpy()[anoms],
    color="red",
    label=f"Anomalies ({count})",
    zorder=5
)
plt.title(f"ARIMA Residuals & Anomalies for {args.tag}")
plt.xlabel("Timestamp")
plt.ylabel("Residual")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

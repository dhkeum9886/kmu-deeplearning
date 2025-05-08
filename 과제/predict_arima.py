"""
predict_arima.py

■ 역할
 1. 학습된 ARIMA 파라미터(.pt)를 로드
 2. 평가용 CSV(공격 포함)에서 동일 전처리 후 시계열 Tensor 생성
 3. 1-step ahead 잔차(residual) 계산
 4. 잔차를 Z-score 표준화 → |Z|>3인 스텝을 이상치로 판단
 5. 이상치 개수, 비율, 타임스탬프(앞10개) 출력

■ 사용 예시
    python predict_arima.py --csv SWaT_Attack.csv --tag LIT101 \
                             --model models/arima_LIT101.pt
"""

import argparse
import torch
from scipy.stats import zscore

from common_utils import load_with_timestamp, ARIMAModel, get_device

# ── 1) 커맨드라인 인자 설정 ─────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",   default="SWaT_Dataset_Attack_v1.csv",
                    help="평가용 CSV 파일 경로")
parser.add_argument("--tag",   default="LIT101",
                    help="예측할 센서 태그명")
parser.add_argument("--model", default="models/arima_LIT101.pt",
                    help="학습된 파라미터(.pt) 경로")
args = parser.parse_args()

# ── 2) 디바이스 확인 ────────────────────────────────────────────
device, _ = get_device()
print(f"▶ 추론 디바이스: {device}")

# ── 3) 데이터 로드 → Tensor, 타임스탬프 인덱스 반환 ─────────────
series, ts_index = load_with_timestamp(args.csv, args.tag)
series = series.to(device)  # (T,)

# ── 4) 모델 초기화 및 파라미터 로드 ───────────────────────────
model = ARIMAModel().to(device)
state = torch.load(args.model, map_location=device)
model.load_state_dict(state)
model.eval()  # 평가 모드: 드롭아웃 등 비활성

# ── 5) 1-step 잔차 계산 ────────────────────────────────────────
with torch.no_grad():  # 추론용: 그래디언트 계산 OFF
    # 모델 forward는 배치 차원이 없으므로 1D 입력 허용
    residuals = model(series).squeeze(0)  # (T-p,)

# ── 6) Z-score > 3로 이상치 판단 ───────────────────────────────
z = zscore(residuals.cpu().numpy(), nan_policy="omit")
anoms = abs(z) > 3

# ── 7) 결과 출력 ───────────────────────────────────────────────
total = len(z)
count = int(anoms.sum())
ratio = count / total * 100
print(f"전체 {total} 스텝 중 이상치 {count}개 ({ratio:.2f}%)")
print("앞 10개 이상치 타임스탬프:")
for t in ts_index[model.d + model.p :][anoms][:10]:
    print(" •", t)

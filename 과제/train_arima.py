"""
train_arima.py

■ 역할
 1. SWaT 정상 데이터 CSV로부터 시계열 Tensor 로드
 2. ARIMAModel(p=2,d=1,q=2)를 학습
 3. 학습된 파라미터를 models/arima_<TAG>.pt에 저장
 4. GPU 자동 사용 여부 및 학습 시간 로깅

■ 사용 예시
    python train_arima.py --csv SWaT_Normal.csv --tag LIT101 \
                          --epochs 300 --lr 0.01
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.parallel import DataParallel

from common_utils import load_with_timestamp, ARIMAModel, get_device

# ── 1) 커맨드라인 인자 설정 ─────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",    default="SWaT_Dataset_Normal_v1.csv",
                    help="정상 구간 CSV 파일 경로")
parser.add_argument("--tag",    default="LIT101",
                    help="학습할 센서 태그명")
parser.add_argument("--epochs", type=int, default=300,
                    help="전체 학습 반복 횟수(Epoch)")
parser.add_argument("--lr",     type=float, default=1e-2,
                    help="학습률(Learning Rate)")
args = parser.parse_args()

# ── 2) 디바이스 확인 ────────────────────────────────────────────
device, n_gpu = get_device()
print(f"▶ 학습 디바이스: {device}  (GPU 개수={n_gpu})")

# ── 3) 데이터 로드 → Tensor, 배치(Batch) 형태로 변환 ────────────────
# series: 1D Tensor(T,), _unused: 시계열 타임스탬프(index)
series, _ = load_with_timestamp(args.csv, args.tag)

# 1차원 → 2차원으로 변환: (1, T)
y = series.unsqueeze(0).to(device)

# GPU가 2개 이상일 때는 DataParallel을 위해 동일 데이터를 GPU 수만큼 복제
if n_gpu > 1:
    y = y.repeat(n_gpu, 1)  # (B=n_gpu, T)

# ── 4) 모델 생성 & 옵티마이저 ────────────────────────────────────
base_model = ARIMAModel().to(device)
model = DataParallel(base_model) if n_gpu > 1 else base_model

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ── 5) 학습 루프 + 시간 로깅 ────────────────────────────────────
start_wall = datetime.now()       # 실제 날짜·시간
start_perf = time.perf_counter()  # 고해상도 타이머
print("▶ 학습 시작:", start_wall.strftime("%Y-%m-%d %H:%M:%S"))

prev_perf = start_perf
for epoch in range(1, args.epochs + 1):
    optimizer.zero_grad()         # 이전 기울기 초기화
    loss = torch.mean(model(y) ** 2)  # MSE(평균 제곱 오차) 계산
    loss.backward()               # 역전파: loss가 작아지도록 기울기 계산
    optimizer.step()              # 파라미터 업데이트

    # 에폭별 소요시간(ms) 측정
    now_perf = time.perf_counter()
    delta_ms = (now_perf - prev_perf) * 1000
    prev_perf = now_perf

    # 첫 에폭, 50의 배수 에폭, 마지막 에폭에만 출력
    if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
        print(f"[{epoch:03d}/{args.epochs}] "
              f"loss={loss.item():.6f}  |  Δt={delta_ms:.1f} ms")

# 전체 학습 소요시간 출력
total_s = time.perf_counter() - start_perf
print(f"■ 학습 종료 ({total_s:.2f} s 소요)")

# ── 6) 학습된 파라미터 저장 ────────────────────────────────────
Path("models").mkdir(exist_ok=True)
torch.save(base_model.state_dict(), f"models/arima_{args.tag}.pt")
print(f"✔ 모델 저장 완료: models/arima_{args.tag}.pt")

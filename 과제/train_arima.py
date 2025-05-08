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
from tqdm import trange

from common_utils import load_with_timestamp, ARIMAModel, get_device

# ──────────────────────────────────────────────────────────────
# 1) 커맨드라인 인자 정의
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",    default="SWaT_Dataset_Normal_v1.csv",
                    help="정상 구간 CSV 파일 경로")
parser.add_argument("--tag",    default="LIT101",
                    help="학습할 센서 태그명")
parser.add_argument("--epochs", type=int, default=100,
                    help="전체 학습 반복 횟수(Epoch)")
parser.add_argument("--lr",     type=float, default=1e-2,
                    help="학습률(Learning Rate)")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────
# 2) 디바이스 확인
# ──────────────────────────────────────────────────────────────
device, n_gpu = get_device()
print(f"▶ 학습 디바이스: {device}  (GPU 개수={n_gpu})")

# ──────────────────────────────────────────────────────────────
# 3) 데이터 로드 → Tensor, 배치(Batch) 구성
# ──────────────────────────────────────────────────────────────
series, _ = load_with_timestamp(args.csv, args.tag)  # 1D Tensor (T,)
y = series.unsqueeze(0).to(device)                   # (1, T)
if n_gpu > 1:
    y = y.repeat(n_gpu, 1)                           # (B=n_gpu, T)

# ──────────────────────────────────────────────────────────────
# 4) 모델 생성 & 옵티마이저
# ──────────────────────────────────────────────────────────────
base_model = ARIMAModel().to(device)
model = DataParallel(base_model) if n_gpu > 1 else base_model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ──────────────────────────────────────────────────────────────
# 5) 학습 루프 + tqdm 진행률 표시 + 에폭별 로그
# ──────────────────────────────────────────────────────────────
start_wall = datetime.now()
start_perf = time.perf_counter()
print("▶ 학습 시작:", start_wall.strftime("%Y-%m-%d %H:%M:%S"))

prev_perf = start_perf
# trange 를 이용해 진행 바 표시
for epoch in trange(1, args.epochs + 1, desc="Epochs"):
    optimizer.zero_grad()
    loss = torch.mean(model(y) ** 2)   # MSE(평균 제곱 오차)
    loss.backward()
    optimizer.step()

    # 에폭별 소요시간(ms) 측정
    now_perf = time.perf_counter()
    delta_ms = (now_perf - prev_perf) * 1000
    prev_perf = now_perf

    # 매 에폭마다 로스와 Δt(ms) 출력
    print(f"[{epoch:03d}/{args.epochs}] "
          f"loss={loss.item():.6f}  |  Δt={delta_ms:.1f} ms")

end_perf = time.perf_counter()
total_s = end_perf - start_perf
print(f"■ 학습 종료 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
      f"(총 소요 {total_s:.2f} s)")

# ──────────────────────────────────────────────────────────────
# 6) 학습된 파라미터 저장
# ──────────────────────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
torch.save(base_model.state_dict(), f"models/arima_{args.tag}.pt")
print(f"✔ 모델 저장 완료: models/arima_{args.tag}.pt")

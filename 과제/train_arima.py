"""
train_arima.py

■ 역할
  1. 정상 데이터 CSV를 읽어 시계열 Tensor 생성
  2. ARIMAModel(p=2, d=1, q=2) 객체 생성 후 GPU/CPU로 이동
  3. DataParallel로 다중 GPU 사용 지원
  4. MSE 손실을 최소화하도록 파라미터 학습
  5. 학습 과정 시간 로깅(실제시각, 에폭별 소요)
  6. 최종 파라미터를 models/arima_<TAG>.pt에 저장

■ 사용법
    python train_arima.py \
      --csv train.csv \
      --tag LIT101 \
      --epochs 100 \
      --lr 1e-2
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.parallel import DataParallel
from tqdm import trange

from common_utils import load_csv, ARIMAModel, get_device

# 1) 인자 파싱
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="train.csv", help="정상 구간 CSV 경로")
parser.add_argument("--tag", default="LIT101", help="센서 태그명")
parser.add_argument("--epochs", type=int, default=100, help="전체 Epoch 수")
parser.add_argument("--lr", type=float, default=1e-2, help="학습률")
args = parser.parse_args()

# 2) 디바이스 설정
device, n_gpu = get_device()
print(f"▶ 학습 디바이스: {device}  (GPU 개수={n_gpu})")

# 3) 데이터 로드
series, _ = load_csv(args.csv, args.tag)  # 1D Tensor (T,)
y = series.unsqueeze(0).to(device)  # (1, T)
if n_gpu > 1:
    y = y.repeat(n_gpu, 1)  # (B=n_gpu, T)

# 4) 모델 생성 + 다중 GPU 지원
base_model = ARIMAModel().to(device)
model = DataParallel(base_model) if n_gpu > 1 else base_model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# 5) 학습 루프 + 시간 로깅
start_wall = datetime.now()
start_perf = time.perf_counter()
print(f"▶ 학습 시작: {start_wall.strftime('%Y-%m-%d %H:%M:%S')}")
prev_perf = start_perf
for epoch in trange(1, args.epochs + 1, desc="Epochs"):
    optimizer.zero_grad()
    loss = torch.mean((model(y)) ** 2)  # MSE
    loss.backward()
    optimizer.step()

    now_perf = time.perf_counter()
    delta_ms = (now_perf - prev_perf) * 1000
    prev_perf = now_perf

    # 매 에폭 로깅
    print(f"[{epoch:03d}/{args.epochs}] loss={loss.item():.6f} | Δt={delta_ms:.1f}ms")

# 6) 학습 종료
total_s = time.perf_counter() - start_perf
print(f"■ 학습 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (총 {total_s:.2f}s)")

# 7) 모델 저장
Path("models").mkdir(exist_ok=True)
torch.save(base_model.state_dict(), f"models/arima_{args.tag}.pt")
print(f"✔ 모델 저장 완료: models/arima_{args.tag}.pt")

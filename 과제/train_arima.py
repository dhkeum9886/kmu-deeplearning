"""
train_arima.py

■ 역할
  • 정상 데이터 CSV를 읽어 시계열 Tensor 생성
  • ARIMAModel(p=2, d=1, q=2) 객체 생성 후 GPU/CPU로 이동
  • DataParallel로 다중 GPU 사용 지원
  • MSE 손실을 최소화하도록 파라미터 학습
  • 학습 과정 시간 로깅(실제시각, 에폭별 소요)
  • 최종 파라미터를 models/arima_<TAG>.pt에 저장

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

import sys

from common_utils import load_csv, ARIMAModel, get_device


def train_loop(model, data, epochs, optimizer):
    """
    • model    :  ARIMA 모델
    • data     : 입력 텐서
    • epochs   : 학습 반복 횟수
    • optimizer: 옵티마이저
    """
    # 로깅을 위한 타이머, 데이트타임객체
    start_wall = datetime.now()
    start_perf = time.perf_counter()
    print(f"학습 시작: {start_wall:%Y-%m-%d %H:%M:%S}")
    prev_perf = start_perf

    for epoch in trange(1, epochs + 1, desc="Epochs"):
        # 기울기 초기화
        optimizer.zero_grad()
        # 순전파(Forward) + 손실 계산
        loss = torch.mean((model(data)) ** 2)
        # 역전파
        loss.backward()
        # 파라미터 업데이트
        optimizer.step()

        # 로깅을 위한 시간 계산
        now_perf = time.perf_counter()
        delta_ms = (now_perf - prev_perf) * 1000
        prev_perf = now_perf

        # 상태 로깅
        print(f"[{epoch:03d}/{epochs}] loss={loss.item():.6f} | Δt={delta_ms:.1f} ms")

    total_time = time.perf_counter() - start_perf
    print(f"학습 종료: {datetime.now():%Y-%m-%d %H:%M:%S} (총 {total_time:.2f}s)")
    return total_time


def main():
    # 1) 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="train.csv", help="정상 구간 CSV 경로")
    parser.add_argument("--tag", default="LIT101", help="센서 태그명")
    parser.add_argument("--epochs", type=int, default=100, help="전체 Epoch 수")
    parser.add_argument("--lr", type=float, default=1e-2, help="학습률")
    args = parser.parse_args()

    # 2) 디바이스 설정
    device, n_gpu = get_device()
    print(f"학습 디바이스: {device}  (GPU 개수={n_gpu})")

    # 3) 데이터 로드
    try:
        series, _ = load_csv(args.csv, args.tag)
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        sys.exit(1)
    else:
        # 성공 시에만
        y = series.unsqueeze(0).to(device)  # (1, T)
        if n_gpu > 1:
            y = y.repeat(n_gpu, 1)  # (B=n_gpu, T)

    # 4) 모델 생성
    base_model = ARIMAModel().to(device)
    model = DataParallel(base_model) if n_gpu > 1 else base_model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5) 학습 수행
    train_loop(model, y, args.epochs, optimizer)

    # 6) 모델 저장
    Path("models").mkdir(exist_ok=True)
    torch.save(base_model.state_dict(), f"models/arima_{args.tag}.pt")
    print(f"모델 저장 완료: models/arima_{args.tag}.pt")


if __name__ == "__main__":
    main()

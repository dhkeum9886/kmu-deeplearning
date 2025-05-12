"""
predict_arima.py

■ 역할
  • 학습된 ARIMA 파라미터(.pt)를 로드
  • 평가용 CSV에서 동일 전처리 후 시계열 Tensor 생성
  • 1단계앞 잔차 계산
  • 잔차를 Z-score 표준화 → |Z|>3인 포인트를 이상치로 분류
  • 이상치 개수 및 비율, 타임스탬프 출력
  • Matplotlib으로 잔차 시계열과 이상치 시각화

■ 사용법
    python predict_arima.py \
      --csv test.csv \
      --tag LIT101 \
      --model models/arima_LIT101.pt
"""
import argparse
import sys

import torch
from scipy.stats import zscore
import matplotlib.pyplot as plt

from common_utils import load_csv, ARIMAModel, get_device


def main():
    # 1) 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test.csv", help="평가용 CSV 파일 경로")
    parser.add_argument("--tag", default="LIT101", help="예측할 센서 태그명")
    parser.add_argument("--model", default="models/arima_LIT101.pt", help="학습된 파라미터(.pt) 경로")
    args = parser.parse_args()

    # 2) 디바이스 확인
    device, _ = get_device()
    print(f"추론 디바이스: {device}")

    # 3) 테스트 데이터 로드 및 전처리
    try:
        series, ts_index = load_csv(args.csv, args.tag)
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        sys.exit(1)
    else:
        # 성공 시에만
        series = series.to(device)

    # 4) 모델 초기화 및 가중치 로드
    model = ARIMAModel().to(device)

    # 저장한 pt 파일을 불러와, 모델에 매핑
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 5) 잔차 계산
    with torch.no_grad():
        # forward()가 호출됨, 1단계앞 예측 후 실제값과의 차이가 잔차
        residuals = model(series).squeeze(0)

    # 잔차의 총 길이, 너무 짧으면 계산 못함.
    res_len = residuals.numel()
    if res_len == 0:
        print("!! 입력 시계열이 너무 짧아 잔차 계산 불가 !!")
        sys.exit(1)

    # 6) Z-score로 이상치 판정
    # 잔차를 정규분포의 표준정규(z)값 표준화
    z_scores = zscore(residuals.cpu().numpy(), nan_policy="omit")

    # 3시그마로 이상치 판정
    anoms = abs(z_scores) > 3

    # 7) 잔차 길이만큼 타임스탬프 정렬
    aligned_ts = ts_index[-res_len:]

    # 8) 결과 출력
    count = int(anoms.sum())
    ratio = count / res_len * 100
    print(f"전체 {res_len} 스텝 중 이상치 {count}개 ({ratio:.2f}%)")
    print("이상치 타임스탬프 (10개만):")
    for t in aligned_ts[anoms][:10]:
        print(" * ", t)

    # 9) 시각화
    plt.figure(figsize=(12, 4))
    plt.plot(aligned_ts, residuals.cpu().numpy(), label="Residuals")
    plt.scatter(aligned_ts[anoms], residuals.cpu().numpy()[anoms],
                color="red", label=f"Anomalies ({count})", zorder=5)
    plt.title(f"Residuals & Anomalies for {args.tag}")
    plt.xlabel("Timestamp")
    plt.ylabel("Residual")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

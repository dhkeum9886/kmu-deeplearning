# train_arima.py
# CUDA 사용 가능하면 GPU, 아니면 CPU에서 ARIMA(p=2,d=1,q=2) 학습
# models/arima_<TAG>.pt 에 파라미터 저장

import argparse, unicodedata
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

DATE_FMT = "%d/%m/%Y %I:%M:%S %p"        # 22/12/2015 4:30:00 PM

# ─────────────────────────────────────────────
# 0. 데이터 로드
# ─────────────────────────────────────────────
def load_with_timestamp(csv_path: str, tag: str) -> torch.Tensor:
    raw_cols = pd.read_csv(csv_path, nrows=0).columns
    ts_col = next(c for c in raw_cols
                  if "timestamp" in unicodedata.normalize("NFKC", c).lower())
    df = pd.read_csv(
        csv_path, usecols=[ts_col, tag], parse_dates=[ts_col],
        date_format=DATE_FMT, index_col=ts_col, decimal=",",
    )
    if not pd.api.types.is_numeric_dtype(df[tag]):
        df[tag] = df[tag].astype(str).str.replace(",", ".", regex=False).astype("float32")

    return torch.tensor(
        df[tag].asfreq("1s").interpolate("linear").values,
        dtype=torch.float32,
    )

# ─────────────────────────────────────────────
# 1. ARIMA Module
# ─────────────────────────────────────────────
class ARIMAModel(nn.Module):
    def __init__(self, p=2, d=1, q=2):
        super().__init__()
        self.p, self.d, self.q = p, d, q
        self.phi   = nn.Parameter(torch.zeros(p))
        self.theta = nn.Parameter(torch.zeros(q))
        self.mu    = nn.Parameter(torch.zeros(1))

    def forward(self, y):               # y : (B, T)
        d, p, q = self.d, self.p, self.q
        T = y.size(1) - d
        eps   = y.new_zeros(y.size(0), T + q)
        preds = y.new_zeros(y.size(0), T)
        yd = y[:, d:] - y[:, :-d]

        for t in range(p, T):
            ar = (self.phi * torch.flip(y[:, d+t-p:d+t], dims=[1])).sum(-1)
            ma = (self.theta * torch.flip(eps[:, t-q:t], dims=[1])).sum(-1)
            preds[:, t] = self.mu + ar + ma
            eps[:, t]   = yd[:, t] - preds[:, t]
        return eps[:, p:]               # (B, T-p)

# ─────────────────────────────────────────────
# 2. CLI 인자
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="SWaT_Dataset_Normal_v1.csv")
parser.add_argument("--tag", default="LIT101")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-2)
args = parser.parse_args()

# ─────────────────────────────────────────────
# 3. 디바이스 설정
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu  = torch.cuda.device_count() if device.type == "cuda" else 0
print(f"Running on {device}  (GPUs detected: {n_gpu})")

# ─────────────────────────────────────────────
# 4. 데이터 → 배치 구성
# ─────────────────────────────────────────────
series = load_with_timestamp(args.csv, args.tag)
y = series.unsqueeze(0).to(device)           # (1, T)
if n_gpu > 1:                                # DataParallel 입력 배치 복제
    y = y.repeat(n_gpu, 1)                   # (B=n_gpu, T)

# ─────────────────────────────────────────────
# 5. 모델 & 옵티마이저
# ─────────────────────────────────────────────
base_model = ARIMAModel().to(device)
model = DataParallel(base_model) if n_gpu > 1 else base_model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ─────────────────────────────────────────────
# 6. 학습
# ─────────────────────────────────────────────
for epoch in range(1, args.epochs + 1):
    optimizer.zero_grad()
    loss = torch.mean(model(y) ** 2)
    loss.backward()
    optimizer.step()
    if epoch == 1 or epoch % 50 == 0:
        print(f"[{epoch:03d}/{args.epochs}] MSE={loss.item():.6f}")

# ─────────────────────────────────────────────
# 7. 저장 (DataParallel → .module)
# ─────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
torch.save(base_model.state_dict(), f"models/arima_{args.tag}.pt")
print(f"✅  models/arima_{args.tag}.pt saved.")

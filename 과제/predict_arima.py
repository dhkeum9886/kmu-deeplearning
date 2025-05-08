# predict_arima.py
# CUDA 가능 시 GPU, 아니면 CPU에서 추론 + 이상 구간(|Z|>3) 표시

import argparse, unicodedata
import pandas as pd
import torch
from scipy.stats import zscore

DATE_FMT = "%d/%m/%Y %I:%M:%S %p"

def load_with_timestamp(csv_path: str, tag: str) -> torch.Tensor:
    raw_cols = pd.read_csv(csv_path, nrows=0).columns
    ts_col = next(c for c in raw_cols if "timestamp" in unicodedata.normalize("NFKC", c).lower())
    df = pd.read_csv(csv_path, usecols=[ts_col, tag], parse_dates=[ts_col],
                     date_format=DATE_FMT, index_col=ts_col, decimal=",")
    if not pd.api.types.is_numeric_dtype(df[tag]):
        df[tag] = df[tag].astype(str).str.replace(",", ".", regex=False).astype("float32")
    return torch.tensor(df[tag].asfreq("1s").interpolate("linear").values, dtype=torch.float32), df.index

class ARIMAModel(torch.nn.Module):
    def __init__(self, p=2, d=1, q=2):
        super().__init__()
        self.p, self.d, self.q = p, d, q
        self.phi   = torch.nn.Parameter(torch.zeros(p))
        self.theta = torch.nn.Parameter(torch.zeros(q))
        self.mu    = torch.nn.Parameter(torch.zeros(1))

    def forward(self, y):                # y : (T,)
        d, p, q = self.d, self.p, self.q
        T = y.size(0) - d
        eps = y.new_zeros(T + q)
        yd  = y[d:] - y[:-d]
        for t in range(p, T):
            ar = (self.phi * torch.flip(y[d+t-p:d+t], dims=[0])).sum()
            ma = (self.theta * torch.flip(eps[t-q:t], dims=[0])).sum()
            pred = self.mu + ar + ma
            eps[t] = yd[t] - pred
        return eps[p:]

# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="SWaT_Dataset_Attack_v1.csv")
parser.add_argument("--tag", default="LIT101")
parser.add_argument("--model", default="models/arima_LIT101.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference device: {device}")

series_tensor, ts_index = load_with_timestamp(args.csv, args.tag)
series_tensor = series_tensor.to(device)

model = ARIMAModel().to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()

with torch.no_grad():
    residuals = model(series_tensor)

z = zscore(residuals.cpu().numpy(), nan_policy="omit")
anoms = (abs(z) > 3)

print(f"스텝 {len(z)}개 중 이상치 {anoms.sum()}개 ({anoms.mean()*100:.2f}%)")
print("앞 10개 타임스탬프:", ts_index[model.d + model.p:][anoms][:10].to_list())

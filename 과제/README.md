SWaT_Dataset_Normal_v1.csv
https://drive.google.com/file/d/1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw/view

SWaT_Dataset_Attack_v0.csv
https://drive.google.com/file/d/1iDYc0OEmidN712fquOBRFjln90SbpaE7/view

torch_arima: PyTorch용 ARIMA 예시 구현
https://github.com/BenZickel/torch_arima?utm_source=chatgpt.com

python train_arima.py --csv anomalous_swat.csv --tag LIT101 --epochs 100 --lr 1e-2


python predict_arima.py --csv anomalous_swat.csv --tag LIT101 --model models/arima_LIT101.pt

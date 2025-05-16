import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import matplotlib.pyplot as plt

# from google.colab import files
# uploaded = files.upload()

df=pd.read_csv('iris.csv')
df.info()
print(df)

data=pd.get_dummies(df, columns=['Species'])
print(data)


x, y = train_test_split(data, test_size=0.2, random_state=123)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit(data).transform(x)
y_scaled = scaler.fit(y).transform(y)

x=torch.from_numpy(x_scaled).float()
y=torch.from_numpy(y_scaled).float()

print(x.shape)
print(y.shape)

num_clusters = 3
cluster_idx_x, cluster_centers = kmeans(X=x, num_clusters=num_clusters, distance='euclidean', device=device)

cluster_idx_y = kmeans_predict(y, cluster_centers, 'euclidean', device=device)
print(cluster_idx_y)

plt.figure(figsize=(4, 3), dpi=160)
plt.scatter(y[:, 0], y[:, 1], c=cluster_idx_y, cmap='viridis', marker='x')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='white', alpha=0.6, edgecolors='black', linewidths=2)
plt.tight_layout()
plt.show()
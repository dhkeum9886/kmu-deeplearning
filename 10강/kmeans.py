import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import matplotlib.pyplot as plt

df=pd.read_csv('./iris.csv')
df.info()
print('--------------------------')
print(df)

data = pd.get_dummies(df, columns='Species')
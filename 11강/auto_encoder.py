import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from cuda import get_device
from torch.nn.parallel import DataParallel

device, n_gpu = get_device()
print(f"학습 디바이스: {device}  (GPU 개수={n_gpu})")

dataset = torchvision.datasets.MNIST('./data/', download=True, train=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


base_model = Autoencoder().to(device)
model = DataParallel(base_model) if n_gpu > 1 else base_model


def normalize_output(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img


def check_plot():
    with torch.no_grad():
        for data in trainloader:
            inputs = data[0].to(device)
            outputs = model(inputs.view(-1, 28 * 28))
            outputs = outputs.view(-1, 1, 28, 28)

            # 원래 이미지
            input_samples = inputs.permute(0, 2, 3, 1).cpu().numpy()

            # 생성 이미지
            reconstructed_samples = outputs.permute(0, 2, 3, 1).cpu().numpy()
            break
    columns = 10  # 시각화 전체 너비
    rows = 5  # 시각화 전체 높이
    fig = plt.figure(figsize=(columns, rows))  # figure 선언
    for i in range(1, columns * rows + 1):
        img = input_samples[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img.squeeze())
        plt.axis('off')
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(columns, rows))
    for i in range(1, columns * rows + 1):
        img = reconstructed_samples[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img.squeeze())
        plt.axis('off')
    plt.show()


criterion = nn.MSELoss()  # MSE 사용
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(51):
    running_loss = 0.0
    for data in trainloader:
        inputs = data[0].to(device)
        optimizer.zero_grad()
        outputs = model(inputs.view(-1, 28 * 28))
        outputs = outputs.view(-1, 1, 28, 28)
        loss = criterion(inputs, outputs)
        # 라벨 대신 입력 이미지와 출력 이미지를 비교
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    cost = running_loss / len(trainloader)
    print('[%d] loss: %.3f' % (epoch + 1, cost))

check_plot()
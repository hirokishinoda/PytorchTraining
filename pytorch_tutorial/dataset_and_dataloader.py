from turtle import window_width
from pytest import skip
import torch
import torchvision
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset

class WineDataset(Dataset):
    
    def __init__(self):
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

# training
epochs = 2
total_samples = (len(dataset))
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, etc...
        print(f'epoch : {epoch+1}/{epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

# already datasets
#torchvision.datasets.MNIST()
#fashion-mnist, etc...
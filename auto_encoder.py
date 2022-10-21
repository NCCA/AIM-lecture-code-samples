
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import os


# models
class AutoEncoder(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(28 * 28, 64),
			nn.ReLU(),
			nn.Linear(64, 3))
		self.decoder = nn.Sequential(
			nn.Linear(3, 64),
			nn.ReLU(),
			nn.Linear(64, 28 * 28))
	def forward(self, x):
		return self.decoder(self.decoder(x))


# download on rank 0 only
mnist_train = MNIST(os.getcwd(), train=True, download=True)

# download on rank 0 only
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)

# train (55,000 images), val split (5,000 images)
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

# The dataloaders handle shuffling, batching, etc...
mnist_train = DataLoader(mnist_train, batch_size=64)
mnist_val = DataLoader(mnist_val, batch_size=64)

# init model
model = AutoEncoder()
model.cuda(0)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# TRAIN LOOP
model.train()
num_epochs = 1
for epoch in range(num_epochs):
    for train_batch in mnist_train:
        x, y = train_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        print('train loss: ', loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# EVAL LOOP
model.eval()
with torch.no_grad():
    val_loss = []
    for val_batch in mnist_val:
        x, y = val_batch
        x = x.cuda(0)
        x = x.view(x.size(0), -1)
        x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        val_loss.append(loss)
        val_loss = torch.mean(torch.tensor(val_loss))
        model.train()
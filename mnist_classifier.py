import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pathlib
import numpy as np


from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import wandb

class SimpleClassifier(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = F.dropout(F.relu(self.l1(x)))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


def train_step(model, train_loader, optimizer, epoch, log_interval=10):
    model.train()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # Zero gradient
        optimizer.zero_grad()
        
        # forward pass
        y_hat = model(x)

        # Compute loss
        loss = F.cross_entropy(y_hat, y)

        # backward step
        loss.backward()
        optimizer.step()

        # Logging
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} \tLoss: {loss.item()}')
            wandb.log({"loss": loss})



def test_step(model, test_loader, note="Testing", log=False):
    model.eval()

    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            pred = y_hat.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

        print(f'\n{note}: test_acc: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)\n')

        if log:
            wandb.log({"test_acc": 100. * correct / len(test_loader.dataset)})


def checkpoint(model, optimizer, epoch, save_path):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        save_path,
        )

def load_sates(model, optimizer, path):
    print(f'Loading model from checkpoint: {path}')

    states = torch.load(path)
    print(f"model was trained for {states['epoch']} epochs")
    model.load_state_dict(states['model_state_dict'])
    optimizer.load_state_dict(states['optimizer_state_dict'])

def main():
    # ------------
    # hparms
    # ------------
    epochs = 5
    batch_size = 64
    learning_rate = 1e-3
    hidden_dim = 64
    checkpoint_path = pathlib.Path('/workspace/AIM-lecture-code-samples/checkpoints/SimpleClassifier.pt')
    load_checkpoint = False
        

    # ------------
    # data
    # ------------
    mnist_train = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(mnist_train, batch_size=batch_size)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)

    # ------------
    # model
    # ------------
    model = SimpleClassifier(hidden_dim).to(device)

    # ------------
    # optimizer
    # ------------
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # ------------
    # Load states
    # ------------
    if load_checkpoint and checkpoint_path.is_file():
        load_sates(model, optimizer, checkpoint_path)

    # ------------
    # Logger
    # ------------
    config = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "learning_rate" : learning_rate,
        "hidden_dim" : hidden_dim,
    }
    wandb.init(project="mnist_classifier", config=config)

    # ------------
    # training
    # ------------
    test_step(model, test_loader, "Model accuracy before training")

    for epoch in range(epochs):
        train_step(model, train_loader, optimizer, epoch, 1000)
        checkpoint(model, optimizer, epoch, checkpoint_path)
        
        if lr_scheduler:
            wandb.log({"learning_rate": lr_scheduler.get_last_lr()[0]})
            lr_scheduler.step()

    # ------------
    # test
    # ------------
    test_step(model, test_loader, log=True)

if __name__ == '__main__':
    main()
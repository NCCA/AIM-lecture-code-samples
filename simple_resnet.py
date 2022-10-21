import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    '''
    A simple implemetation of the ResNet
    '''
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Conv2d(3,512,3, padding='same')

        self.conv2d1 = nn.Conv2d(512,512,3, padding='same')
        self.conv2d2 = nn.Conv2d(512,512,3, padding='same')
        self.conv2d3 = nn.Conv2d(512,512,3, padding='same')
        self.conv2d4 = nn.Conv2d(512,512,3, padding='same')

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

        self.out = nn.Conv2d(512,3,3, padding='same')
        
    def forward(self, x):
        x = F.relu(self.project(x))

        # depth = 1
        residual = x
        x = F.relu(self.bn1(self.conv2d1(x)))
        x = self.bn2(self.conv2d2(x))
        x = x+residual

        # depth = 2
        residual = x
        x = F.relu(self.bn3(self.conv2d3(x)))
        x = self.bn4(self.conv2d4(x))
        x = x+residual

        return F.relu(self.out(x))

if __name__ == '__main__':
    model = ResNet()
    y_hat = model(torch.rand((1,3,256,256)))
    print("y_hat: ", y_hat.shape)
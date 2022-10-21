import torch
import torch.nn as nn

class MyModule(nn.Module):
    '''
    Class_Discription
    '''

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,20,5),
        self.relu1 = nn.ReLU(),
        self.conv2 = nn.Conv2d(20,64,5),
        self.rely2 = nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.rely2(self.conv2(x))
        return x
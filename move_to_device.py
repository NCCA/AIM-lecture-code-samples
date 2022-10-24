import torch
import torch.nn as nn

# Default device is set to CPU
m_tensor = torch.rand((2,2))
print(m_tensor.get_device(), m_tensor)

# Move to GPU method 1
m_tensor = m_tensor.cuda()
print(m_tensor.get_device(), m_tensor)

# Move back to CPU method 2
m_tensor = m_tensor.cpu()
print(m_tensor.get_device())

# Move to GPU method 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m_tensor = m_tensor.to(device)
print(m_tensor.get_device(), m_tensor)

# Init tensor on device
m_tensor = torch.rand((2,2), device=device)
print(m_tensor.get_device(), m_tensor)

# define model and move to device
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

model = MyModule().to(device)
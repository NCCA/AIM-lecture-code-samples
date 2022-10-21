import torch
import torch.nn as nn
import collections
    
model1 = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

model2 = nn.Sequential(collections.OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

if __name__ == '__main__':
    print(model1)
    '''
    Sequential(
        (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
        (1): ReLU()
        (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
        (3): ReLU()
    )
    '''
    print(model2)
    '''
    Sequential(
        (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
        (relu1): ReLU()
        (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
        (relu2): ReLU()
    )
    '''

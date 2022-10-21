import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
    
class MyData(Dataset):
    '''
    Pytorch implementation of torch.utils.data.Dataset
    '''
    def __init__(self, filepath:str):
        self.df = pd.read_csv(filepath, header=None)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df.iloc[idx]

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

def main():
    dataset = MyData('path/to/my/dataset.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MyModule()

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()


    for epoch in range(2):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    main()
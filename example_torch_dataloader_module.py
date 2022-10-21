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

if __name__ == '__main__':
    dataset = MyData('path/to/my/dataset.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
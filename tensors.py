import torch

x = torch.rand((2,2)) # Genorate a random tensor
# >>> tensor([[0.9381, 0.7326],
#             [0.3474, 0.2743]]) 
# >>> torch.Size([2, 2])


torch.rand([1]).shape
# >>> tensor([0.4140]) 
# >>> torch.Size([1])

torch.rand([1,2]).shape
# >>> torch.Size([1,1])
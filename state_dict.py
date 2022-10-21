import torch
import torch.nn as nn
import torch.nn.functional as F

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()



# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# Model's state_dict:#
# >>> conv1.weight 	 torch.Size([6, 3, 5, 5])
# >>> conv1.bias 	 torch.Size([6])
# >>> conv2.weight 	 torch.Size([16, 6, 5, 5])
# >>> conv2.bias 	 torch.Size([16])
# >>> fc1.weight 	 torch.Size([120, 400])
# >>> fc1.bias 	 torch.Size([120])
# >>> fc2.weight 	 torch.Size([84, 120])
# >>> fc2.bias 	 torch.Size([84])
# >>> fc3.weight 	 torch.Size([10, 84])
# >>> fc3.bias 	 torch.Size([10])

# Initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# Optimizer's state_dict:
# >>> state 	 {}
# >>> param_groups 	 [{ 'lr': 0.001, 
# >>>                   'momentum': 0.9, 
# >>>                   'dampening': 0, 
# >>>                   'weight_decay': 0, 
# >>>                   'nesterov': False, 
# >>>                   'maximize': False, '
# >>>                   params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]
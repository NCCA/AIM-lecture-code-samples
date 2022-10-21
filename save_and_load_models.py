import torch
import torch.nn as nn

model = nn.Linear(256, 256)

# State_dict
model_save_path = "path/to/save/mode.pt"
torch.save(model.state_dict(), model_save_path)

# Entire model
model_save_path = "path/to/save/mode.pt"
torch.save(model, model_save_path)
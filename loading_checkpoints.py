import torch
import torch.nn as nn

model = nn.Linear(256, 256)

# State_dict
model_save_path = "path/to/save/mode.pt"
model.load_state_dict(torch.load(model_save_path))

# Entire model
model_save_path = "path/to/save/mode.pt"
model = torch.load(model_save_path)
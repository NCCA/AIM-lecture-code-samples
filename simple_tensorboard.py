import torch
from torch.utils.tensorboard import SummaryWriter

# Create dummy data
x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -10 * x + 0.1 * torch.randn(x.size())

# Create dummy Model
model = torch.nn.Linear(1, 1)

# Init loss function
criterion = torch.nn.MSELoss()

# Init opimiser
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

# Init tensorboard writer
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/simple_tensorboard_1')

# Define our basic training loop
def train_model(epochs):
    for epoch in range(epochs):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        y_hat = model(x)
        loss = criterion(y_hat, y)

        # Add out loss to be tracked
        writer.add_scalar("Loss/train", loss, epoch)
        
        loss.backward()
        optimizer.step()

    # Call flush() to make sure all pending events have been written to disk.
    writer.flush()
    # Closer write after we are finished with it
    writer.close()

if __name__ == '__main__':
    train_model(10)
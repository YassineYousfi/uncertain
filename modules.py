import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, dropout=0.05, hidden_size=128, num_layers=2, num_outputs=2):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(1, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout))
    for _ in range(num_layers):
      self.model.append(nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout)))
    self.model.append(nn.Linear(hidden_size, num_outputs))
  def forward(self, x):
    return self.model(x)
  
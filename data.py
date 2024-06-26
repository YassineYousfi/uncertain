import torch
import numpy as np
from torch.utils.data import Dataset

def squiggle(x):
  return np.sin(x) * np.exp(-0.5*x**2) * (x+x**2+x**3+x**4-0.5*x**5-0.5*x**6)

class Data(Dataset):
  """
  Sample some data and add noise
  Optionally reject samples \in mask_between 
  """
  def __init__(self, func, n=1000, low=-6, high=6,
               noise_model=np.random.randn, sigma=.2, reject_mult=5, seed=None):
    super().__init__()
    np.random.seed(seed)
    self.n = n
    self.x = np.random.uniform(low=low, high=high, size=(reject_mult*self.n))
    self.y = func(self.x) 
    self.x = self.x[np.isfinite(self.y)][:self.n]
    self.y = self.y[np.isfinite(self.y)][:self.n]
    self.y += noise_model(self.n) * sigma
    # poor man's rejection sampling
    assert len(self.x) == self.n, 'increase reject_mult'
    self.x, self.y = torch.from_numpy(self.x[...,None]).float(), torch.from_numpy(self.y[...,None]).float()
  def __len__(self):
    return self.n
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
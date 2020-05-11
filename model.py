import torch.nn as nn
import torch.optim as optim
from torch import randn

# TODO: remove these
####################
n_in = 2
n_h = 64
n_latent = 2
n_noise = 2
####################

class AutoEncoder(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_latent)
        )

        self.decode = nn.Sequential(
            nn.Linear(n_latent, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_in)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def minimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Classifier(nn.Module):
    def __init__(self, lr):
        super().__init__()
    
        self.main = nn.Sequential(
            nn.Linear(n_latent, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, 1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        return self.main(x)
    
    def maximize(self, loss):
        self.optimizer.zero_grad()
        (-loss).backward()
        self.optimizer.step()


class Generator(nn.Module):
    def __init__(self, lr):
        super().__init__()
    
        self.main = nn.Sequential(
            nn.Linear(n_noise, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_in)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, batch_size):
        return self.main(randn(batch_size, n_noise))
    
    def minimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
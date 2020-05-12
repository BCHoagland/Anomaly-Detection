import torch.nn as nn
import torch.optim as optim
from torch import randn, randn_like

# TODO: remove these
####################
n_in = 2
n_h = 64
n_latent = 2
####################

class VariationalAutoEncoder(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh()
        )

        self.mean = nn.Sequential(
            nn.Linear(n_h, n_latent)
        )

        self.log_std = nn.Sequential(
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
    
    def encode(self, x):
        x = self.main(x)
        mean = self.mean(x)
        std = self.log_std(x).exp()
        return mean + std * randn_like(std)
    
    def generate(self, batch_size):
        noise = randn(batch_size, n_latent)
        return self.decode(noise)
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def forward_with_dist(self, x):
        x = self.main(x)
        mean = self.mean(x)
        std = self.log_std(x).exp()
        latent = mean + std * randn_like(std)
        return self.decode(latent), mean, std
    
    def minimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
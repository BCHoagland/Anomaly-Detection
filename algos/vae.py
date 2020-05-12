import torch
from visualize import scatter, map

#####################################
# VARIATIONAL AUTO-ENCODER TRAINING #
#####################################

class VaeAlgo:
    def step(ae, data):
        out, mean, std = ae.forward_with_dist(data)

        # minimize regularized MSE
        mse = ((out - data) ** 2).mean()
        kl = 0.5 * (mean**2 + std**2 - torch.log(std**2) - 1).sum(dim=1).mean()
        loss = mse + kl
        ae.minimize(loss)

        # plot loss
        return (loss.item(), 'VAE loss')

    def vis(ae):
        scatter(ae.generate(500), win='VAE Generated', name='VAE', color=[200,100,0])
        map(lambda x: torch.norm(ae(x) - x, dim=-1), 'VAE Error')
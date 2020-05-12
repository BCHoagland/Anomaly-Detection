import torch
from visualize import scatter, map

#########################
# AUTO-ENCODER TRAINING #
#########################

class AeAlgo:
    def step(ae, data):
        # minimize MSE
        loss = ((ae(data) - data) ** 2).mean()
        ae.minimize(loss)

        # plot loss
        return (loss.item(), 'AE loss')


    def vis(ae):
        scatter(ae.generate(500), win='AE Generated', name='VAE', color=[255,0,0])
        map(lambda x: torch.norm(ae(x) - x, dim=-1), 'AE Error')
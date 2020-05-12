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
        map(lambda x: torch.norm(ae(x) - x), 'AE Error')

        # determine AE error on generated data -> use to determine error threshold
        with torch.no_grad():
            generated_data = ae.generate(1000)
            errors = torch.norm(ae(generated_data) - generated_data, dim=1)
            threshold = max(errors)

        # plot AE-error method's decision boundary
        def border(threshold, x):
            ae_error = torch.norm(ae(x) - x)
            return 1 if ae_error < threshold else 0
        map(lambda x: border(threshold, x), 'AE classification')
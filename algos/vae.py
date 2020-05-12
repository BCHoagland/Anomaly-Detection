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
        map(lambda x: torch.norm(ae(x) - x), 'VAE Error')

        # determine AE error on generated data -> use to determine error threshold
        with torch.no_grad():
            generated_data = ae.generate(1000)
            errors = torch.norm(ae(generated_data) - generated_data, dim=1)
            threshold = max(errors)

        # plot AE-error method's decision boundary
        def border(threshold, x):
            ae_error = torch.norm(ae(x) - x)
            return 1 if ae_error < threshold else 0
        map(lambda x: border(threshold, x), 'VAE classification')

        #! when making decision, maybe perturb point a bit to see what surrounding points are like
        # def soft_border(threshold, x):
        #     x_perturb = [x + torch.randn_like(x) for _ in range(5)]
        #     all_x = [x] + x_perturb
        #     avg_ae_error = sum([torch.norm(auto_encoder(x) - x) for x in all_x]) / len(all_x)
        #     return 1 if avg_ae_error < threshold else 0
        # map(lambda x: soft_border(threshold, x), 'Soft AE classification')

import torch
from torch.autograd import grad
from visualize import scatter, map

λ = 10

############################
# WASSERSTEIN-GAN TRAINING #
############################

def interpolate(x1, x2):
    i = torch.rand_like(x1)
    return (i * x1) + ((1 - i) * x2)

class WganAlgo:
    def step(gan, data):
        # necessary data
        for _ in range(5):
            generated_data = gan.generate(len(data))
            interpolated_data = interpolate(data, generated_data)

            # gradient penalty
            out = gan.classify(interpolated_data)
            grads = grad(out, interpolated_data, torch.ones(out.shape), create_graph=True)[0]
            grad_penalty = λ * ((torch.norm(grads, dim=-1) - 1) ** 2)

            # improve classifier
            obj = (gan.classify(data) - gan.classify(generated_data.detach()) - grad_penalty).mean()
            gan.maximize(gan.classifier_optimizer, obj)

        # improve generator
        loss = -gan.classify(gan.generate(len(data))).mean()
        gan.minimize(gan.generator_optimizer, loss)

        return [(obj.item(), 'Wasserstein objective'), (loss.item(), 'WGAN Generator loss')]

    def vis(gan):
        scatter(gan.generate(500), win='WGAN Generated', name='WGAN', color=[0,0,255])
        map(gan.classify, 'WGAN Classification')
import torch
from visualize import scatter, map

################
# GAN TRAINING #
################

class GanAlgo:
    def step(gan, data):
        # improve classifier
        generated_data = gan.generate(len(data))
        obj = (torch.log(gan.classify(data)) + torch.log(1 - gan.classify(generated_data))).mean()
        gan.maximize(gan.classifier_optimizer, obj)

        # improve generator
        loss = torch.log(1 - gan.classify(gan.generate(len(data)))).mean()
        gan.minimize(gan.generator_optimizer, loss)

        return [(obj.item(), 'GAN Classifier Objective'), (loss.item(), 'GAN Generator loss')]

    def vis(gan):
        scatter(gan.generate(500), win='GAN Generated', name='GAN', color=[0,255,0])
        map(gan.classify, 'GAN Classification')

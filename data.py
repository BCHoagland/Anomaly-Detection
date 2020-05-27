import numpy as np
import torch
from torch.distributions import MultivariateNormal, Uniform
from math import floor


bad_data_prob = 0.0001

##########################
# DISTRIBUTION FUNCTIONS #
##########################
def set_cov_matrix(stds):
    n = len(stds)
    cov_matrix = torch.zeros((n, n))
    for i in range(n):
        cov_matrix[i][i] = stds[i]**2
    return cov_matrix

def make_normal_dist(mean, std):
    return MultivariateNormal(torch.FloatTensor(mean), set_cov_matrix(std))


#############
# GOOD DATA #
#############
# μ = [[1, 6], [-5, -10]]
# σ = [[2, 3], [4, 2]]
mean = [1, 6]
std = [2, 3]
good_dist = make_normal_dist(mean, std)


##################
# ANOMALOUS DATA #
##################
# μ = [[25, -10], [-10, 25], [0, -24]]
# σ = [[1, 2], [3, 1], [2, 2]]
# bad_dists = [make_normal_dist(mean, std) for mean, std in zip(μ, σ)]
bad_dist = Uniform(torch.tensor([-30.0, -30.0]), torch.tensor([30.0, 30.0]))


######################
# SAMPLING FUNCTIONS #
######################

# sample from both distributions, with the given probability of choosing bad data
def sample_data(batch_size):
    p = [1.-bad_data_prob, bad_data_prob]
    dists = [good_dist, bad_dist]

    data = []
    for _ in range(batch_size):
        dist = np.random.choice(dists)
        data.append(dist.sample())
    return torch.stack(data)
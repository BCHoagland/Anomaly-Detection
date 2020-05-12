import numpy as np
import torch
from torch.distributions import MultivariateNormal, Uniform
from math import floor


bad_data_prob = 0.01

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
μ = [[1, 6]]
σ = [[2, 3]]
good_dists = [make_normal_dist(mean, std) for mean, std in zip(μ, σ)]


##################
# ANOMALOUS DATA #
##################
# μ = [[25, -10], [-10, 25], [0, -24]]
# σ = [[1, 2], [3, 1], [2, 2]]
# bad_dists = [make_normal_dist(mean, std) for mean, std in zip(μ, σ)]
bad_dists = [Uniform(torch.tensor([-30.0, -30.0]), torch.tensor([30.0, 30.0]))]


######################
# SAMPLING FUNCTIONS #
######################

# sample from a given array of distributions
def sample_from_dists(batch_size, dists):
    data = []
    for _ in range(batch_size):
        dist = np.random.choice(dists)
        data.append(dist.sample())
    return torch.stack(data)

# sample from both distributions, with the given probability of choosing bad data
def sample_data(batch_size):
    bad_size = floor(batch_size * bad_data_prob)
    good_data = sample_from_dists(batch_size-bad_size, good_dists)
    bad_data = sample_from_dists(bad_size, bad_dists)
    return torch.cat((good_data, bad_data), dim=0)
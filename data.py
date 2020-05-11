import numpy as np
import torch
from torch.distributions import MultivariateNormal


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
μ = [[15, 5]]
σ = [[1, 2]]
bad_dists = [make_normal_dist(mean, std) for mean, std in zip(μ, σ)]


######################
# SAMPLING FUNCTIONS #
######################

# sample from a given array of distributions
def sample_from_dists(batch_size, dists):
    data = []
    n = len(dists)
    for dist in dists:
        data.append(dist.sample(torch.Size([batch_size // n])))
    data = torch.stack(data)
    return data.view(-1, data.shape[2])

# sample from both distributions, with the given probability of choosing bad data
def sample_data(batch_size, bad_data_prob):
    data = sample_from_dists(batch_size, good_dists)
    for i in range(len(data)):
        if np.random.random() < bad_data_prob:
            data[i] = sample_from_dists(1, bad_dists)
    return data
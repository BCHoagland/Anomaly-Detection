import torch
import numpy as np

from data import sample_data
from model import VariationalAutoEncoder
from visualize import scatter, line, heatmap


# hyperparameters
bad_data_prob = 0.01
batch_size = 128
lr = 3e-4


def train(network_class, network_name, epochs, train_step, vis=None, use_saved_model=True):
    net = network_class(lr)

    save_path = f'model_params/{network_name}'

    # if a saved model exists, use that
    if use_saved_model:
        try:
            saved_params = torch.load(save_path)
            net.load_state_dict(saved_params)
            return net
        # if not, train a new one and save it
        except FileNotFoundError:
            pass

    # training loop
    for epoch in range(epochs):
        # Take an optimization step and visualize if necessary
        train_step(net)
        if vis is not None and epoch % 100 == 99:
            with torch.no_grad(): vis(net)

    # save final model
    torch.save(net.state_dict(), save_path)
        
    return net


def map(fn, name):
    with torch.no_grad():
        x_range = torch.arange(-30, 30)
        y_range = torch.arange(-30, 30)
        arr = torch.zeros((len(x_range), len(y_range)))
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                x, y = x_range[i], y_range[j]
                arr[i][j] = fn(torch.FloatTensor([x, y]))
        heatmap(arr, name, x_range.tolist(), y_range.tolist())


#########################
# AUTO-ENCODER TRAINING #
#########################

def auto_encoder_step(auto_encoder):
    inp = sample_data(batch_size, bad_data_prob)
    out, mean, std = auto_encoder.forward_with_dist(inp)

    # minimize regularized MSE
    mse = ((out - inp) ** 2).mean()
    kl = 0.5 * (mean**2 + std**2 - torch.log(std**2) - 1).sum(dim=1).mean()
    loss = mse + kl
    auto_encoder.minimize(loss)

    # plot loss
    line(loss.item(), 'AE loss')

def auto_encoder_vis(auto_encoder):
    scatter(auto_encoder.generate(500), 'Generated', color=[255,0,0])


#################
# FULL PIPELINE #
#################

# plot a large sample of the data (bad points included)
scatter(sample_data(500, bad_data_prob))

# train AE and classifier
auto_encoder = train(VariationalAutoEncoder, 'auto_encoder', 2000, auto_encoder_step, auto_encoder_vis, use_saved_model=True)

# testing
map(lambda x: torch.norm(auto_encoder(x) - x), 'AE Error')

# determine AE error on generated data -> use to determine error threshold
with torch.no_grad():
    generated_data = auto_encoder.generate(1000)
    errors = torch.norm(auto_encoder(generated_data) - generated_data, dim=1)
    threshold = np.percentile(errors, 99.9)

# plot AE-error method's decision boundary
def border(threshold, x):
    ae_error = torch.norm(auto_encoder(x) - x)
    return 1 if ae_error < threshold else 0
map(lambda x: border(threshold, x), 'AE classification')


#! when making decision, maybe perturb point a bit to see what surrounding points are like
def border(threshold, x):
    x_perturb = [x + torch.randn_like(x) for _ in range(10)]
    all_x = [x] + x_perturb
    avg_ae_error = sum([torch.norm(auto_encoder(x) - x) for x in all_x]) / len(all_x)
    return 1 if avg_ae_error < threshold else 0
map(lambda x: border(threshold, x), 'Soft AE classification')
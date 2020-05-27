import torch

from data import sample_data
from model import AutoEncoder, VariationalAutoEncoder, Gan, WGan
from visualize import scatter, line, heatmap
import algos.ae, algos.vae, algos.gan, algos.wgan


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# hyperparameters
batch_size = 512
lr = 3e-4
vis_iter = 50


def train(network_class, network_name, epochs, algo, use_saved_model=True):
    net = network_class(lr).to(device)

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
        data = sample_data(batch_size).to(device)
        stats = algo.step(net, data)
        if epoch % vis_iter == vis_iter - 1:
            if isinstance(stats, tuple):
                line(epoch, *stats)
            else:
                for stat in stats: line(epoch, *stat)
        if epoch % 500 == 499:
            with torch.no_grad(): algo.vis(net)

    # save final model
    torch.save(net.state_dict(), save_path)
        
    return net


#################
# FULL PIPELINE #
#################

# plot a large sample of the data (bad points included)
scatter(sample_data(500))

# train AE and GAN
# ae = train(AutoEncoder, 'ae', 20000, algos.ae.AeAlgo, use_saved_model=True)
vae = train(VariationalAutoEncoder, 'vae', 20000, algos.vae.VaeAlgo, use_saved_model=True)
# gan = train(Gan, 'gan', 20000, algos.gan.GanAlgo, use_saved_model=True)
# wgan = train(WGan, 'wgan', 20000, algos.wgan.WganAlgo, use_saved_model=True)
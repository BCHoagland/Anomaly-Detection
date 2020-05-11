import torch

from data import sample_data
from model import AutoEncoder, Classifier, Generator
from visualize import scatter, line, heatmap


# hyperparameters
bad_data_prob = 0.02
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
        x_range = torch.arange(-20, 20)
        y_range = torch.arange(-20, 20)
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
    out = auto_encoder(inp)

    # minimizing step on MSE loss
    loss = ((out - inp) ** 2).mean()
    auto_encoder.minimize(loss)

    # plot loss
    line(loss.item(), 'AE loss')


######################
# GENERATOR TRAINING #
######################

def generator_step(auto_encoder, generator):
    data = sample_data(batch_size, bad_data_prob)

    # improve generator
    loss = ((data - generator(batch_size))**2).mean()
    generator.minimize(loss)

    # plot objective
    line(loss.item(), 'Generator Loss')


def generator_vis(auto_encoder, generator):
    with torch.no_grad(): scatter(auto_encoder.decode(generator(500)), 'Generated', color=[255,0,0])


#######################
# CLASSIFIER TRAINING #
#######################

def classifier_step(auto_encoder, classifier):
    # sample data and use AE to encode it
    data = auto_encoder.encode(sample_data(batch_size, bad_data_prob))

    # improve classifier
    obj = torch.log(classifier(data)).mean()
    classifier.maximize(obj)

    # plot objective
    line(obj.item(), 'Classifier Objective')


def classifier_vis(auto_encoder, classifier):
    map(lambda x: classifier(auto_encoder.encode(x)), 'Classification Probabilities')


#################
# FULL PIPELINE #
#################

# plot a large sample of the data (bad points included)
scatter(sample_data(500, bad_data_prob))

# train AE and classifier
auto_encoder = train(AutoEncoder, 'auto_encoder', 1000, auto_encoder_step)
generator = train(Generator, 'generator', 1000, lambda x: generator_step(auto_encoder, x), lambda x: generator_vis(auto_encoder, x), use_saved_model=False)
classifier = train(Classifier, 'classifier', 1000, lambda x: classifier_step(auto_encoder, x), lambda x: classifier_vis(auto_encoder, x))

# testing
map(lambda x: torch.norm(auto_encoder(x) - x, 2), 'AE Error')
map(lambda x: classifier(auto_encoder.encode(x)), 'Classification Probabilities')
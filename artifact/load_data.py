import os
import random

import torch
import torchvision
from torch.utils.data import TensorDataset

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data parameters
num_classes = 10
input_shape = (1, 28, 28)

# drop slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]

def load(train_size=1):
    """
    # Load the data
    """

    # the data, split between train and test sets
    train = torchvision.datasets.MNIST("./", train=True, download=True)

    (x_train, y_train) = (train.data, train.targets)

    # split off a validation set for hyperparameter tuning
    x_train = x_train[:train_size].clone()

    y_train = y_train[:train_size].clone()

    training_set = TensorDataset(x_train, y_train)

    datasets = [training_set]

    return datasets


def preprocess(dataset, normalize=True, expand_dims=True):
    """
    ## Prepare the data
    """
    x, y = dataset.tensors

    if normalize:
        # Scale images to the [0, 1] range
        x = x.type(torch.float32) / 255

    if expand_dims:
        # Make sure images have shape (1, 28, 28)
        x = torch.unsqueeze(x, 1)

    return TensorDataset(x, y)


def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)

if __name__ == '__main__':
    data = load()

import pickle

import numpy as np

import network
import train
import parse_mnist

network = None
with open('network.ann', 'rb') as f:
    network = pickle.load(f)

data = parse_mnist.load_data()

print(train.get_accuracy(network, data))


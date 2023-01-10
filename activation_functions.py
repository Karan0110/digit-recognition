import numpy as np

def sigmoid(x):
    return (x / (1+np.abs(x)) + 1) / 2
    #return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

def relu(x):
    return np.maximum(x,0)

def relu_deriv(x):
    return (np.sign(x) + 1)/2

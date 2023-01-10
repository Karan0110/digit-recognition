import numpy as np

from util import basis_vector

import activation_functions

class NeuralNetwork(object):
    def __init__(self, shape, weights=None, biases=None):
        self.shape = shape
        self.num_layers = len(self.shape)

        self.a = [None] * self.num_layers
        self.b = [None] * self.num_layers

        self.activation =       activation_functions.sigmoid
        self.activation_deriv = activation_functions.sigmoid_deriv

        if weights is None:
            self.weights = [(10* np.random.random((self.shape[i+1],self.shape[i])) - 5) for i in range(self.num_layers-1)]
        else:
            self.weights = weights

        if biases is None:
            self.biases =  [(10 * np.random.random(self.shape[i+1]) - 5) for i in range(self.num_layers-1)]
        else:
            self.biases = biases
    
    def copy(self):
        return NeuralNetwork(self.shape, [np.array(W) for W in self.weights], [np.array(B) for B in self.biases])

    def process(self, inputs):
        if len(inputs) != self.shape[0]:
            raise TypeError("Input is size {}, when it should be {}".format(len(inputs), self.shape[0]))

        self.a[0] = np.array(inputs)

        for i in range(1,self.num_layers):
            self.b[i] =  self.biases[i-1] + np.matmul(self.weights[i-1],self.a[i-1])
            self.a[i] = self.activation(self.b[i])

        output = self.a[-1]
        return output

    #derivative of cost function at x with respect to W_ij^{k}
    #network must have correct values for input x already
    def grad_w(self, x, k, i, j, target):
        grad_w_a = self.activation_deriv(self.b[k+1]) * basis_vector(self.shape[k+1], i) * self.a[k][j]
        for l in range(k+2, self.num_layers):
            grad_w_a = self.activation_deriv(self.b[l]) * np.matmul(self.weights[l-1], grad_w_a)
        return 2 * np.dot(self.a[-1] - target, grad_w_a)

    #derivative of cost function at x with respect to B_i^{k}
    #network must have correct values for input x already
    def grad_b(self, x, k, i, target):
        grad_b_a = self.activation_deriv(self.b[k+1]) * basis_vector(self.shape[k+1], i)        
        for l in range(k+2, self.num_layers):
            grad_b_a = self.activation_deriv(self.b[l]) * np.matmul(self.weights[l-1], grad_b_a)
        return 2 * np.dot(self.a[-1] - target, grad_b_a)




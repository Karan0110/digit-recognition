import pickle
import sys
import itertools

import numpy as np

from network import NeuralNetwork
from parse_mnist import load_data

STEP_SIZE = 10.0
BATCH_SIZE = 100

#derivative of cost function at x with respect to W_ij^{k}
#network must have correct values for input x already
def grad_w(network, x, k, i, j, target):
    grad_w_a = network.activation_deriv(network.b[k+1]) * basis_vector(network.shape[k+1], i) * network.a[k][j]
    for l in range(k+2, network.num_layers):
        grad_w_a = network.activation_deriv(network.b[l]) * np.matmul(network.weights[l-1], grad_w_a)
    return 2 * np.dot(network.a[-1] - target, grad_w_a)

#derivative of cost function at x with respect to B_i^{k}
#network must have correct values for input x already
def grad_b(network, x, k, i, target):
    grad_b_a = network.activation_deriv(network.b[k+1]) * basis_vector(network.shape[k+1], i)        
    for l in range(k+2, network.num_layers):
        grad_b_a = network.activation_deriv(network.b[l]) * np.matmul(network.weights[l-1], grad_b_a)
    return 2 * np.dot(network.a[-1] - target, grad_b_a)

def cost(network, data, samples=None):
    c = 0.

    if samples is None:
        for x, target in data:
            output = network.process(x)
            c += np.sum((target - output)**2)
        c /= len(data)
    else:
        indexes = np.random.randint(0, len(data), samples)
        for index in indexes:
            x, target = data[index]
            output = network.process(x)
            c += np.sum((target-output)**2)
        c /= samples

    return c

def apply_stochastic_gradient_descent(network, data, step_size=STEP_SIZE, batch_size=BATCH_SIZE):
    batch = np.random.randint(0, len(data), batch_size)

    for index in batch:
        x, target = data[index]

        output = network.process(x)
        
        for k in range(network.num_layers-1):
            for i in range(network.shape[k+1]):
                grad = network.grad_b(x, k, i, target)
                network.biases[k][i] -= grad * (step_size/batch_size)
                for j in range(network.shape[k]):
                    grad = network.grad_w(x, k, i, j, target)
                    network.weights[k][i][j] -= grad * (step_size/batch_size)

def get_accuracy(network, data):
    correct = 0
    for x, target in data:
        output = network.process(x)
        if np.argmax(output) == np.argmax(target):
            correct += 1
    return correct / len(data)

if __name__ == '__main__':
    print("Loading training data...")
    training_data = load_data('training')
    testing_data = load_data('testing')
    print("Loaded training data")
    
    network = None
    with open('network.ann', 'rb') as file:
        network = pickle.load(file)
    last_network = network
    last_distance = None

    try:
        for epoch in itertools.count():
            if epoch % 50 == 0:
                distance = cost(network, training_data)
                training_accuracy = get_accuracy(network, training_data)
                testing_accuracy = get_accuracy(network, testing_data)

                print("Cost: {}".format(distance)) 
                print("Accuracy on training data: {}%".format(training_accuracy * 100))
                print("Accuracy on testing data: {}%".format(testing_accuracy * 100))

            if epoch % 10 == 0:
                print("Epoch {}".format(epoch))

            last_network = network.copy()

            apply_stochastic_gradient_descent(network, training_data, 0.2)
    except KeyboardInterrupt:
        pass

    
    filepath = 'network.ann' if len(sys.argv) <= 1 else "{}.ann".format(sys.argv[1])
    with open(filepath, 'wb') as file:
        pickle.dump(last_network, file)
    print("\nSaved network to {}.".format(filepath))

                

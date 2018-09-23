import numpy as np

from neural_network import NeuralNetwork

net = NeuralNetwork((2, 2, 1))

X = list(map(np.array, [(0, 0), (0, 1), (1, 0), (1, 1)]))
y  = list(map(np.array, [(0,), (1,), (1,), (0,)]))

net.fit(X, y, num_epochs=10000)

for i in X:
	print("I think {} ^ {} = {}".format(i[0], i[1], net.forward_propagate(i)[0]))


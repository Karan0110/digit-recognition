import time

import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
	x = np.array(x)

	return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x, sigmoid_x=None):
	x = np.array(x)

	if sigmoid_x is None:
		sigmoid_x = sigmoid(x)

	return sigmoid_x * (1 - sigmoid_x)

def _minibatch_generator(X, y, minibatch_size):
	i = 0

	#shuffles X and y, keeping paired data in same index
	pairs = list(zip(X, y))
	np.random.shuffle(pairs)
	X, y = (list(i) for i in zip(*pairs))

	while True:
		if i + minibatch_size >= len(X):
			i = 0

			#shuffles X and y, keeping paired data in same index
			pairs = list(zip(X, y))
			np.random.shuffle(pairs)
			X, y = (list(i) for i in zip(*pairs))

		data_slice = slice(i, i + minibatch_size)
		yield (X[data_slice], y[data_slice])	

		i += minibatch_size

class NeuralNetwork(object):
	def __init__(self, layer_sizes, weights=None, biases=None):
		self.layer_sizes = layer_sizes	
		self.num_layers = len(self.layer_sizes)
		
		self.activation = sigmoid
		self.activation_deriv = sigmoid_deriv

		self.z = [None] * self.num_layers
		self.a = [None] * self.num_layers

		self.w = [np.nan] + [2 * np.random.random((self.layer_sizes[L], self.layer_sizes[L-1])) - 1 for L in range(1, self.num_layers)]
		self.b = [np.nan] + [2 * np.random.random(x) - 1 for x in self.layer_sizes[1:]]
	
	def forward_propagate(self, inputs):
		#temporarily convert a[L], z[L] and b[L] to column vectors (inductively in the case of a and z) to allow dot product
		#this is reverted at end of function
		self.z[0] = inputs.reshape((-1, 1))
		self.a[0] = inputs.reshape((-1, 1))

		self.b = [np.nan] + [x.reshape((-1, 1)) for x in self.b[1:]]

		for L in range(1, self.num_layers):
			self.z[L] = self.w[L].dot(self.a[L-1]) + self.b[L]
			self.a[L] = self.activation(self.z[L])

		#reverts a[L], z[L] and b[L] back into row vectors
		self.a = [x.reshape((-1,)) for x in self.a]
		self.z = [x.reshape((-1,)) for x in self.z]
		self.b = [np.nan] + [x.reshape((-1,)) for x in self.b[1:]]

		return self.a[-1]

	def backpropagate(self, X, y, learning_rate):
		net_weight_change = [np.nan] + [np.zeros((self.layer_sizes[L], self.layer_sizes[L-1])) for L in range(1, self.num_layers)]
		net_bias_change = [np.nan] + list(map(np.zeros, self.layer_sizes[1:]))

		for inputs, expected_outputs in zip(X, y):
			self.forward_propagate(inputs)

			deltas = list(map(np.empty, self.layer_sizes))
			deltas[-1] = 2 * (self.a[-1] - expected_outputs)
			for L in reversed(range(self.num_layers - 1)): 	
				deltas[L] = sum(deltas[L+1][k] * self.activation_deriv(self.z[L+1][k], self.a[L+1][k]) * self.w[L+1][k] for k in range(self.layer_sizes[L+1]))

			cost_weight_deriv = [np.nan] * self.num_layers
			for L in range(1, self.num_layers):
				cost_weight_deriv[L] = np.empty((self.layer_sizes[L], self.layer_sizes[L-1]))

				for i in range(self.layer_sizes[L]):
					for j in range(self.layer_sizes[L-1]):
						cost_weight_deriv[L][i][j] = deltas[L][i] * self.activation_deriv(self.z[L][i], self.a[L][i]) * self.a[L-1][j]

			cost_bias_deriv = [np.nan] * self.num_layers
			for L in range(1, self.num_layers):
				cost_bias_deriv[L] = deltas[L] * self.activation_deriv(self.z[L], self.a[L])

			for L in range(1, self.num_layers):
				net_weight_change[L] -= cost_weight_deriv[L]
				net_bias_change[L] -= cost_bias_deriv[L]

		weight_change = [(x / len(X)) * learning_rate for x in net_weight_change]
		bias_change = [(x / len(X)) * learning_rate for x in net_bias_change]

		self.w = [np.nan] + [self.w[L] + weight_change[L] for L in range(1, self.num_layers)]	
		self.b = [np.nan] + [self.b[L] + bias_change[L] for L in range(1, self.num_layers)]	

	def fit(self, X, y, minibatch_size=None, num_epochs=1000, learning_rate=0.5, reporting=False):
		if minibatch_size is None:
			minibatch_size = len(X)
		mbg = _minibatch_generator(X, y, minibatch_size)

		start_time = time.time()

		for epoch in range(num_epochs):
			start = time.time()
			
			minibatch = next(mbg)

			if reporting and epoch % int(reporting) == 0: 
				print("Epoch {} ({} seconds elapsed)".format(epoch, (time.time() - start_time)))
		
			self.backpropagate(*minibatch, learning_rate=learning_rate)


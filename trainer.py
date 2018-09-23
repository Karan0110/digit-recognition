import pickle

import numpy as np
from mnist import MNIST

from neural_network import NeuralNetwork

def one_hot_encode(x, num_categories):
	encoding = np.zeros(num_categories)
	encoding[x] = 1.

	return encoding

def one_hot_decode(x):
	return x.argmax()

def get_data(train_test_split=0.9):
	mndata = MNIST('./MNIST')
	X, y = mndata.load_training()

	#converts images into flat array with brightness in [0, 1], instead of [0, 255]
	X = [np.array(i).reshape((-1,)) / 255 for i in X]

	#converts digit into one-hot encoding
	y = [one_hot_encode(i, 10) for i in y[:-1]]

	train_test_divide = int(len(X) * train_test_split)
	X_train, X_test = X[:train_test_divide], X[train_test_divide:]
	y_train, y_test = y[:train_test_divide], y[train_test_divide:]

	return X_train, X_test, y_train, y_test

def get_prediction(net, inputs):
	outputs = net.forward_propagate(inputs)

	return one_hot_decode(outputs)

def get_accuracy(net, X, y):
	correct = 0

	for inputs, expected_outputs in zip(X, y):
		prediction = get_prediction(net, inputs)
		if prediction == one_hot_decode(expected_outputs):
			correct += 1

	return correct / len(X)

np.random.seed(1)

net = NeuralNetwork((28 ** 2, 16, 16, 10))

print("Getting data...")

X_train, X_test, y_train, y_test = get_data()

print("Training...")

net.fit(X_train, y_train, minibatch_size=10, num_epochs=50000, learning_rate=.5, reporting=100)

print("Saving neural network...")
with open('network.ann', 'wb') as f:
	pickle.dump(net, f)

print("Finding accuracy...")

training_accuracy = get_accuracy(net, X_train, y_train)
testing_accuracy = get_accuracy(net, X_test, y_test)

print("Accuracy for training data: {}%".format(training_accuracy * 100))
print("Accuracy for testing data: {}%".format(testing_accuracy * 100))

#92-93%
#92-93%


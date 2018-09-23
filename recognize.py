import sys

import pickle
from PIL import Image

import numpy as np

from neural_network import NeuralNetwork

def img_to_input(img):
	img = img.convert('L')

	inp = np.array(img.getdata()) / 255

	return inp
	
def get_prediction(net, file_path):
	img = Image.open(file_path)

	network_input = img_to_input(img)

	out = net.forward_propagate(network_input)

	return out.argmax()

with open('network.ann', 'rb') as f:
	net = pickle.load(f)

if len(sys.argv) <= 1:
	file_path = input("Enter file path: ")
	print(get_prediction(net, file_path))
else:
	for fp in sys.argv[1:]:
		print("{} - {}".format(fp, get_prediction(net, fp)))	


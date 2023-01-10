import numpy as np
import matplotlib.pyplot as plt

FILES = {
    'training-labels': 'mnist/train-labels-idx1-ubyte', 
    'training-images': 'mnist/train-images-idx3-ubyte', 
    'testing-labels':  'mnist/t10k-labels-idx1-ubyte', 
    'testing-images':  'mnist/t10k-images-idx3-ubyte',
} 

def bytes_to_int(xs):
    xs = xs[::-1]
    ans = 0
    power = 1

    for x in xs:
        ans += x * power
        power *= 256

    return ans

def one_hot_encode(digit):
    vec = np.zeros(10)
    vec[digit] = 1.
    return vec

def save_to_png(img, filepath):
    pixel_plot = plt.figure()
    #pixel_plot.add_axes([0,0, 100, 100])
    
    plt.title('MNIST')
    
    pixel_plot = plt.imshow(img, cmap='gray', interpolation='nearest')
    
    plt.colorbar(pixel_plot)
    
    plt.savefig(filepath)

def visualise(img):
    pixel_plot = plt.figure()
    
    plt.title('MNIST')
    
    pixel_plot = plt.imshow(img, cmap='gray', interpolation='nearest')
    
    plt.colorbar(pixel_plot)
    
    plt.show()

def load_data(s='training'):
    data = []
    
    label_file = open(FILES['{}-labels'.format(s)], 'rb')
    image_file = open(FILES['{}-images'.format(s)], 'rb')
    
    label_magic_number = bytes_to_int(label_file.read(4))
    if label_magic_number != 2049:
        print("Error: wrong magic number for training label file")
        return
    label_num_items = bytes_to_int(label_file.read(4))
    
    image_magic_number = bytes_to_int(image_file.read(4))
    if image_magic_number != 2051:
        print("Error: wrong magic number for training image file")
        return
    image_num_items = bytes_to_int(image_file.read(4))
    num_rows = bytes_to_int(image_file.read(4))
    num_cols = bytes_to_int(image_file.read(4))
    
    if image_num_items != label_num_items:
        print("Error: number of images doesn't match number of labels")
        return
    num_items = image_num_items
    
    for i in range(num_items):
        digit = one_hot_encode(ord(label_file.read(1)))
        img = np.array([ord(image_file.read(1)) for i in range(num_rows*num_cols)]) / 255
        data.append((img, digit))

    return data


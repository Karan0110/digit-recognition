import numpy as np

def basis_vector(n, i):
    vec = np.zeros(n)
    vec[i] = 1.
    return vec



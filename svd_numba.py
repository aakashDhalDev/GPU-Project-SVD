import numpy as np
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
from numba import jit, prange

@jit(nopython=True, parallel=True)
def random_unit_vector(size):
    unnormalized = [normalvariate(0, 1) for _ in range(size)]
    norm_val = sqrt(sum(v * v for v in unnormalized))
    return [v / norm_val for v in unnormalized]

@jit(nopython=True, parallel=True)
def power_iterate(X, epsilon=1e-10):
    n, m = X.shape
    start_v = random_unit_vector(m)
    prev_eigenvector = np.zeros(m)
    curr_eigenvector = start_v
    covariance_matrix = np.dot(X.T, X)

    it = 0
    while True:
        it += 1
        prev_eigenvector = curr_eigenvector
        curr_eigenvector = np.dot(covariance_matrix, prev_eigenvector)
        curr_eigenvector = curr_eigenvector / norm(curr_eigenvector)

        if abs(np.dot(curr_eigenvector, prev_eigenvector)) > 1 - epsilon:
            return curr_eigenvector

@jit(nopython=True, parallel=True)
def svd(X, epsilon=1e-10):
    n, m = X.shape
    change_of_basis = np.empty((m, m))
    for i in prange(m):
        data_matrix = X.copy()

        for j in range(i):
            sigma, u, v = change_of_basis[j]
            data_matrix -= sigma * np.outer(u, v)

        v = power_iterate(data_matrix, epsilon=epsilon)
        u_sigma = np.dot(X, v)
        sigma = norm(u_sigma)
        u = u_sigma / sigma

        change_of_basis[i] = sigma, u, v

    sigmas, us, v_transposes = [np.array(x) for x in zip(*change_of_basis)]

    return sigmas, us.T, v_transposes

if __name__ == "__main__":
    dataset = np.random.random_sample((8, 3))
    results = svd(dataset)
    print("sigmas", results[0])
    print("u: data points in new coordinate system", results[1])
    print("v transpose: change of basis matrix", results[2])

import time
import numpy as np

from math import exp
from numba import jit, prange
from pycheeger import compute_cheeger


std = 0.2
coeffs = 0.5 * np.array([1.0, 1.0, -1.1, -0.7])
means = np.array([[-0.1, -0.4], [0.0, 0.4], [0.15, 0.0], [-0.65, -0.5]])


@jit(nopython=True, parallel=True)
def eta_aux1(x):
    squared_norm = (x[0] - means[0, 0]) ** 2 + (x[1] - means[0, 1]) ** 2
    res = coeffs[0] * exp(-squared_norm / (2 * std ** 2))

    for i in prange(1, len(coeffs)):
        squared_norm = (x[0] - means[i, 0]) ** 2 + (x[1] - means[i, 1]) ** 2
        res += coeffs[i] * exp(-squared_norm / (2 * std ** 2))

    return res


@jit(nopython=True, parallel=True)
def eta_aux2(x, res):
    for j in prange(x.shape[0]):
        for i in prange(len(coeffs)):
            squared_norm = (x[j, 0] - means[i, 0]) ** 2 + (x[j, 1] - means[i, 1]) ** 2
            res[j] += coeffs[i] * exp(-squared_norm / (2 * std**2))


@jit(nopython=True, parallel=True)
def eta_aux3(x, res):
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(len(coeffs)):
                squared_norm = (x[i, j, 0] - means[k, 0]) ** 2 + (x[i, j, 1] - means[k, 1]) ** 2
                res[i, j] += coeffs[k] * exp(-squared_norm / (2 * std**2))


def eta(x):
    if x.ndim == 1:
        return eta_aux1(x)
    elif x.ndim == 2:
        res = np.zeros(x.shape[0])
        eta_aux2(x, res)
        return res
    else:
        res = np.zeros((x.shape[0], x.shape[1]))
        eta_aux3(x, res)
        return res


start = time.time()
simple_set, obj_tab, grad_norm_tab = compute_cheeger(eta, max_tri_area=0.001, max_primal_dual_iter=20000,
                                                     step_size=1e-2, convergence_tol=1e-2)
end = time.time()

print(end - start)

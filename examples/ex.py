import time
import numpy as np

from numba import jit
from pycheeger import compute_cheeger


std = 0.2
coeffs = 0.5 * np.array([1.0, 1.0, -1.1, -0.7])
means = np.array([[-0.1, -0.4], [0.0, 0.4], [0.15, 0.0], [-0.65, -0.5]])


@jit(nopython=True)
def eta_aux1(x):
    squared_norm = (x[0] - means[0, 0]) ** 2 + (x[1] - means[0, 1]) ** 2
    res = coeffs[0] * np.exp(-squared_norm / (2 * std ** 2))

    for i in range(1, len(coeffs)):
        squared_norm = (x[0] - means[i, 0]) ** 2 + (x[1] - means[i, 1]) ** 2
        res += coeffs[i] * np.exp(-squared_norm / (2 * std ** 2))

    return res


@jit(nopython=True)
def eta_aux2(x, res):
    for j in range(x.shape[1]):
        for i in range(len(coeffs)):
            squared_norm = (x[0, j] - means[i, 0]) ** 2 + (x[1, j] - means[i, 1]) ** 2
            res[j] += coeffs[i] * np.exp(-squared_norm / (2 * std**2))


def eta(x):
    if x.ndim == 1:
        return eta_aux1(x)
    else:
        res = np.zeros(x.shape[1])
        eta_aux2(x, res)
        return res


start = time.time()
simple_set, obj_tab, grad_norm_tab = compute_cheeger(eta, max_tri_area=0.001, max_primal_dual_iter=20000, step_size=1e-2)
end = time.time()

print(end - start)

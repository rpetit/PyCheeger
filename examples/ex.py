import time
import numpy as np

from math import exp
from numba import jit, prange
from pycheeger import compute_cheeger


std = 0.2
coeffs = 0.5 * np.array([1.0, 1.0, -1.1, -0.7])
means = np.array([[-0.1, -0.4], [0.0, 0.4], [0.15, 0.0], [-0.65, -0.5]])


@jit(nopython=True, parallel=True)
def eta_aux(x, res):
    for i in prange(x.shape[0]):
        for j in prange(len(coeffs)):
            squared_norm = (x[i, 0] - means[j, 0]) ** 2 + (x[i, 1] - means[j, 1]) ** 2
            res[i] += coeffs[j] * exp(-squared_norm / (2 * std**2))


def eta(x):
    if x.ndim == 1:
        tmp = np.zeros(1)
        eta_aux(np.reshape(x, (1, 2)), tmp)
        res = tmp[0]
    else:
        res = np.zeros(x.shape[0])
        eta_aux(x, res)
    return res


start = time.time()

simple_set, obj_tab, grad_norm_tab = compute_cheeger(eta,
                                                     max_tri_area=0.001, max_primal_dual_iter=20000,
                                                     step_size=1e-2, convergence_tol=1e-2)

end = time.time()

print(end - start)

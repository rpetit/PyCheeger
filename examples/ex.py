import time
import numpy as np
import matplotlib.pyplot as plt

from math import exp
from numba import jit, prange
from pycheeger import compute_cheeger


std = 0.2
coeffs = 0.5 * np.array([1.0, 1.0, -1.1, -0.9])
means = np.array([0.2, 0.1]) + np.array([[-0.1, -0.3], [0.0, 0.4], [0.2, 0.0], [-0.65, -0.5]])


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
                                                     max_tri_area_fm=1e-3, max_iter_fm=20000, plot_results_fm=True,
                                                     num_boundary_vertices_ld=75, max_tri_area_ld=1e-2,
                                                     step_size_ld=1e-4, max_iter_ld=2000, convergence_tol_ld=1e-3,
                                                     num_iter_resampling_ld=10, plot_results_ld=True)

plt.plot(obj_tab)
plt.show()

plt.plot(grad_norm_tab)
plt.show()

print(grad_norm_tab[-1])

end = time.time()

print(end - start)

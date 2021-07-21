import time
import numpy as np
import matplotlib.pyplot as plt

from pycheeger import compute_cheeger, GaussianPolynomial


std = 0.2
coeffs = 0.5 * np.array([1.0, 1.0, -1.1, -0.9])
means = np.array([0.2, 0.1]) + np.array([[-0.1, -0.3], [0.0, 0.4], [0.2, 0.0], [-0.65, -0.5]])


eta = GaussianPolynomial(means, coeffs, std)


start = time.time()

simple_set, obj_tab, grad_norm_tab = compute_cheeger(eta,
                                                     grid_size_fm=80, max_iter_fm=5000, plot_results_fm=True,
                                                     num_boundary_vertices_ld=75, max_tri_area_ld=1e-2,
                                                     step_size_ld=1e-4, max_iter_ld=1000, convergence_tol_ld=1e-3,
                                                     num_iter_resampling_ld=10, plot_results_ld=True)

plt.plot(obj_tab)
plt.show()

plt.plot(grad_norm_tab)
plt.show()

print(grad_norm_tab[-1])

end = time.time()

print(end - start)

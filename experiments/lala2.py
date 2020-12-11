import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import quadpy

from math import exp
from numba import jit, prange

std = 0.2


@jit(nopython=True, parallel=True)
def eta_aux(x, res):
    for i in prange(x.shape[0]):
        squared_norm = x[i, 0] ** 2 + x[i, 1] ** 2
        res[i] += exp(-squared_norm / (2 * std**2))


def eta(x):
    if x.ndim == 1:
        tmp = np.zeros(1)
        eta_aux(np.reshape(x, (1, 2)), tmp)
        res = tmp[0]
    else:
        res = np.zeros(x.shape[0])
        eta_aux(x, res)
    return res


alpha_tab = np.linspace(0.1, 5)
beta_tab = np.linspace(0.1, 5)

alpha_grid, beta_grid = np.meshgrid(alpha_tab, beta_tab)

w_da_grid = np.zeros_like(alpha_grid)
w_dc_grid = np.zeros_like(alpha_grid)
w_ad_grid = np.zeros_like(alpha_grid)
w_cd_grid = np.zeros_like(alpha_grid)

err_grid = np.zeros_like(alpha_grid)


def compute_angle(a, b, c):
    inner = np.inner(b-a, c-b)
    cross = np.cross(b-a, c-b)
    if inner == 0:
        return np.sign(cross) * np.pi / 2
    else:
        return np.arctan(cross / inner)


max_err = 0

for i in range(alpha_grid.shape[0]):
    for j in range(beta_grid.shape[1]):
        alpha = alpha_grid[i, j]
        beta = beta_grid[i, j]

        a = np.array([alpha, 0.0])
        b = np.array([0.0, beta])
        c = np.array([-alpha, 0.0])
        d = np.array([0.0, -beta])

        w_ab, err_w_ab = quadpy.quad(lambda t: eta(np.outer(1 - t, a) + np.outer(t, b)) * (1 - t), 0, 1)
        w_ba, err_w_ba = quadpy.quad(lambda t: eta(np.outer(1 - t, b) + np.outer(t, a)) * (1 - t), 0, 1)
        w_cb, err_w_cb = quadpy.quad(lambda t: eta(np.outer(1 - t, c) + np.outer(t, b)) * (1 - t), 0, 1)
        w_da, err_w_da = quadpy.quad(lambda t: eta(np.outer(1 - t, d) + np.outer(t, a)) * (1 - t), 0, 1)

        max_err = max(max_err, err_w_ab, err_w_ba, err_w_cb, err_w_da)

        theta_a = compute_angle(d, a, b)
        theta_b = compute_angle(a, b, c)
        theta_c = compute_angle(b, c, d)
        theta_d = compute_angle(c, d, a)

        qa = w_ab / np.tan(theta_a / 2)
        qb = w_ba / np.tan(theta_b / 2)
        qc = w_cb / np.tan(theta_c / 2)
        qd = w_da / np.tan(theta_c / 2)

        err_grid[i, j] = np.abs(qa - qb) / np.abs(qa) + np.abs(qc - qb) / np.abs(qb) + np.abs(qd - qc) / np.abs(qc) + np.abs(qa - qd) / np.abs(qd)

print(max_err)

fig, ax = plt.subplots()

err_grid = err_grid + 1e-10

im = ax.pcolormesh(alpha_grid, beta_grid, err_grid,
                   norm=colors.LogNorm(vmin=np.min(err_grid), vmax=np.max(err_grid)), cmap='PuBu_r')

plt.colorbar(im, ax=ax, extend='max')
plt.show()

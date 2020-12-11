import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import quadpy

from math import exp, pi, cos, sin
from numba import jit, prange

std = 0.5


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


a = np.array([0.25, 0.0])
b = np.array([0, 0.5])
c = np.array([-0.25, 0.0])

w_ab, err_w_ab = quadpy.quad(lambda t: eta(np.outer(1-t, a) + np.outer(t, b)) * (1-t), 0, 1)
w_ba, err_w_ba = quadpy.quad(lambda t: eta(np.outer(1-t, b) + np.outer(t, a)) * (1-t), 0, 1)
w_ac, err_w_ac = quadpy.quad(lambda t: eta(np.outer(1-t, a) + np.outer(t, c)) * (1-t), 0, 1)
w_ca, err_w_ca = quadpy.quad(lambda t: eta(np.outer(1-t, c) + np.outer(t, a)) * (1-t), 0, 1)
w_bc, err_w_bc = quadpy.quad(lambda t: eta(np.outer(1-t, b) + np.outer(t, c)) * (1-t), 0, 1)
w_cb, err_w_cb = quadpy.quad(lambda t: eta(np.outer(1-t, c) + np.outer(t, b)) * (1-t), 0, 1)

x_tab = np.linspace(-2, 2, 100)
y_tab = np.linspace(-2, 2, 100)

x_grid, y_grid = np.meshgrid(x_tab, y_tab)

max_err = 0

w_da_grid = np.zeros_like(x_grid)
w_dc_grid = np.zeros_like(x_grid)
w_ad_grid = np.zeros_like(x_grid)
w_cd_grid = np.zeros_like(x_grid)

for i in range(x_grid.shape[0]):
    for j in range(x_grid.shape[1]):
        x = x_grid[i, j]
        y = y_grid[i, j]

        d = np.array([x, y])

        w_da, err_w_da = quadpy.quad(lambda t: eta(np.outer(1-t, d) + np.outer(t, a)) * (1-t), 0, 1)
        max_err = max(max_err, err_w_da)
        w_da_grid[i, j] = w_da

        w_dc, err_w_dc = quadpy.quad(lambda t: eta(np.outer(1 - t, d) + np.outer(t, c)) * (1 - t), 0, 1)
        max_err = max(max_err, err_w_dc)
        w_dc_grid[i, j] = w_dc

        w_ad, err_w_ad = quadpy.quad(lambda t: eta(np.outer(1 - t, a) + np.outer(t, d)) * (1 - t), 0, 1)
        max_err = max(max_err, err_w_ad)
        w_ad_grid[i, j] = w_ad

        w_cd, err_w_cd = quadpy.quad(lambda t: eta(np.outer(1 - t, c) + np.outer(t, d)) * (1 - t), 0, 1)
        max_err = max(max_err, err_w_cd)
        w_cd_grid[i, j] = w_cd

print("largest error: {}".format(max_err))

err_grid_a = np.abs(w_ad_grid - w_ab)
err_grid_c = np.abs(w_cd_grid - w_cb)
err_grid_d = np.abs(w_da_grid - w_dc_grid)

err_grid_tot = err_grid_a + err_grid_c + err_grid_d

err_grids = {'a': err_grid_a, 'c': err_grid_c, 'd': err_grid_d, 'tot': err_grid_tot}

fig, axs = plt.subplots(nrows=4, figsize=(7, 20))

labels = ["a", "c", "d", "tot"]

(i, j) = np.unravel_index(np.argmin(err_grid_tot, axis=None), err_grid_tot.shape)
d = np.array([x_grid[i, j], y_grid[i, j]])
# c = np.array([x_grid[140, 139], y_grid[140, 139]])
d = np.array([0, -0.5])

for k in range(4):
    label = labels[k]
    err_grid = err_grids[label]
    im = axs[k].pcolormesh(x_grid, y_grid, err_grid,
                           norm=colors.LogNorm(vmin=np.min(err_grid), vmax=np.max(err_grid)), cmap='PuBu_r')

    axs[k].scatter(a[0], a[1], color='black', alpha=0.7, marker='+')
    axs[k].scatter(b[0], b[1], color='black', alpha=0.7, marker='+')
    axs[k].scatter(c[0], c[1], color='black', alpha=0.7, marker='+')
    axs[k].scatter(d[0], d[1], color='red', alpha=0.7, marker='+')
    axs[k].scatter(0, 0, color='white', alpha=0.7, marker='+')
    axs[k].set_title("constraint on vertex " + label)

    plt.colorbar(im, ax=axs[k], extend='max')

plt.show()

theta_a = np.arctan(np.cross(a-d, b-a) / np.inner(a-d, b-a))
theta_b = np.arctan(np.cross(b-a, c-b) / np.inner(b-a, c-b))
theta_c = np.arctan(np.cross(c-b, d-c) / np.inner(c-b, d-c))
theta_d = np.arctan(np.cross(d-c, a-d) / np.inner(d-c, a-d))

qa = w_ab / np.tan(theta_a / 2)
qb = w_ba / np.tan(theta_b / 2)
qc = w_cb / np.tan(theta_c / 2)
qd1 = w_da_grid[i, j] / np.tan(theta_c / 2)
qd2 = w_dc_grid[i, j] / np.tan(theta_c / 2)

print(qa)
print(qb)
print(qc)
print(qd1)
print(qd2)

l1 = np.linalg.norm(a - b)
l2 = np.linalg.norm(b - c)
l3 = np.linalg.norm(d - c)
l4 = np.linalg.norm(d - a)

perim = l1 + l2 + l3 + l4

scheme = quadpy.t2.get_good_scheme(12)
area = np.sum(scheme.weights * eta(scheme.points.T.dot(np.stack([a, b, c], axis=0)))) + \
       np.sum(scheme.weights * eta(scheme.points.T.dot(np.stack([c, d, a], axis=0))))

print(area / perim)


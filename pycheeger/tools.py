import numpy as np
import quadpy

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from numba import jit
from pymesh import triangle


def triangulate(vertices, max_area=0.005):
    tri = triangle()
    tri.points = vertices

    tri.max_area = max_area

    tri.split_boundary = True
    tri.verbosity = 0
    tri.run()

    raw_mesh = tri.mesh

    return raw_mesh


@jit(nopython=True)
def find_threshold(x):
    y = np.sort(np.abs(x))[::-1]
    j = len(y)
    stop = False

    partial_sum = np.sum(y)

    while j >= 1 and not stop:
        j = j - 1
        stop = (y[j] - (partial_sum - 1) / (j + 1) > 0)

        if not stop:
            partial_sum -= y[j]

    res = (partial_sum - 1) / (j + 1)

    return res


def proj_one_unit_ball(x):
    if np.sum(np.abs(x)) > 1:
        thresh = find_threshold(np.abs(x))
        res = np.where(np.abs(x) > thresh, (1 - thresh / np.abs(x)) * x, 0)
    else:
        res = x

    return res


def proj_unit_square(x):
    res = x.copy()
    res[np.where(res > 1)] = 1
    res[np.where(res < -1)] = -1

    return res


def prox_inf_norm(x, tau):
    return x - tau * proj_one_unit_ball(x / tau)


def prox_dot_prod(x, tau, eta):
    return x - tau * eta


def integrate_on_triangle(eta, vertices):
    # TODO: add scheme options
    scheme = quadpy.triangle.xiao_gimbutas_09()
    val = scheme.integrate(eta, vertices)

    return val


def postprocess_indicator(x, grad_mat):
    res = np.zeros_like(x)
    _, bins = np.histogram(x, bins=2)
    i1 = np.where(x < bins[1])
    i2 = np.where(x > bins[1])
    mean1 = np.mean(x[i1])
    mean2 = np.mean(x[i2])

    if abs(mean1) < abs(mean2):
        res[i1] = 0
        res[i2] = mean2
    else:
        res[i2] = 0
        res[i1] = mean1

    return res / np.linalg.norm(grad_mat.dot(res), ord=1)


def run_primal_dual(mesh, eta, max_iter, grad_mat_norm, verbose=True):
    sigma = 0.99 / grad_mat_norm
    tau = 0.99 / grad_mat_norm

    phi = np.zeros(mesh.num_edges)
    u = np.zeros(mesh.num_faces)
    former_u = u

    track_u = []
    track_phi = []

    for _ in range(max_iter):
        former_phi = phi
        phi = prox_inf_norm(phi + sigma * mesh.grad_mat.dot(2 * u - former_u), sigma)

        track_phi.append(np.linalg.norm(phi - former_phi))

        former_u = u
        u = prox_dot_prod(u - tau * mesh.grad_mat.T.dot(phi), tau, eta)

        track_u.append(np.linalg.norm(u - former_u))

    if verbose:
        print(np.linalg.norm(u - former_u) / np.linalg.norm(u))
        print(np.linalg.norm(mesh.grad_mat.dot(u), ord=1))

    return postprocess_indicator(u, mesh.grad_mat)


def plot_set_boundary(simple_set, eta):
    x = np.arange(-1.0, 1.0, 0.01)
    y = np.arange(-1.0, 1.0, 0.01)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.zeros_like(x_grid)

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            z_grid[i, j] = eta(np.array([x_grid[i, j], y_grid[i, j]]))

    x_curve = np.append(simple_set.boundary_vertices[:, 0], simple_set.boundary_vertices[0, 0])
    y_curve = np.append(simple_set.boundary_vertices[:, 1], simple_set.boundary_vertices[0, 1])

    fig, ax = plt.subplots()

    v_abs_max = np.max(np.abs(z_grid))

    im = ax.contourf(x_grid, y_grid, z_grid, levels=30, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
    ax.plot(x_curve, y_curve, color='black')

    fig.colorbar(im, ax=ax)
    ax.axis('equal')

    plt.show()


def plot_results(mesh, u, eta_bar):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 14))

    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces)

    eta_avg = eta_bar / np.array([mesh.get_face_area(face_index) for face_index in range(mesh.num_faces)])

    v_abs_max = max(np.max(np.abs(u)), np.max(np.abs(eta_avg)))

    axs[0].triplot(triangulation, color='black', alpha=0.1)
    axs[0].axis('equal')
    im = axs[0].tripcolor(triangulation, facecolors=eta_avg, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
    fig.colorbar(im, ax=axs[0])

    axs[1].triplot(triangulation, color='black', alpha=0.1)
    axs[1].axis('equal')
    im = axs[1].tripcolor(triangulation, facecolors=u, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
    fig.colorbar(im, ax=axs[1])

    plt.show()

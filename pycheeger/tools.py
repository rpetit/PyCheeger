import numpy as np
import quadpy

from numba import jit, prange
from pymesh import triangle


@jit(nopython=True, parallel=True)
def winding(x, vertices):
    wn = 0

    for i_current in prange(len(vertices)):
        if i_current != len(vertices) - 1:
            i_next = i_current + 1
        else:
            i_next = 0

        det = (vertices[i_next, 0] - vertices[i_current, 0]) * (x[1] - vertices[i_current, 1])
        det -= (vertices[i_next, 1] - vertices[i_current, 1]) * (x[0] - vertices[i_current, 0])

        if det > 0 and vertices[i_current, 1] <= x[1] < vertices[i_next, 1]:
            wn += 1

        if det < 0 and vertices[i_next, 1] < x[1] < vertices[i_current, 1]:
            wn += -1

    return wn


def triangulate(vertices, max_area=0.005):
    tri = triangle()
    tri.points = vertices

    tri.segments = np.array([[i, (i+1) % len(vertices)] for i in range(len(vertices))])

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

    for _ in range(max_iter):
        phi = prox_inf_norm(phi + sigma * mesh.grad_mat.dot(2 * u - former_u), sigma)

        former_u = u
        u = prox_dot_prod(u - tau * mesh.grad_mat.T.dot(phi), tau, eta)

    if verbose:
        print(np.linalg.norm(u - former_u) / np.linalg.norm(u))
        print(np.linalg.norm(mesh.grad_mat.dot(u), ord=1))

    return postprocess_indicator(u, mesh.grad_mat)

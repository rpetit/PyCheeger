import numpy as np
import quadpy
import triangle

import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def winding(x, vertices):
    """
    Compute the winding number of a closed polygonal curve described by its vertices around the point x. This number is
    zero if and only if x is outside the polygon.

    Parameters
    ----------
    x : array, shape (2,)
        Point around which the winding number is computed
    vertices : array, shape (N, 2)
        Coordinates of the curve's vertices

    Returns
    -------
    wn : float
        The winding number

    """
    wn = 0

    for i_current in range(len(vertices)):
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


def resample(curve, num_points):
    """
    Resample a closed polygonal chain

    Parameters
    ----------
    curve : array, shape (N, 2)
        Curve to be resampled, described by the list of its vertices. The last vertex should not be equal to the first
        one (the input array should be a minimal description of the closed polygonal chain)
    num_points : int
        Number of vertices of the output curve

    Returns
    -------
    array, shape (num_points, 2)
        Resampled curve

    """
    periodized_curve = np.concatenate([curve, [curve[0]]])

    # computation of the curvilinear absicssa
    curvabs = np.concatenate([[0], np.cumsum(np.linalg.norm(periodized_curve[1:] - periodized_curve[:-1], axis=1))])
    curvabs = curvabs / curvabs[-1]

    # linear interpolation
    new_x = np.interp(np.arange(num_points) / num_points, curvabs, periodized_curve[:, 0])
    new_y = np.interp(np.arange(num_points) / num_points, curvabs, periodized_curve[:, 1])

    return np.stack([new_x, new_y], axis=1)


def triangulate(vertices, max_triangle_area=None, split_boundary=False, plot_result=False):
    """
    Triangulate the interior of a closed polygonal curve using the triangle library (Python wrapper around Shewchuk's
    Triangle mesh generator)

    Parameters
    ----------
    vertices : array, shape (N, 2)
        Coordinates of the curve's vertices
    max_triangle_area : None or float
        Maximum area allowed for triangles, see Shewchuk's Triangle mesh generator, defaut None (no constraint)
    split_boundary : bool
        Whether to allow boundary segments to be splitted or not, defaut False
    plot_result : bool
        If True, the resulting triangulation is shown along with the input, defaut False

    Returns
    -------
    raw_mesh : dict
        Output mesh, see the documentation of the triangle library

    """
    # vertices and segments define a planar straight line graph (vertices are assumed to be given ordered)
    segments = np.array([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])
    triangle_input = dict(vertices=vertices, segments=segments)

    opts = 'qpe'

    if max_triangle_area is not None:
        opts = opts + 'a{}'.format(max_triangle_area)

    if not split_boundary:
        opts = opts + 'Y'

    raw_mesh = triangle.triangulate(triangle_input, opts)

    if plot_result:
        triangle.compare(plt, triangle_input, raw_mesh)
        plt.show()

    return raw_mesh


@jit(nopython=True)
def find_threshold(y):
    """
    Compute the value of the Lagrange multiplier involved in the projection of x into the unit l1 ball

    Parameters
    ----------
    x : array, shape (N,)
        Vector to be projected

    Returns
    -------
    res : float
        Value of the Lagrange multiplier

    Notes
    -----
    Sorting based algorithm. See [1]_ for a detailed explanation of the computations and alternative algorithms.

    References
    ----------
    .. [1] L. Condat, *Fast Projection onto the Simplex and the l1 Ball*, Mathematical Programming,
           Series A, Springer, 2016, 158 (1), pp.575-585.

    """
    # y = np.sort(np.abs(x))[::-1]
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


# @jit(nopython=True)
# def find_threshold_bis(y):
#     v = [y[0]]
#     tilde_v = []
#     rho = y[0] - 1
#     for n in range(1, len(y)):
#         if y[n] > rho:
#             rho += (y[n] - rho) / (len(v) + 1)
#             if rho > y[n] - 1:
#                 v.append(y[n])
#             else:
#                 for x in v:
#                     tilde_v.append(x)
#                 v = [y[n]]
#                 rho = y[n] - 1
#     if len(tilde_v) > 0:
#         for x in tilde_v:
#             if x > rho:
#                 v.append(x)
#                 rho += (x - rho) / len(v)
#     convergence = False
#     while not convergence:
#         i = 0
#         convergence = True
#         while i < len(v):
#             if v[i] > rho:
#                 i += 1
#             else:
#                 rho += (rho - v[i]) / len(v)
#                 del v[i]
#                 i = len(v)
#                 convergence = False
#     return rho


@jit(nopython=True)
def proj_two_one_unit_ball_aux(x, norm_row_x, thresh, res):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if norm_row_x[i, j] > thresh:
                    res[i, j] = (norm_row_x[i, j] - thresh) * x[i, j] / norm_row_x[i, j]


def proj_two_one_unit_ball(x):
    """
    Projection onto the (2,1) unit ball

    Parameters
    ----------
    x : array, shape (N, M, 2)
        Should be seen as a (N*M, 2) matrix to be projected

    Returns
    -------
    res : array, shape (N, M, 2)
        Projection

    Notes
    -----
    See [1]_ for a detailed explanation of the computations and alternative algorithms.

    References
    ----------
    .. [1] L. Condat, *Fast Projection onto the Simplex and the l1 Ball*, Mathematical Programming,
           Series A, Springer, 2016, 158 (1), pp.575-585.

    """
    norm_row_x = np.linalg.norm(x, axis=-1)
    norm_x = np.sum(norm_row_x)

    if norm_x > 1:
        res = np.zeros_like(x)
        y = np.sort(norm_row_x.ravel())[::-1]
        thresh = find_threshold(y)
        proj_two_one_unit_ball_aux(x, norm_row_x, thresh, res)
    else:
        res = x

    return res


# def proj_two_one_unit_ball_bis(x):
#     norm_row_x = np.linalg.norm(x, axis=-1)
#     norm_x = np.sum(norm_row_x)
#
#     if norm_x > 1:
#         res = np.zeros_like(x)
#         thresh = find_threshold_bis(norm_row_x.ravel())
#         proj_two_one_unit_ball_aux(x, norm_row_x, thresh, res)
#     else:
#         res = x
#
#     return res


def prox_two_inf_norm(x, tau):
    """
    Proximal map of the (2, infinity) norm

    Parameters
    ----------
    x : array, shape (N,)
    tau : float


    Returns
    -------
    array, shape (N,)

    Notes
    -----
    .. math:: prox_{\\tau \\, ||.||_{2,\\infty}}(x) = x - \\tau ~ \\text{proj}_{\\{||.||_{2,1}\\leq 1\\}}(x / \\tau)

    """
    return x - tau * proj_two_one_unit_ball(x / tau)


def prox_dot_prod(x, tau, a):
    """
    Proximal map of the inner product between x and a

    Parameters
    ----------
    x : array, shape (N,)
    tau : float
    a : array, shape (N,)

    Returns
    -------
    array, shape (N,)

    """
    return x - tau * a


# TODO: deal with the case where the solution is a sum of two indicators of disjoint simple sets
def postprocess_indicator(x):
    """
    Post process a piecewise constant function on a mesh to get an indicator function of a union of cells

    Parameters
    ----------
    x : array, shape (N + 2, N + 2)
        Values describing the piecewise constant function to be processed

    Returns
    -------
    array, shape (N, N)
        Values of the indicator function on each pixel of the grid

    """
    res = np.zeros_like(x)

    # the values of x should concentrate around two values
    _, bins = np.histogram(x, bins=2)

    # find the indices where x is clusters around each of the two values
    i1 = np.where(x < bins[1])
    i2 = np.where(x > bins[1])

    # mean of the values in each cluster
    mean1 = np.mean(x[i1])
    mean2 = np.mean(x[i2])

    # the smallest of the means (in absolute value) is shrinked to zero
    if abs(mean1) < abs(mean2):
        res[i1] = 0
        res[i2] = mean2
    else:
        res[i2] = 0
        res[i1] = mean1

    # the output indicator function is normalized to have unit total variation
    res /= np.sum(np.linalg.norm(grad(res), axis=-1))
    return res


@jit(nopython=True)
def update_grad(u, res):
    n = u.shape[0] - 2

    for i in range(n + 1):
        for j in range(n + 1):
            res[i, j, 0] = u[i + 1, j] - u[i, j]
            res[i, j, 1] = u[i, j + 1] - u[i, j]


def grad(u):
    n = u.shape[0] - 2
    res = np.zeros((n + 1, n + 1, 2))
    update_grad(u, res)
    return res


@jit(nopython=True)
def update_adj_grad(phi, res):
    n = phi.shape[0] - 1

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            res[i, j] = -(phi[i, j, 0] + phi[i, j, 1] - phi[i - 1, j, 0] - phi[i, j - 1, 1])


def adj_grad(phi):
    n = phi.shape[0] - 1
    res = np.zeros((n + 2, n + 2))
    update_adj_grad(phi, res)
    return res


def power_method(grid_size, n_iter=100):
    x = np.random.random((grid_size + 2, grid_size + 2))
    x[0, :] = 0
    x[grid_size + 1, :] = 0
    x[:, 0] = 0
    x[:, grid_size + 1] = 0

    for i in range(n_iter):
        x = adj_grad(grad(x))
        x = x / np.linalg.norm(x)

    return np.sqrt(np.sum(x * (adj_grad(grad(x)))) / np.linalg.norm(x))


def integrate_on_grid(eta, grid_size):
    res = np.zeros((grid_size + 2, grid_size + 2))
    h = 2 / grid_size

    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):
            a, b = -1 + (i - 1) * h, -1 + (j - 1) * h
            res[i, j] = eta.integrate_on_square(a, b, h)

    return res


def run_primal_dual(grid_size, eta_bar, max_iter, verbose=False, plot=False):
    """
    Solves the "fixed mesh weighted Cheeger problem" by running a primal dual algorithm

    Parameters
    ----------
    grid_size
    eta_bar : array, shape (N, 2)
        Integral of the weight function on each triangle
    max_iter : integer
        Maximum number of iterations (for now, exact number of iterations, since no convergence criterion is
        implemented yet)
    verbose : bool, defaut False
        Whether to print some information at the end of the algorithm or not
    plot : bool, defaut False
        Whether to regularly plot the image given by the primal variable or not

    Returns
    -------
    array, shape (N, 2)
        Values describing a piecewise constant function on the mesh, which solves the fixed mesh weighted Cheeger problem

    """
    grad_op_norm = power_method(grid_size)

    grad_buffer = np.zeros((grid_size + 1, grid_size + 1, 2))
    adj_grad_buffer = np.zeros((grid_size + 2, grid_size + 2))

    sigma = 0.99 / grad_op_norm
    tau = 0.99 / grad_op_norm

    phi = np.zeros((grid_size + 1, grid_size + 1, 2))  # dual variable
    u = np.zeros((grid_size + 2, grid_size + 2))  # primal variable
    former_u = u

    eta_bar_pad = np.zeros((grid_size + 2, grid_size + 2))
    eta_bar_pad[1:grid_size+1, 1:grid_size+1] = eta_bar

    for iter in range(max_iter):
        update_grad(2 * u - former_u, grad_buffer)
        phi = prox_two_inf_norm(phi + sigma * grad_buffer, sigma)

        former_u = u
        update_adj_grad(phi, adj_grad_buffer)
        u = prox_dot_prod(u - tau * adj_grad_buffer, tau, eta_bar_pad)

    if verbose:
        print(np.linalg.norm(u - former_u) / np.linalg.norm(u))

    return u


def extract_contour(u):
    v = postprocess_indicator(u)

    n = v.shape[0] - 2
    h = 2 / n
    grad_v = grad(v)
    edges = []

    for i in range(n+1):
        for j in range(n+1):
            if np.abs(grad_v[i, j, 0]) > 0:
                x, y = -1 + i * h, -1 + (j - 1) * h
                edges.append([[x, y], [x, y + h]])
            if np.abs(grad_v[i, j, 1]) > 0:
                x, y = - 1 + (i - 1) * h, -1 + j * h
                edges.append([[x, y], [x + h, y]])

    edges = np.array(edges)

    path_vertices = [edges[0][0], edges[0][1]]

    mask = np.ones(len(edges), dtype=bool)
    mask[0] = False

    done = False

    while not done:
        prev_vertex = path_vertices[-1]
        where_next = np.where(np.isclose(edges[mask], prev_vertex[None, None, :]).all(2))

        if where_next[0].size == 0:
            done = True

        else:
            i, j = where_next[0][0], where_next[1][0]

            next_vertex = edges[mask][i, 1 - j]
            path_vertices.append(next_vertex)

            count = 0
            k = 0
            while count < i + 1:
                if mask[k]:
                    count += 1
                k += 1
            mask[k-1] = False

    return np.array(path_vertices[:-1])

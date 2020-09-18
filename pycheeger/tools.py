import numpy as np
import quadpy

from numba import jit, prange
from pymesh import triangle


@jit(nopython=True, parallel=True)
def winding(x, vertices):
    """
    Compute the winding number of a closed polygonal curve described by its vertices around the point x

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
    """
    Triangulate the interior of a closed polygonal curve

    Parameters
    ----------
    vertices : array, shape (N, 2)
        Coordinates of the curve's vertices
    max_area : float
        Maximum triangle area, see pymesh.triangle

    Returns
    -------
    raw_mesh : pymesh.Mesh
        Output mesh, see pymesh.triangle and pymesh.Mesh

    """
    tri = triangle()

    # points and segments define a planar straight line graph (vertices are assumed to be given ordered)
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


def proj_unit_l1_ball(x):
    """
    Projection onto the l1 unit ball

    Parameters
    ----------
    x : array, shape (N,)
        Vector to be projected

    Returns
    -------
    res : array, shape (N,)
        Projection

    Notes
    -----
    See [1]_ for a detailed explanation of the computations and alternative algorithms.

    References
    ----------
    .. [1] L. Condat, *Fast Projection onto the Simplex and the l1 Ball*, Mathematical Programming,
           Series A, Springer, 2016, 158 (1), pp.575-585.

    """
    if np.sum(np.abs(x)) > 1:
        thresh = find_threshold(np.abs(x))
        res = np.where(np.abs(x) > thresh, (1 - thresh / np.abs(x)) * x, 0)
    else:
        res = x

    return res


def prox_inf_norm(x, tau):
    """
    Proximal map of the l-infinity norm

    Parameters
    ----------
    x : array, shape (N,)
    tau : float


    Returns
    -------
    array, shape (N,)

    Notes
    -----
    .. math:: prox_{\tau ||.||_{\infty}}(x) = x - \tau \text{proj}_{\{||.||_{\infty}\leq 1\}}(x / \tau)

    """
    return x - tau * proj_unit_l1_ball(x / tau)


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


SCHEME = quadpy.t2.get_good_scheme(6)


def integrate_on_triangles(f, triangles):
    """
    Numerical integration of f on a list of triangles

    Parameters
    ----------
    f : function
        Function to be integrated. f must handle array inputs with shape (N, 2). It can be vector valued
    triangles : array, shape (N, 3, 2)
        triangles[i, j] contains the coordinates of the j-th vertex of the i-th triangle

    Returns
    -------
    array, shape (N,) or (N, D)
        Value computed for the integral of f on each of the N triangles (if f takes values in dimension D, the shape of
        the resulting array is (N, D))

    """
    num_triangles = len(triangles)
    num_scheme_points = SCHEME.points.shape[1]

    # vectorized computation of the triangles' edges length
    a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)

    # computation of the areas using Heron's formula
    p = (a + b + c) / 2
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))

    # x contains the list of points at which f should be evaluated (depends on the numerical integration scheme)
    x = np.tensordot(triangles, SCHEME.points, axes=([1], [0]))  # shape (num_triangles, 2, num_scheme_points)
    x = np.moveaxis(x, -1, 0)  # reshape to shape (num_scheme_points, num_triangles, 2)
    x_flat = np.reshape(x, (-1, 2))  # reshape to shape (num_scheme_points * num_triangles, 2)

    # evaluations of f are reshaped to shape (num_scheme_points, num_triangles,) or
    # (num_scheme_points, num_triangles, D) if f is vector valued
    evals_flat = f(x_flat)
    evals = np.reshape(evals_flat, (num_scheme_points, num_triangles) + evals_flat.shape[1:])

    # weighted sum of the evaluations to get the value of the integral on each triangle
    weighted_evals = np.tensordot(SCHEME.weights, evals, axes=([0], [0]))

    # a dimension is added to the array of areas if f is vector valued
    return np.expand_dims(area, tuple(np.arange(1, weighted_evals.ndim))) * weighted_evals


def postprocess_indicator(x, grad_mat):
    """


    Parameters
    ----------
    x
    grad_mat

    Returns
    -------

    """
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

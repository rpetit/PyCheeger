import numpy as np
import quadpy
import triangle

import matplotlib.pyplot as plt

from numba import jit, prange


@jit(nopython=True, parallel=True)
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
    .. math:: prox_{\\tau \\, ||.||_{\\infty}}(x) = x - \\tau ~ \\text{proj}_{\\{||.||_{\\infty}\\leq 1\\}}(x / \\tau)

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

    Notes
    -----
    Here, quadpy is only used to extract the scheme's characteristics, in order to speed up computations (in quadpy,
    the code handles arbitrary dimensions and is too generic)

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


# TODO: deal with the case where the solution is a sum of two indicators of disjoint simple sets
def postprocess_indicator(x, grad_mat):
    """
    Post process a piecewise constant function on a mesh to get an indicator function of a union of cells

    Parameters
    ----------
    x : array, shape (N,)
        Values describing the piecewise constant function to be processed
    grad_mat : array, shape (M, N)
        Matrix representing the linear operator which maps the values describing a piecewise constant function to its
        jumps on each edge of the mesh

    Returns
    -------
    array, shape (N,)
        Values of the indicator function on each cell of the mesh

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
    return res / np.linalg.norm(grad_mat.dot(res), ord=1)


def run_primal_dual(mesh, eta_bar, max_iter, grad_mat_norm, verbose=False):
    """
    Solves the "fixed mesh weighted Cheeger problem" by running a primal dual algorithm

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh made of N triangles and M edges
    eta_bar : array, shape (N, 2)
        Integral of the weight function on each triangle
    max_iter : integer
        Maximum number of iterations (for now, exact number of iterations, since no convergence criterion is
        implemented yet)
    grad_mat_norm : float
        Norm of the gradient operator for piecewise constant functions on the mesh
    verbose : bool, defaut False
        Whether to print some information at the end of the algorithm or not

    Returns
    -------
    array, shape (N, 2)
        Values describing a piecewise constant function on the mesh, which solves the fixed mesh weighted Cheeger problem

    """
    sigma = 0.99 / grad_mat_norm
    tau = 0.99 / grad_mat_norm

    phi = np.zeros(mesh.num_edges)  # dual variable
    u = np.zeros(mesh.num_faces)  # primal variable
    former_u = u

    for _ in range(max_iter):
        phi = prox_inf_norm(phi + sigma * mesh.grad_mat.dot(2 * u - former_u), sigma)

        former_u = u
        u = prox_dot_prod(u - tau * mesh.grad_mat.T.dot(phi), tau, eta_bar)

    if verbose:
        print(np.linalg.norm(u - former_u) / np.linalg.norm(u))
        print(np.linalg.norm(mesh.grad_mat.dot(u), ord=1))

    # the output is post processed so that it is an indicator function and has exactly unit total variation
    return postprocess_indicator(u, mesh.grad_mat)

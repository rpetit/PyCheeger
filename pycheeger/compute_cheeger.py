import numpy as np

from scipy.sparse.linalg import svds

from .mesh import Mesh
from .simple_set import SimpleSet
from .optimizer import CheegerOptimizer
from .tools import triangulate, run_primal_dual, resample
from .plot_utils import plot_primal_dual_results, plot_simple_set


def compute_cheeger(eta, max_tri_area_fm=2e-3, max_iter_fm=10000, plot_results_fm=False,
                    num_boundary_vertices_ld=50, max_tri_area_ld=5e-3, step_size_ld=1e-2, max_iter_ld=500,
                    convergence_tol_ld=1e-4, num_iter_resampling_ld=None, plot_results_ld=False):
    """
    Compute the Cheeger set associated to the weight function eta

    Parameters
    ----------
    eta : function
        Function to be integrated. f must handle array inputs with shape (N, 2)
    max_tri_area_fm : float
        Fixed mesh step parameter. Maximum triangle area allowed for the domain mesh
    max_iter_fm : int
        Fixed mesh step parameter. Maximum number of iterations for the primal dual algorithm
    plot_results_fm : bool
        Fixed mesh step parameter. Whether to plot the results of the fixed mesh step or not
    num_boundary_vertices_ld : int
        Local descent step parameter. Number of boundary vertices used to represent the simple set
    max_tri_area_ld : float
        Local descent step parameter. Maximum triangle area allowed for the inner mesh of the simple set
    step_size_ld : float
        Local descent step parameter. Step size used in the local descent
    max_iter_ld : int
        Local descent step parameter. Maximum number of iterations allowed for the local descent
    convergence_tol_ld : float
        Local descent step parameter. Convergence tol for the local descent
    num_iter_resampling_ld : None or int
        Local descent step parameter. Number of iterations between two resampling of the boundary curve (None for no
        resampling)
    plot_results_ld : bool
        Local descent step parameter. Whether to plot the results of the local descent step or not

    Returns
    -------
    simplet_set : SimpleSet
        Cheeger set
    obj_tab : array, shape (n_iter_ld,)
        Values of the objective over the course of the local descent
    grad_norm_tab : array, shape (n_iter_ld,)
        Values of the objective gradient norm over the course of the local descent

    """
    # triangulation of the domain (for now, always the "unit square")
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    raw_mesh = triangulate(vertices, max_triangle_area=max_tri_area_fm, split_boundary=True)
    mesh = Mesh(raw_mesh)

    # compute the integral of the weight function over each triangle
    eta_bar = mesh.integrate(eta)

    # build the gradient matrix and compute its norm
    mesh.build_grad_matrix()
    grad_mat_norm = svds(mesh.grad_mat, k=1, return_singular_vectors=False)

    # perform the fixed mesh optimization step
    u = run_primal_dual(mesh, eta_bar, max_iter_fm, grad_mat_norm)

    if plot_results_fm:
        plot_primal_dual_results(mesh, u, eta_bar)

    # extraction of the boundary vertices from u (which is the indicator function of a union of faces)
    boundary_vertices_index, boundary_edges_index = mesh.find_path(np.where(np.abs(mesh.grad_mat.dot(u)) > 0)[0])
    boundary_vertices = mesh.vertices[boundary_vertices_index]

    # initial set for the local descent
    boundary_vertices = resample(boundary_vertices, num_boundary_vertices_ld)
    simple_set = SimpleSet(boundary_vertices, max_tri_area_ld)

    # perform the local descent step
    optimizer = CheegerOptimizer(step_size_ld, max_iter_ld, convergence_tol_ld, num_boundary_vertices_ld,
                                 max_tri_area_ld, num_iter_resampling_ld, 0.1, 0.5)
    cheeger_set, obj_tab, grad_norm_tab = optimizer.run(eta, simple_set)

    if plot_results_ld:
        plot_simple_set(cheeger_set, eta=eta, display_inner_mesh=False)

    return cheeger_set, obj_tab, grad_norm_tab

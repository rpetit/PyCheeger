import numpy as np

from scipy.sparse.linalg import svds

from .mesh import CustomMesh
from .simple_set import SimpleSet
from .tools import triangulate, run_primal_dual
from .plot_utils import plot_primal_dual_results, plot_simple_set


def compute_cheeger(eta, max_tri_area=0.002, max_primal_dual_iter=10000,
                    step_size=1e-2, max_iter=500, convergence_tol=1e-4, plot_results=False):

    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    raw_mesh = triangulate(vertices, max_area=max_tri_area)
    mesh = CustomMesh(raw_mesh.vertices, raw_mesh.faces)

    eta_bar = mesh.integrate(eta)

    mesh.build_grad_matrix()
    grad_mat_norm = svds(mesh.grad_mat, k=1, return_singular_vectors=False)

    u = run_primal_dual(mesh, eta_bar, max_primal_dual_iter, grad_mat_norm)

    if plot_results:
        plot_primal_dual_results(mesh, u, eta_bar)

    boundary_vertices_index, boundary_edges_index = mesh.find_path(np.where(np.abs(mesh.grad_mat.dot(u)) > 0)[0])
    boundary_vertices = mesh.vertices[boundary_vertices_index][::2]
    simple_set = SimpleSet(boundary_vertices)

    obj_tab, grad_norm_tab = simple_set.perform_gradient_descent(eta, step_size, max_iter, convergence_tol)

    if plot_results:
        plot_simple_set(simple_set, eta, display_inner_mesh=True)

    return simple_set, obj_tab, grad_norm_tab

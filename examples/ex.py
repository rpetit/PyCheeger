import numpy as np
import pymesh

from pycheeger import *

vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
raw_mesh = triangulate(vertices, max_area=0.0025)
mesh = CustomMesh(raw_mesh.vertices, raw_mesh.faces)

std = 0.2
coeffs = 0.5 * np.array([0.9, 1.0, -1.1, -0.7])
means = np.array([[-0.1, -0.3], [0.0, 0.4], [0.1, 0.0], [-0.6, -0.5]])


def eta(x):
    if x.ndim == 1:
        res = coeffs[0] * np.exp(-np.linalg.norm(x - means[0]) ** 2 / (2 * std ** 2))

        for i in range(1, len(coeffs)):
            res += coeffs[i] * np.exp(-np.linalg.norm(x - means[i]) ** 2 / (2 * std ** 2))
    else:
        res = coeffs[0] * np.exp(-np.linalg.norm(x - means[0, :, np.newaxis], axis=0) ** 2 / (2 * std**2))

        for i in range(1, len(coeffs)):
            res += coeffs[i] * np.exp(-np.linalg.norm(x - means[i, :, np.newaxis], axis=0) ** 2 / (2 * std**2))

    return res


eta_bar = mesh.integrate(eta)

mesh.build_grad_matrix()
grad_mat_norm = np.linalg.norm(mesh.grad_mat.toarray(), ord=2)

max_iter = 10000

u = run_primal_dual(mesh, eta_bar, max_iter, grad_mat_norm)
plot_results(mesh, u, eta_bar)

boundary_vertices_index, _ = mesh.find_path(np.where(np.abs(mesh.grad_mat.dot(u)) > 0)[0])
boundary_vertices = mesh.vertices[boundary_vertices_index]
mesh = pymesh.submesh(mesh.raw_mesh, np.where(np.abs(u) > 0)[0], 0)

simple_set = SimpleSet(boundary_vertices, mesh)

step_size = 1e-5
n_iter = 200
simple_set.perform_gradient_descent(eta, step_size, n_iter)

plot_set_boundary(simple_set, eta)

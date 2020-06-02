import numpy as np
import pymesh

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from pycheeger import *

vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

tri = pymesh.triangle()
tri.points = vertices
tri.max_area = 0.1

tri.split_boundary = True
tri.verbosity = 0
tri.run()

mesh = tri.mesh

div_mat, edges = build_divergence_matrix(mesh)

eval_mat1, eval_mat2 = build_eval_mat(mesh, div_mat, edges)
adjoint_eval_mat1, adjoint_eval_mat2 = build_adjoint_eval_mat(mesh, div_mat, edges)

# f = np.random.rand(len(edges))
# z = np.random.random((2, 3 * mesh.num_faces))
#
# print(np.dot(eval_mat1.dot(f), z[0, :]) + np.dot(eval_mat2.dot(f), z[1, :]))
# print(np.dot(f, adjoint_eval_mat1.dot(z[0, :]) + adjoint_eval_mat2.dot(z[1, :])))

psi = lambda t: np.array([(t[0] - 0.5)**3 / 6 - 0.5 * t[0], (t[1]-0.5)**3/6])

max_iter = 100
sigma = 0.1
tau = 0.1
theta = 0.1

phi = np.zeros(len(edges))
phi_bar = phi
z = np.zeros((2, 3 * mesh.num_faces))

i = 0
convergence = False

while i < max_iter and not convergence:
    z = prox_two_inf_norm(z + sigma * np.stack([eval_mat1.dot(phi_bar), eval_mat2.dot(phi_bar)]))

    former_phi = phi
    phi = project_div_constraint(phi - tau * (adjoint_eval_mat1.dot(z[0, :]) + adjoint_eval_mat2.dot(z[1, :])),
                                 project_piecewise_constant(psi, mesh), div_mat)

    phi_bar = phi + theta * (phi - former_phi)

    convergence = False
    i += 1

print(phi)

lala = np.stack([eval_mat1.dot(phi), eval_mat2.dot(phi)])
print(np.linalg.norm(lala, axis=0))

triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces)
plt.triplot(triangulation, color='black')

x, y = np.meshgrid(np.linspace(0, 1, 15), np.linspace(0, 1, 15))
u, v = np.zeros_like(x), np.zeros_like(x)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        vec = eval_vector_field(np.array([x[i, j], y[i, j]]), mesh, div_mat, edges, phi)
        u[i, j] = vec[0]
        v[i, j] = vec[1]

plt.quiver(x, y, u, v, color='red')

plt.axis('equal')
plt.show()

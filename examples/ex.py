import numpy as np
import pymesh

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from pycheeger import *

vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

tri = pymesh.triangle()
tri.points = vertices
tri.max_area = 0.5
tri.split_boundary = True
tri.verbosity = 0
tri.run()

mesh = tri.mesh

# triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces)
# plt.triplot(triangulation, color='black')
# plt.axis('equal')
# plt.show()

div_mat, edges = build_divergence_matrix(mesh)

eval_mat1, eval_mat2 = build_eval_mat(mesh, div_mat, edges)
adjoint_eval_mat1, adjoint_eval_mat2 = build_adjoint_eval_mat(mesh, div_mat, edges)

# f = np.random.rand(len(edges))
# z = np.random.random((2, 3 * mesh.num_faces))
#
# print(np.dot(eval_mat1.dot(f), z[0, :]) + np.dot(eval_mat2.dot(f), z[1, :]))
# print(np.dot(f, adjoint_eval_mat1.dot(z[0, :]) + adjoint_eval_mat2.dot(z[1, :])))

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
    phi = project_div_constraint(phi - tau * (adjoint_eval_mat1.dot(z[0, :]) + adjoint_eval_mat2.dot(z[1, :])), np.array([1, -1]), div_mat)

    phi_bar = phi + theta * (phi - former_phi)

    convergence = False
    i += 1

print(phi)

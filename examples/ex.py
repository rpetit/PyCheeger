import pymesh

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from pycheeger import *

vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

tri = pymesh.triangle()
tri.points = vertices
tri.max_area = 0.01

tri.split_boundary = True
tri.verbosity = 0
tri.run()

raw_mesh = tri.mesh
mesh = CustomMesh(raw_mesh.vertices, raw_mesh.faces)

grad_mat = build_grad_matrix(mesh)
adjoint_grad_mat = grad_mat.transpose()

psi = lambda t: np.array([(t[0] - 0.5)**3 / 6 - 0.5 * t[0], (t[1]-0.5)**3/6])
eta = project_piecewise_constant(psi, mesh)

max_iter = 300
sigma = 0.05
tau = 0.05
theta = 0.1

phi = np.zeros(mesh.num_edges)
u = np.zeros(mesh.num_faces)
u_bar = u

i = 0
convergence = False

while i < max_iter and not convergence:
    phi = prox_inf_norm(phi + sigma * grad_mat.dot(u_bar), sigma)

    former_u = u
    u = prox_dot_prod(u - tau * adjoint_grad_mat.dot(phi), tau, eta)

    u_bar = u + theta * (u - former_u)

    convergence = False
    i += 1

print(np.linalg.norm(u - former_u))

triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces)
plt.tripcolor(triangulation, facecolors=u, cmap='Greys')

plt.axis('equal')
plt.show()

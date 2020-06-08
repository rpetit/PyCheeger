import pymesh

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from scipy.special import erf

from pycheeger import *

vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

tri = pymesh.triangle()
tri.points = vertices
tri.max_area = 0.005

tri.split_boundary = True
tri.verbosity = 0
tri.run()

raw_mesh = tri.mesh
mesh = CustomMesh(raw_mesh.vertices, raw_mesh.faces)

grad_mat = build_grad_matrix(mesh)
adjoint_grad_mat = grad_mat.transpose()

grad_mat_norm = np.linalg.norm(grad_mat.toarray(), ord=2)


def custom_erf(x, mean, std):
    return 0.5 * np.sqrt(2 * np.pi) * std * erf((x - mean) / (np.sqrt(2) * std))


std = 0.1


# def psi(t):
#     return 0.5 * np.array([np.exp(-t[0]**2 / (2 * std**2)) * custom_erf(t[1], 0, std),
#                            np.exp(-t[1]**2 / (2 * std**2)) * custom_erf(t[0], 0, std)])


def psi(t):
    return np.array([t[0]**3 / 6 - 0.3 * t[0], t[1]**3 / 6])


eta = project_piecewise_constant(lambda t: psi(t), mesh)

max_iter = 20000
sigma = 0.99 / grad_mat_norm
tau = 0.99 / grad_mat_norm
theta = 1

phi = np.zeros(mesh.num_edges)
u = np.zeros(mesh.num_faces)
former_u = u

track_u = []
track_phi = []

for _ in range(max_iter):
    former_phi = phi
    phi = prox_inf_norm(phi + sigma * grad_mat.dot(2 * u - former_u), sigma)

    track_phi.append(np.linalg.norm(phi - former_phi))

    former_u = u
    u = prox_dot_prod(u - tau * adjoint_grad_mat.dot(phi), tau, eta)

    track_u.append(np.linalg.norm(u - former_u))

fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].plot(track_u)
axs[1].plot(track_phi)

plt.show()

print(np.linalg.norm(u - former_u) / np.linalg.norm(u))
print(np.linalg.norm(grad_mat.dot(u), ord=1))

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 21))

triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces)
axs[0].triplot(triangulation, color='black')
axs[0].axis('off')

v_abs_max = np.max(np.abs(u))
im = axs[1].tripcolor(triangulation, facecolors=eta, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
fig.colorbar(im, ax=axs[1])

v_abs_max = np.max(np.abs(u))
im = axs[2].tripcolor(triangulation, facecolors=u, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
fig.colorbar(im, ax=axs[2])

plt.show()

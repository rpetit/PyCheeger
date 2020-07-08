import pymesh

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

std = 0.25
coeffs = np.array([0.3, 0.4, -0.3])
means = np.array([[0.0, 0.0], [-0.5, -0.2], [0.1, 0.2]])


@jit(nopython=True)
def eta(x):
    res = coeffs[0] * np.exp(-((x[0] - means[0, 0])**2 + (x[1] - means[0, 1])**2) / (2 * std**2))

    for i in range(1, len(coeffs)):
        res += coeffs[i] * np.exp(-((x[0] - means[i, 0])**2 + (x[1] - means[i, 1])**2) / (2 * std**2))

    return res


eta_bar = mesh.integrate(eta)

mesh.build_grad_matrix()
grad_mat_norm = np.linalg.norm(mesh.grad_mat.toarray(), ord=2)

max_iter = 35000

u = run_primal_dual(mesh, eta_bar, max_iter, grad_mat_norm)
plot_results(mesh, u, eta_bar, std)

for i in range(5):
    mesh.refine(u)
    eta_bar = mesh.integrate(eta)

    grad_mat_norm = np.linalg.norm(mesh.grad_mat.toarray(), ord=2)

    u = run_primal_dual(mesh, eta_bar, max_iter, grad_mat_norm)
    plot_results(mesh, u, eta_bar, std)

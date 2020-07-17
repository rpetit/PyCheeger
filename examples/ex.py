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

std = 0.3
# coeffs = np.array([0.9, 1.0, -1.1])
# means = np.array([[-0.1, -0.3], [0.0, 0.6], [0.15, 0.2]])
coeffs = np.array([1.0])
means = np.array([[0.0, 0.0]])


def eta(x):
    res = coeffs[0] * np.exp(-np.linalg.norm(x - means[0, :, np.newaxis], axis=0) ** 2 / (2 * std**2))

    for i in range(1, len(coeffs)):
        res += coeffs[i] * np.exp(-np.linalg.norm(x - means[i, :, np.newaxis], axis=0) ** 2 / (2 * std**2))

    return res


eta_bar = mesh.integrate(eta)

mesh.build_grad_matrix()
grad_mat_norm = np.linalg.norm(mesh.grad_mat.toarray(), ord=2)

max_iter = 30000

u = run_primal_dual(mesh, eta_bar, max_iter, grad_mat_norm)
plot_results(mesh, u, eta_bar, std)

for i in range(20):
    mesh.move_vertices(u, eta, eta_bar)
    u *= 1 / np.linalg.norm(mesh.grad_mat.dot(u), ord=1)
    eta_bar = mesh.integrate(eta)

plot_results(mesh, u, eta_bar, std)

square_vertices = vertices.copy()
square_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

grad_edges = np.where(np.abs(mesh.grad_mat.dot(u)) > 0)[0]
path_vertices, path_edges = mesh.find_path(grad_edges)


def lala_map(x):
    return np.where(np.array(path_vertices) == x)[0][0]


new_vertices = np.vstack([square_vertices, mesh.vertices[path_vertices]])
lala_path_edges = []
for i in range(len(path_edges)):
    j1, j2 = mesh.edges[path_edges[i]]
    lala_path_edges.append([4 + lala_map(j1), 4 + lala_map(j2)])

new_edges = np.vstack([square_edges, np.array(lala_path_edges)])

tri = pymesh.triangle()

tri.points = new_vertices
tri.segments = new_edges
tri.max_area = 0.005

tri.split_boundary = True
tri.verbosity = 0
tri.run()

raw_mesh = tri.mesh
mesh = CustomMesh(raw_mesh.vertices, raw_mesh.faces)

eta_bar = mesh.integrate(eta)

mesh.build_grad_matrix()
grad_mat_norm = np.linalg.norm(mesh.grad_mat.toarray(), ord=2)

max_iter = 30000

u = run_primal_dual(mesh, eta_bar, max_iter, grad_mat_norm)
plot_results(mesh, u, eta_bar, std)

for i in range(20):
    mesh.move_vertices(u, eta, eta_bar)
    u *= 1 / np.linalg.norm(mesh.grad_mat.dot(u), ord=1)
    eta_bar = mesh.integrate(eta)

plot_results(mesh, u, eta_bar, std)


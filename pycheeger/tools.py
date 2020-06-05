import numpy as np

from scipy.sparse import csr_matrix
from scipy.integrate import quad


def find_threshold(x):
    y = np.sort(np.abs(x))[::-1]
    j = len(y)
    stop = False

    while j >= 1 and not stop:
        j = j - 1
        stop = (y[j] - (np.sum(y[:j+1]) - 1) / (j + 1) > 0)

    res = (np.sum(y[:j+1]) - 1) / (j + 1)

    return res


def proj_one_unit_ball(x):
    if np.sum(np.abs(x)) > 1:
        thresh = find_threshold(np.abs(x))
        res = np.zeros_like(x, dtype='float')

        for i in range(x.size):
            if np.abs(x[i]) > thresh:
                res[i] = (1 - thresh / np.abs(x[i])) * x[i]
    else:
        res = x

    return res


def prox_inf_norm(x, tau):
    return x - tau * proj_one_unit_ball(x / tau)


def prox_dot_prod(x, tau, eta):
    return x - tau * eta


def build_grad_matrix(mesh):
    indptr = [0]
    indices = []
    data = []

    for i in range(mesh.num_edges):
        edge = mesh.edges[i]
        edge_length = mesh.get_edge_length(edge)
        edge_adjacent_faces = mesh.get_edge_adjacent_faces(edge)

        indptr.append(indptr[-1] + len(edge_adjacent_faces))

        for face_index in edge_adjacent_faces:
            indices.append(face_index)
            data.append(-mesh.get_orientation(face_index, edge) * edge_length)

    grad_mat = csr_matrix((data, indices, indptr))

    return grad_mat


def project_piecewise_constant(phi, mesh):
    proj = np.zeros(mesh.num_faces)

    for i in range(mesh.num_faces):
        face_edges = mesh.get_face_edges(i)

        for edge in face_edges:
            v1, v2 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
            edge_vector = v2 - v1
            edge_normal = np.array([-edge_vector[1], edge_vector[0]])

            proj[i] += quad(lambda t: np.dot(phi(v1 + t * (v2 - v1)), edge_normal), 0, 1)[0]

    return proj

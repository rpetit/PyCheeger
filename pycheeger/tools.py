import numpy as np

from scipy.sparse import csr_matrix
from scipy.integrate import quad

from numba import jit


@jit(nopython=True)
def find_threshold(x):
    y = np.sort(np.abs(x))[::-1]
    j = len(y)
    stop = False

    partial_sum = np.sum(y)

    while j >= 1 and not stop:
        j = j - 1
        stop = (y[j] - (partial_sum - 1) / (j + 1) > 0)

        if not stop:
            partial_sum -= y[j]

    res = (partial_sum - 1) / (j + 1)

    return res


def proj_one_unit_ball(x):
    if np.sum(np.abs(x)) > 1:
        thresh = find_threshold(np.abs(x))
        res = np.where(np.abs(x) > thresh, (1 - thresh / np.abs(x)) * x, 0)
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
        v1, v2, v3 = mesh.vertices[mesh.faces[i]]

        for edge in [[v1, v2], [v2, v3], [v3, v1]]:
            edge_vector = edge[1] - edge[0]
            edge_center = (edge[0] + edge[1]) / 2
            face_centroid = mesh.get_face_centroid(i)
            edge_normal = np.array([-edge_vector[1], edge_vector[0]])
            edge_normal = np.sign(np.dot(edge_center - face_centroid, edge_normal)) * edge_normal

            proj[i] += quad(lambda t: np.dot(phi(edge[0] + t * (edge[1] - edge[0])), edge_normal), 0, 1)[0]

        proj[i] = proj[i] / mesh.get_face_area(i)

    return proj

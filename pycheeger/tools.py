import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from pymesh import mesh_to_graph


def find_threshold(x):
    y = np.sort(np.abs(x))[::-1]
    j = len(y)
    stop = False

    while j >= 1 and not stop:
        j = j - 1
        stop = (y[j] - (np.sum(y[:j+1]) - 1) / (j + 1) > 0)

    res = (np.sum(y[:j+1]) - 1) / (j + 1)

    return res


def prox_two_inf_norm(x):
    if np.sum(np.linalg.norm(x, axis=1)) > 1:
        thresh = find_threshold(np.linalg.norm(x, axis=1))
        res = np.zeros_like(x, dtype='float')

        for i in range(x.shape[0]):
            if np.linalg.norm(x[i]) > thresh:
                res[i] = (1 - thresh / np.linalg.norm(x[i])) * x[i]
    else:
        res = x

    return res


def find_row_index(mat, row):
    return np.where(np.all(mat == row, axis=1))[0][0]


def build_divergence_matrix(mesh):
    mesh.add_attribute("face_centroid")
    face_centroids = mesh.get_face_attribute("face_centroid")
    _, edges = mesh_to_graph(mesh)

    indptr = [3 * i for i in range(mesh.num_faces + 1)]
    indices = []
    data = []

    for i in range(mesh.num_faces):
        face_vertices = np.sort(mesh.faces[i])
        face_centroid = face_centroids[i]

        for edge in [[face_vertices[0], face_vertices[1]],
                     [face_vertices[1], face_vertices[2]],
                     [face_vertices[0], face_vertices[2]]]:

            edge_index = find_row_index(edges, edge)
            indices.append(edge_index)

            edge_center = 0.5 * (mesh.vertices[edge[0]] + mesh.vertices[edge[1]])
            edge_vector = mesh.vertices[edge[1]] - mesh.vertices[edge[0]]
            edge_normal = np.array([-edge_vector[1], edge_vector[0]])

            if np.dot(edge_normal, edge_center - face_centroid) >= 0:
                data.append(1)
            else:
                data.append(-1)

    div_mat = csr_matrix((data, indices, indptr))

    return div_mat, edges


def project_div_constraint(x, div_mat):


    return 0

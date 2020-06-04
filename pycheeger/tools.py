import warnings
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.integrate import quad

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


def find_row_index(mat, row):
    return np.where(np.all(mat == row, axis=1))[0][0]


def build_grad_matrix(mesh):
    mesh.add_attribute("face_centroid")
    face_centroids = mesh.get_face_attribute("face_centroid")
    _, edges = mesh_to_graph(mesh)

    indptr = [0]
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
                data.append(-np.linalg.norm(edge_vector))
            else:
                data.append(np.linalg.norm(edge_vector))

    grad_mat = csr_matrix((data, indices, indptr))

    return grad_mat


def build_adjoint_grad_mat(mesh):
    mesh.add_attribute("face_centroid")
    face_centroids = mesh.get_face_attribute("face_centroid")
    _, edges = mesh_to_graph(mesh)

    indptr = [0]
    indices = []
    data1 = []
    data2 = []

    for i in range(len(edges)):
        j1, j2 = np.sort(edges[i])
        adjacent_faces_index = np.intersect1d(mesh.get_vertex_adjacent_faces(j1), mesh.get_vertex_adjacent_faces(j2))

        if adjacent_faces_index.size == 1:
            indptr.append(indptr[-1] + 2)
        else:
            indptr.append(indptr[-1] + 4)

        for face_index in adjacent_faces_index:

            j1_index_in_face = np.argwhere(np.sort(mesh.faces[face_index]) == j1)[0, 0]
            j2_index_in_face = np.argwhere(np.sort(mesh.faces[face_index]) == j2)[0, 0]
            j3_index_in_face = np.argwhere(np.logical_and(np.sort(mesh.faces[face_index]) != j1, np.sort(mesh.faces[face_index]) != j2))[0, 0]
            j3 = np.sort(mesh.faces[face_index])[j3_index_in_face]

            indices.append(3 * face_index + j1_index_in_face)
            indices.append(3 * face_index + j2_index_in_face)

            lala = div_mat[face_index, i] * (mesh.vertices[j1] - mesh.vertices[j3]) / (2 * faces_area[face_index, 0])

            data1.append(lala[0])
            data2.append(lala[1])

            lala = div_mat[face_index, i] * (mesh.vertices[j2] - mesh.vertices[j3]) / (2 * faces_area[face_index, 0])

            data1.append(lala[0])
            data2.append(lala[1])

    adjoint_eval_mat1 = csr_matrix((data1, indices, indptr))
    adjoint_eval_mat2 = csr_matrix((data2, indices, indptr))

    return adjoint_eval_mat1, adjoint_eval_mat2


def project_div_constraint(x, b, div_mat):
    z, info = cg(div_mat.dot(div_mat.transpose()), b - div_mat.dot(x))

    if info != 0:
        warnings.warn("problem in conjugate gradient !")

    return x + (div_mat.transpose()).dot(z)


def project_piecewise_constant(phi, mesh):
    mesh.add_attribute("face_centroid")
    faces_centroid = mesh.get_face_attribute("face_centroid")

    proj = np.zeros(mesh.num_faces)

    for i in range(mesh.num_faces):
        face = mesh.faces[i]

        for j in range(3):
            edge = np.array([mesh.vertices[face[j]], mesh.vertices[face[(j+1) % 3]]])
            edge_center = 0.5 * (edge[0] + edge[1])
            normal = np.array([(edge[1] - edge[0])[1], -(edge[1] - edge[0])[0]])
            normal = normal * np.dot(edge_center - faces_centroid[i], normal)
            normal = normal * (1 / np.linalg.norm(normal))

            proj[i] += quad(lambda t: np.dot(phi(edge[0] + t * (edge[1] - edge[0])), normal), 0, 1)[0]

    return proj


def is_left(a, b, x):
    return (b[0] - a[0]) * (x[1] - a[1]) - (x[0] - a[0]) * (b[1] - a[1])


def winding(x, poly):
    wn = 0   # the winding number counter

    # repeat the first vertex at end
    poly = np.vstack([poly, poly[0]])

    # loop through all edges of the polygon
    for i in range(poly.shape[0]-1):     # edge from V[i] to V[i+1]
        if poly[i, 1] <= x[1]:        # start y <= P[1]
            if poly[i+1, 1] > x[1]:     # an upward crossing
                if is_left(poly[i], poly[i+1], x) > 0: # P left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if poly[i+1, 1] <= x[1]:    # a downward crossing
                if is_left(poly[i], poly[i+1], x) < 0: # P right of edge
                    wn -= 1           # have a valid down intersect
    return wn


def eval_vector_field(x, mesh, div_mat, edges, fluxes):
    face_index = -1
    in_face = False

    while face_index < mesh.num_faces - 1 and not in_face:
        face_index += 1
        in_face = (winding(x, mesh.vertices[mesh.faces[face_index]]) != 0)

    mesh.add_attribute("face_area")
    face_area = mesh.get_face_attribute("face_area")[face_index][0]

    face_edges_index = div_mat[face_index].nonzero()[1]
    face = mesh.faces[face_index]

    res = np.zeros(2)

    for j in range(3):
        edge = edges[face_edges_index[j]]
        opposite_vertex = np.argwhere(np.logical_and(face != edge[0], face != edge[1]))[0, 0]

        res = res + div_mat[face_index, face_edges_index[j]] * fluxes[face_edges_index[j]] * (x - mesh.vertices[face[opposite_vertex]])

    return res / (2 * face_area)
import numpy as np

from scipy.sparse import csr_matrix

from pymesh import form_mesh, mesh_to_graph
from .tools import integrate_on_triangle


class CustomMesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

        self.raw_mesh = form_mesh(vertices, faces)

        _, edges = mesh_to_graph(self.raw_mesh)
        self.edges = edges

        self.num_edges = len(edges)
        self.num_faces = len(faces)
        self.num_vertices = len(vertices)

        self.grad_mat = None

    def get_edge_index(self, edge):
        where_edge = np.where(np.all(self.edges == edge, axis=1))[0]

        if len(where_edge) == 0:
            raise ValueError("edge not found")
        elif len(where_edge) > 1:
            raise ValueError("edge found multiple times")
        else:
            return where_edge[0]

    def get_edge_adjacent_faces(self, edge):
        self.raw_mesh.enable_connectivity()

        v1_adjacent_faces = self.raw_mesh.get_vertex_adjacent_faces(edge[0])
        v2_adjacent_faces = self.raw_mesh.get_vertex_adjacent_faces(edge[1])

        edge_adjacent_faces = np.intersect1d(v1_adjacent_faces, v2_adjacent_faces)

        return edge_adjacent_faces

    def get_orientation(self, face_index, edge):
        self.raw_mesh.add_attribute("face_centroid")

        face_centroid = self.raw_mesh.get_face_attribute("face_centroid")[face_index]

        v1, v2 = self.vertices[edge[0]], self.vertices[edge[1]]
        edge_center = (v1 + v2) / 2
        edge_vector = v2 - v1
        edge_normal = np.array([-edge_vector[1], edge_vector[0]])

        if np.dot(edge_center - face_centroid, edge_normal) >= 0:
            return 1
        else:
            return -1

    def get_edge_length(self, edge):
        v1, v2 = self.vertices[edge[0]], self.vertices[edge[1]]
        return np.linalg.norm(v2 - v1)

    def get_face_edges(self, face_index):
        face_vertices = self.faces[face_index]

        return [np.sort([face_vertices[0], face_vertices[1]]),
                np.sort([face_vertices[1], face_vertices[2]]),
                np.sort([face_vertices[0], face_vertices[2]])]

    def get_face_area(self, face_index):
        # TODO: INVESTIGATE NEGATIVE FACE AREA ...
        self.raw_mesh.add_attribute("face_area")
        return np.abs(self.raw_mesh.get_face_attribute("face_area")[face_index][0])

    def get_face_centroid(self, face_index):
        self.raw_mesh.add_attribute("face_centroid")
        return self.raw_mesh.get_face_attribute("face_centroid")[face_index]

    def build_grad_matrix(self):
        indptr = [0]
        indices = []
        data = []

        for i in range(self.num_edges):
            edge = self.edges[i]
            edge_length = self.get_edge_length(edge)
            edge_adjacent_faces = self.get_edge_adjacent_faces(edge)

            indptr.append(indptr[-1] + len(edge_adjacent_faces))

            for face_index in edge_adjacent_faces:
                indices.append(face_index)
                data.append(-self.get_orientation(face_index, edge) * edge_length)

        self.grad_mat = csr_matrix((data, indices, indptr))

    def integrate(self, eta):
        res = np.zeros(self.num_faces)

        for i in range(self.num_faces):
            res[i] = integrate_on_triangle(eta, self.vertices[self.faces[i]])

        return res

    def find_path(self, edges_index):
        edges = self.edges[edges_index]
        path_vertices = [edges[0, 0], edges[0, 1]]
        path_edges = [edges_index[0]]

        mask = np.ones(len(edges), dtype=bool)
        mask[0] = False

        for i in range(len(edges) - 1):
            prev_vertex = path_vertices[-1]
            where_next = np.where(edges[mask] == prev_vertex)
            i, j = where_next[0][0], where_next[1][0]

            next_vertex = edges[mask][i, 1 - j]
            next_edge = edges_index[mask][i]

            path_vertices.append(next_vertex)
            path_edges.append(next_edge)

            mask[np.where(edges_index == next_edge)] = False

        return path_vertices[:-1], path_edges

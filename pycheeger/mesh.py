import numpy as np
import quadpy

from scipy.sparse import csr_matrix, vstack, hstack, eye
from scipy.optimize import linprog

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

    def solve_max_flow(self, eta):
        if self.grad_mat is None:
            self.build_grad_matrix()

        block1 = hstack([self.grad_mat, -eye(self.num_edges)])
        block2 = hstack([-self.grad_mat, -eye(self.num_edges)])
        block3 = csr_matrix(np.concatenate([np.zeros(self.num_faces), np.ones(self.num_edges)]))

        A = vstack([block1, block2, block3]).toarray()
        b = np.concatenate([np.zeros(2 * self.num_edges), [1]])

        c = np.concatenate([eta, np.zeros(self.num_edges)])

        res = linprog(c, A_ub=A, b_ub=b)

        return res

    def refine(self, u):
        count = 0
        faces_to_remove = []
        faces_to_add = []
        vertices_to_add = []

        grad_edges = np.where(np.abs(self.grad_mat.dot(u)) > 0)[0]

        for edge_index in grad_edges:
            edge = self.edges[edge_index]
            adjacent_faces = self.get_edge_adjacent_faces(edge)

            for face_index in adjacent_faces:
                centroid = self.get_face_centroid(face_index)
                i1, i2, i3 = self.faces[face_index]

                vertices_to_add.append(centroid)

                faces_to_remove.append(face_index)

                faces_to_add.append([i1, i2, self.num_vertices + count])
                faces_to_add.append([i1, i3, self.num_vertices + count])
                faces_to_add.append([i2, i3, self.num_vertices + count])

                count += 1

        new_vertices = np.append(self.vertices, vertices_to_add, axis=0)
        mask = np.ones(len(self.faces), dtype=bool)
        mask[faces_to_remove] = False
        new_faces = np.append(self.faces[mask], faces_to_add, axis=0)

        self.__init__(new_vertices, new_faces)
        self.build_grad_matrix()

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

    def move_vertices(self, u, eta, eta_bar):
        grad_edges = np.where(np.abs(self.grad_mat.dot(u)) > 0)[0]
        path_vertices, path_edges = self.find_path(grad_edges)

        scheme = quadpy.line_segment.gauss_patterson(5)
        vals = np.zeros((len(path_edges), 2))
        normals = np.zeros((len(path_edges), 2))

        perim = 0

        for i in range(len(path_edges)):
            edge_index = path_edges[i]

            i1, i2 = self.edges[edge_index]
            v1, v2 = self.vertices[i1], self.vertices[i2]

            vals[i, 0] = scheme.integrate(lambda t: eta(np.outer(v1, 1-t) + np.outer(v2, t)) * t,
                                          [0.0, 1.0])
            vals[i, 1] = scheme.integrate(lambda t: eta(np.outer(v1, 1-t) + np.outer(v2, t)) * (1-t),
                                          [0.0, 1.0])

            # TODO: deal with domain boundaries
            adjacent_faces = self.get_edge_adjacent_faces(self.edges[edge_index])
            assert (u[adjacent_faces[0]] == 0 and u[adjacent_faces[1]] != 0) or (u[adjacent_faces[1]] == 0 and u[adjacent_faces[0]] != 0)

            edge_vector = v2 - v1
            perim += np.linalg.norm(edge_vector)
            edge_normal = np.array([-edge_vector[1], edge_vector[0]])
            c0 = self.get_face_centroid(adjacent_faces[0])
            c1 = self.get_face_centroid(adjacent_faces[1])

            if u[adjacent_faces[0]] == 0:
                edge_normal = edge_normal * np.sign(np.dot(edge_normal, c0 - c1))
            else:
                edge_normal = edge_normal * np.sign(np.dot(edge_normal, c1 - c0))

            normals[i] = edge_normal

        new_vertices = self.vertices.copy()

        for i in range(len(path_vertices)):
            e1 = self.vertices[path_vertices[i-1]] - self.vertices[path_vertices[i]]
            e2 = self.vertices[path_vertices[(i+1)%len(path_vertices)]] - self.vertices[path_vertices[i]]
            deriv_perim = (e1 / np.linalg.norm(e1) + e2 / np.linalg.norm(e2))
            deriv_area = (vals[i-1, 1] * normals[i-1] + vals[i, 0] * normals[i])
            area = np.dot(u, eta_bar)
            deriv = (deriv_perim * area - perim * deriv_area) / area ** 2
            new_vertices[path_vertices[i]] = new_vertices[path_vertices[i]] + 2e-4 * np.sign(area) * deriv

        self.__init__(new_vertices, self.faces)
        self.build_grad_matrix()

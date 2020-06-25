import numpy as np

from pymesh import form_mesh, mesh_to_graph


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
        self.raw_mesh.add_attribute("face_area")

        return self.raw_mesh.get_face_attribute("face_area")[face_index]

    def get_face_centroid(self, face_index):
        self.raw_mesh.add_attribute("face_centroid")

        return self.raw_mesh.get_face_attribute("face_centroid")[face_index]

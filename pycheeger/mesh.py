import numpy as np

from scipy.sparse import csr_matrix
from .tools import integrate_on_triangles


class Mesh:
    """
    Custom mesh class

    Attributes
    ----------
    vertices : array, shape (N, 2)
        Each row contains the two coordinates of a vertex
    faces : array, shape (M, 3)
        Each row contains the three indices of the face's vertices. For convenience reasons, the values in each row are
        sorted, i.e. the face whose vertices have indices 0, 1, 2 will always be stored as [0, 1, 2] (and not [1, 0, 2])
    edges : array, shape (K, 2)
        Each row contains the two indices of the edge's vertices. For convenience reasons, the values in each row are
        sorted, i.e. the edge whose vertices have indices 0, 1 will always be stored as [0, 1] (and not [1, 0])
    num_edges : int
        Equals K
    num_faces : int
        Equals M
    num_vertices : int
        Equals N
    grad_mat : scipy.sparse.csr_matrix
        Linear operator mapping a vector of length M (the values taken by a piecewise constant function on the mesh) to
        a vector of length K (the jumps / values taken by the gradient on each edge of the mesh)

    """
    def __init__(self, raw_mesh):
        self.vertices = raw_mesh['vertices']
        self.faces = np.sort(raw_mesh['triangles'], axis=1)
        self.edges = np.sort(raw_mesh['edges'], axis=1)

        self.num_edges = len(self.edges)
        self.num_faces = len(self.faces)
        self.num_vertices = len(self.vertices)

        # self.grad_mat = self.build_grad_matrix()

    def get_edge_index(self, edge):
        """
        Find the index of an edge given by its two vertices in the array of edges

        Parameters
        ----------
        edge : array, shape (2,)
            Array containing two integers, which are the indices of the edge's vertices. Does not need to be sorted.

        Returns
        -------
        int
            The index of the input edge (raises an error if the edge is not found or found multiple times)

        """
        # each row of self.edges is sorted
        sorted_edge = np.sort(edge)

        where_edge = np.where(np.all(self.edges == sorted_edge, axis=1))[0]

        if len(where_edge) == 0:
            raise ValueError("edge not found")
        elif len(where_edge) > 1:
            raise ValueError("edge found multiple times")
        else:
            return where_edge[0]

    def get_edge_adjacent_faces(self, edge):
        """
        Find the index of all faces to which a given edge belongs

        Parameters
        ----------
        edge : array, shape (2,)
            Array containing two integers, which are the indices of the edge's vertices. Does not need to be sorted.

        Returns
        -------
        array
            One dimensional integer valued array containing the indices of the relevant faces

        """
        v1_adjacent_faces = np.where(np.any(np.isin(self.faces, edge[0]), axis=1))[0]
        v2_adjacent_faces = np.where(np.any(np.isin(self.faces, edge[1]), axis=1))[0]

        edge_adjacent_faces = np.intersect1d(v1_adjacent_faces, v2_adjacent_faces)

        return edge_adjacent_faces

    def get_orientation(self, face_index, edge):
        """
        The orientation of an edge with respect to a face is defined as +1 if the normal is the outward normal and -1
        otherwise. The normal is computed by applying a counter clockwise rotation to the edge vector, which is himself
        given by v2 - v1 where v1 and v2 are the two vertices of the edge, and the index of v1 in self.vertices is
        smaller than the one of v2 (edges are always stored sorted). This definition of the orientation is of course
        arbitrary, but the only thing we need is to keep one that is consistent all along.

        Parameters
        ----------
        face_index : int
            Index of the face in self.faces
        edge : array, shape (2,)
            Array containing two integers, which are the indices of the edge's vertices. Does not need to be sorted.


        Returns
        -------
        int
            +1 or -1 if the input face contains the input edge (see the explanations above), and 0 otherwise

        """
        if face_index not in self.get_edge_adjacent_faces(edge):
            return 0

        face_centroid = np.sum(self.vertices[self.faces[face_index]], axis=0) / 3

        sorted_edge = np.sort(edge)

        v1, v2 = self.vertices[sorted_edge[0]], self.vertices[sorted_edge[1]]
        edge_center = (v1 + v2) / 2
        edge_vector = v2 - v1
        edge_normal = np.array([-edge_vector[1], edge_vector[0]])  # counter clockwise rotation of the edge vector

        # the sign of this inner product indicates whether the normal is the outward or the inward normal
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

        return csr_matrix((data, indices, indptr))

    def integrate(self, eta):
        return integrate_on_triangles(eta, self.vertices[self.faces])

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

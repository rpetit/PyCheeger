import numpy as np
import quadpy

from .tools import winding, triangulate

# scheme for numerical integration on segments
SCHEME = quadpy.c1.gauss_patterson(2)


class SimpleSet:
    """
    Simple set

    Attributes
    ----------
    boundary_vertices : array, shape (N, 2)
        Each row contains the two coordinates of a boundary vertex
    num_boundary_vertices : int
        Number of boundary vertices
    is_clockwise : bool
        Whether the list of boundary vertices is clockwise ordered or not
    mesh_vertices : array, shape (M, 2)
        Each row contains the two coordinates of a vertex of the simplet set inner mesh
    mesh_faces : array, shape (K, 3)
        Each row contains the three indices describing a face of the simple set inner mesh
    mesh_boundary_faces_indices : array
        One dimensional array, contains the indices of the boundary faces of the simple set inner mesh

    """

    def __init__(self, boundary_vertices, max_tri_area=None):
        """
        Constructor

        Parameters
        ----------
        boundary_vertices : array, shape (N, 2)
            See class description
        max_tri_area : None or float
            Maximum area allowed for triangles, see Shewchuk's Triangle mesh generator, defaut None (no constraint)
        """
        self.num_boundary_vertices = len(boundary_vertices)

        # the curve is clockwise if and only if the sum over the edges of (x2-x1)(y2+y1) is positive
        rolled_boundary_vertices = np.roll(boundary_vertices, -1, axis=0)
        self.is_clockwise = (np.sum((rolled_boundary_vertices[:, 0] - boundary_vertices[:, 0]) *
                                    (rolled_boundary_vertices[:, 1] + boundary_vertices[:, 1])) > 0)

        self.mesh_vertices = None
        self.mesh_faces = None
        self.mesh_boundary_faces_indices = None

        # creation of the inner mesh
        self.create_mesh(boundary_vertices, max_tri_area)

    @property
    def boundary_vertices_indices(self):
        return np.arange(self.num_boundary_vertices)

    # TODO: attention du coup self.boundary_vertices est vue comme une fonction, a voir si probleme avec numba
    @property
    def boundary_vertices(self):
        return self.mesh_vertices[self.boundary_vertices_indices]

    @property
    def mesh_boundary_faces(self):
        return self.mesh_faces[self.mesh_boundary_faces_indices]

    @boundary_vertices.setter
    def boundary_vertices(self, new_boundary_vertices):
        self.mesh_vertices[self.boundary_vertices_indices] = new_boundary_vertices

    def contains(self, x):
        """
        Whether a given point x is inside the set or not

        Parameters
        ----------
        x : array, shape (2,)
            The input point

        Returns
        -------
        bool
            Whether x is in the set or not

        """
        # The point is inside the set if and only if its winding number is non zero
        return winding(x, self.boundary_vertices) != 0

    def compute_perimeter(self):
        """
        Compute the perimeter of the set

        Returns
        -------
        float
            The perimeter

        """
        rolled_boundary_vertices = np.roll(self.boundary_vertices, -1, axis=0)
        res = np.sum(np.linalg.norm(rolled_boundary_vertices - self.boundary_vertices, axis=1))
        return res

    def compute_weighted_area_tab(self, f, boundary_faces_only=False):
        """
        Compute the integral of f on each face of the inner mesh

        Parameters
        ----------
        f : function
            Function to be integrated. f must handle array inputs with shape (N, 2). It can be vector valued
        boundary_faces_only : bool
            Whether to compute weighted areas only on boundary faces, defaut False

        Returns
        -------
        array, shape (N,) or (N,D)
            Value computed for the integral of f on each of the N triangles (if f takes values in dimension D, the shape
            of the resulting array is (N, D))

        """
        if boundary_faces_only:
            triangles = self.mesh_vertices[self.mesh_boundary_faces]
        else:
            triangles = self.mesh_vertices[self.mesh_faces]

        return f.integrate_on_triangles(triangles)

    def compute_weighted_area(self, f):
        # TODO: decide whether output type instability should be dealt with or not
        """
        Compute the integral of f over the set

        Parameters
        ----------
        f : function
            Function to be integrated. f must handle array inputs with shape (N, 2). It can be vector valued

        Returns
        -------
        float or array of shape (D,)
            Value computed for the integral of f over the set (if f takes values in dimension D, the result will be an
            array of shape (D,))
        """
        return np.sum(self.compute_weighted_area_tab(f))

    # TODO: doc
    def compute_mesh_faces_orientation(self):
        faces = self.mesh_vertices[self.mesh_faces]
        diff1 = faces[:, 1] - faces[:, 0]
        diff2 = faces[:, 2] - faces[:, 1]
        res = np.sign(np.cross(diff1, diff2)).astype('int')

        return res

    def create_mesh(self, boundary_vertices, max_tri_area):
        """
        Create the inner mesh of the set

        Parameters
        ----------
        boundary_vertices : array, shape (N, 2)
            Each row contains the two coordinates of a boundary vertex
        max_tri_area : float
            Maximum triangle area for the inner mesh

        """
        mesh = triangulate(boundary_vertices, max_triangle_area=max_tri_area)

        self.mesh_vertices = mesh['vertices']
        self.mesh_faces = mesh['triangles']

        # TODO: comment
        orientations = self.compute_mesh_faces_orientation()
        indices = np.where(orientations < 0)[0]
        for i in range(len(indices)):
            index = indices[i]
            tmp_face = self.mesh_faces[index].copy()
            self.mesh_faces[index, 1] = tmp_face[index, 2]
            self.mesh_faces[index, 2] = tmp_face[index, 1]

        assert np.alltrue(orientations > 0)

        boundary_faces_indices = []

        for i in range(len(self.mesh_faces)):
            # find the faces which have at least one vertex among the boundary vertices (the indices of boundary
            # vertices in self.vertices are 0,1,...,self.num_boundary_vertices-1)
            if len(np.intersect1d(np.arange(self.num_boundary_vertices), self.mesh_faces[i])) > 0:
                boundary_faces_indices.append(i)

        self.mesh_boundary_faces_indices = np.array(boundary_faces_indices)

    # TODO: numba
    def compute_perimeter_gradient(self):
        """
        Compute the "gradient" of the perimeter

        Returns
        -------
        array, shape (N, 2)
            Each row contains the two coordinates of the translation to apply at each boundary vertex

        Notes
        -----
        See [1]_ (first variation of the perimeter)

        References
        ----------
        .. [1] Maggi, F. (2012). Sets of finite perimeter and geometric variational problems: an introduction to
               Geometric Measure Theory (No. 135). Cambridge University Press.

        """
        gradient = np.zeros_like(self.boundary_vertices)

        for i in range(self.num_boundary_vertices):
            e1 = self.boundary_vertices[(i-1) % self.num_boundary_vertices] - self.boundary_vertices[i]
            e2 = self.boundary_vertices[(i+1) % self.num_boundary_vertices] - self.boundary_vertices[i]

            # the i-th component of the gradient is -(ti_1 + ti_2) where ti_1 and ti_2 are the two tangent vectors
            # going away from the i-th vertex TODO: clarify
            gradient[i] = - (e1 / np.linalg.norm(e1) + e2 / np.linalg.norm(e2))

        return gradient

    def compute_weighted_area_gradient(self, f):
        """
        Compute the "gradient" of the weighted area, for a given weight function

        Parameters
        ----------
        f : function
            Function to be integrated. f must handle array inputs with shape (N, 2). It can be vector valued

        Returns
        -------
        array, shape

        Notes
        -----
        Vectorized computations are really nasty here, mainly because f can be vector valued.

        """
        # TODO: externaliser calcul normales (et compiler ?)

        # rotation matrix used to compute outward normals
        rot = np.array([[0, -1], [1, 0]]) if self.is_clockwise else np.array([[0, 1], [-1, 0]])

        rolled_vertices1 = np.roll(self.boundary_vertices, 1, axis=0)
        rolled_vertices2 = np.roll(self.boundary_vertices, -1, axis=0)

        weights = f.integrate_on_polygonal_curve(self.boundary_vertices)

        normals1 = np.dot(self.boundary_vertices - rolled_vertices1, rot.T)
        normals2 = np.dot(rolled_vertices2 - self.boundary_vertices, rot.T)

        gradient = weights[:, 0, None] * normals1 + weights[:, 1, None] * normals2

        return gradient


def disk(center, radius, num_vertices=20, max_tri_area=None):
    """
    Create a SimpleSet with boundary vertices regularly spaced on a circle

    Parameters
    ----------
    center : array, shape (2,)
        Coordinates of the center
    radius : float
        Radius of the circles
    num_vertices : int
        Number of boundary vertices (on the disk)
    max_tri_area : None or float
        Maximum area allowed for triangles, see Shewchuk's Triangle mesh generator, defaut None (no constraint)

    Returns
    -------
    SimpleSet

    """
    t = np.linspace(0, 2 * np.pi, num_vertices + 1)[:-1]
    complex_vertices = center[0] + 1j * center[1] + radius * np.exp(1j * t)
    vertices = np.stack([np.real(complex_vertices), np.imag(complex_vertices)], axis=1)

    return SimpleSet(vertices, max_tri_area)

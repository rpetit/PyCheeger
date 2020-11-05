import numpy as np
import quadpy

from .tools import winding, resample, integrate_on_triangles, triangulate


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
    boundary_faces_indices : array
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
        self.boundary_vertices = boundary_vertices
        self.num_boundary_vertices = len(boundary_vertices)

        # the curve is clockwise if and only if the sum over the edges of (x2-x1)(y2+y1) is positive
        rolled_boundary_vertices = np.roll(boundary_vertices, -1, axis=0)
        self.is_clockwise = (np.sum((rolled_boundary_vertices[:, 0] - boundary_vertices[:, 0]) *
                                    (rolled_boundary_vertices[:, 1] + boundary_vertices[:, 1])) > 0)

        self.mesh_vertices = None
        self.mesh_faces = None
        self.boundary_faces_indices = None

        # creation of the inner mesh
        self.mesh(max_tri_area)

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

    def compute_weighted_areas(self, f):
        """
        Compute the integral of f on each face of the inner mesh

        Parameters
        ----------
        f : function
            Function to be integrated. f must handle array inputs with shape (N, 2). It can be vector valued

        Returns
        -------
        array, shape (N,) or (N,D)
            Value computed for the integral of f on each of the N triangles (if f takes values in dimension D, the shape
            of the resulting array is (N, D))

        """
        triangles = self.mesh_vertices[self.mesh_faces]
        return integrate_on_triangles(f, triangles)

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
        return np.sum(self.compute_weighted_areas(f))

    def resample_boundary(self, num_points, max_tri_area):
        """
        Resample the boundary of the simple set, and creates a new inner mesh

        Parameters
        ----------
        num_points : int
            Number of points in the resampled boundary
        max_tri_area : float
            Maximum triangle area for the new inner mesh

        """
        new_boundary_vertices = resample(self.boundary_vertices, num_points)
        self.__init__(new_boundary_vertices, max_tri_area=max_tri_area)

    def mesh(self, max_tri_area):
        """
        Create the inner mesh of the set

        Parameters
        ----------
        max_tri_area : float
            Maximum triangle area for the inner mesh

        """
        mesh = triangulate(self.boundary_vertices, max_triangle_area=max_tri_area)

        # TODO: check if the deep copy is necessary
        self.mesh_vertices = mesh['vertices'].copy()
        self.mesh_faces = mesh['triangles'].copy()

        boundary_faces_indices = []

        for i in range(len(self.mesh_faces)):
            # find the faces which have at least one vertex among the boundary vertices (the indices of boundary
            # vertices in self.vertices are 0,1,...,self.num_boundary_vertices-1)
            if len(np.intersect1d(np.arange(self.num_boundary_vertices), self.mesh_faces[i])) > 0:
                boundary_faces_indices.append(i)

        self.boundary_faces_indices = np.array(boundary_faces_indices)

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
        # TODO: clean up this mess
        # scheme for numerical integration on segments
        scheme = quadpy.c1.gauss_patterson(6)

        # rotation matrix used to compute outward normals
        if self.is_clockwise:
            rot = np.array([[0, -1], [1, 0]])
        else:
            rot = np.array([[0, 1], [-1, 0]])

        rolled_vertices1 = np.roll(self.boundary_vertices, 1, axis=0)
        rolled_vertices2 = np.roll(self.boundary_vertices, -1, axis=0)

        t = 0.5 * (1 + scheme.points)
        x1 = np.multiply.outer(1-t, rolled_vertices1) + np.multiply.outer(t, self.boundary_vertices)
        x2 = np.multiply.outer(1-t, self.boundary_vertices) + np.multiply.outer(t, rolled_vertices2)

        eval1_flat = f(np.reshape(x1, (-1, 2)))
        eval2_flat = f(np.reshape(x2, (-1, 2)))

        eval1 = np.reshape(eval1_flat, x1.shape[:2] + eval1_flat.shape[1:])
        eval1 = eval1 * np.expand_dims(t, tuple(np.arange(1, eval1.ndim)))

        eval2 = np.reshape(eval2_flat, x2.shape[:2] + eval2_flat.shape[1:])
        eval2 = eval2 * np.expand_dims(1-t, tuple(np.arange(1, eval2.ndim)))

        weights1 = 0.5 * np.sum(np.expand_dims(scheme.weights, tuple(np.arange(1, eval1.ndim))) * eval1, axis=0)
        weights2 = 0.5 * np.sum(np.expand_dims(scheme.weights, tuple(np.arange(1, eval2.ndim))) * eval2, axis=0)

        normals1 = np.dot(self.boundary_vertices - rolled_vertices1, rot.T)
        normals2 = np.dot(rolled_vertices2 - self.boundary_vertices, rot.T)

        gradient1 = np.expand_dims(np.moveaxis(weights1, 0, -1), -1) * \
                    np.expand_dims(normals1, tuple(np.arange(weights1.ndim-1)))
        gradient2 = np.expand_dims(np.moveaxis(weights2, 0, -1), -1) * \
                    np.expand_dims(normals2, tuple(np.arange(weights2.ndim-1)))

        return gradient1 + gradient2

    def perform_gradient_descent(self, f, step_size, max_iter, eps_stop, num_points, max_tri_area, num_iter_resampling,
                                 alpha=0.1, beta=0.5):
        # TODO: allow to perform a fixed step gradient descent
        obj_tab = []
        grad_norm_tab = []

        convergence = False
        n_iter = 0

        areas = self.compute_weighted_areas(f)
        perimeter = self.compute_perimeter()
        area = np.sum(areas)

        obj = perimeter / np.abs(area)
        obj_tab.append(obj)

        while not convergence and n_iter < max_iter:
            perimeter_gradient = self.compute_perimeter_gradient()
            area_gradient = self.compute_weighted_area_gradient(f)

            gradient = np.sign(area) * (perimeter_gradient * area - area_gradient * perimeter) / area ** 2
            grad_norm_tab.append(np.max(np.linalg.norm(gradient, axis=1)))

            t = step_size

            ag_condition = False

            former_obj = obj
            former_boundary_vertices = self.boundary_vertices

            while not ag_condition:
                self.boundary_vertices = former_boundary_vertices - t * gradient
                # first vertices are boundary vertices
                self.mesh_vertices[:self.num_boundary_vertices] = self.boundary_vertices

                areas[self.boundary_faces_indices] = integrate_on_triangles(f, self.mesh_vertices[self.mesh_faces[self.boundary_faces_indices]])

                area = np.sum(areas)
                perimeter = self.compute_perimeter()
                obj = perimeter / np.abs(area)

                ag_condition = (obj <= former_obj - alpha * t * np.linalg.norm(gradient) ** 2)
                t = beta * t

            n_iter += 1
            obj_tab.append(obj)

            convergence = np.max(np.linalg.norm(gradient, axis=1)) <= eps_stop

            if num_iter_resampling is not None and n_iter % num_iter_resampling == 0:
                self.resample_boundary(num_points, max_tri_area)
                areas = self.compute_weighted_areas(f)
                area = np.sum(areas)
                perimeter = self.compute_perimeter()
                obj = perimeter / np.abs(area)

        self.mesh(max_tri_area)

        return obj_tab, grad_norm_tab


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

    SimpleSet(vertices, max_tri_area)

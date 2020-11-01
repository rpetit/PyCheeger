import numpy as np
import quadpy

from .tools import winding, resample, integrate_on_triangles, triangulate
from .plot_utils import plot_simple_set


class SimpleSet:
    def __init__(self, boundary_vertices, max_tri_area=0.002):
        self.boundary_vertices = boundary_vertices
        self.num_boundary_vertices = len(boundary_vertices)
        self.boundary_vertices_indices = np.arange(self.num_boundary_vertices)

        rolled_boundary_vertices = np.roll(boundary_vertices, -1, axis=0)
        self.is_clockwise = (np.sum((rolled_boundary_vertices[:, 0] - boundary_vertices[:, 0]) *
                                    (rolled_boundary_vertices[:, 1] + boundary_vertices[:, 1])) > 0)

        self.mesh_vertices = None
        self.mesh_faces = None
        self.boundary_faces_indices = None

        self.mesh(max_tri_area)

    def contains(self, x):
        return winding(x, self.boundary_vertices) != 0

    def compute_perimeter(self):
        rolled_boundary_vertices = np.roll(self.boundary_vertices, -1, axis=0)
        res = np.sum(np.linalg.norm(rolled_boundary_vertices - self.boundary_vertices, axis=1))
        return res

    def compute_weighted_areas(self, f):
        triangles = self.mesh_vertices[self.mesh_faces]
        return integrate_on_triangles(f, triangles)

    def compute_weighted_area(self, f):
        return np.sum(self.compute_weighted_areas(f))

    def resample_boundary(self, num_points, max_tri_area):
        new_boundary_vertices = resample(self.boundary_vertices, num_points)
        self.__init__(new_boundary_vertices, max_tri_area=max_tri_area)

    def mesh(self, max_tri_area):
        mesh = triangulate(self.boundary_vertices, max_area=max_tri_area)

        self.mesh_vertices = mesh.vertices.copy()
        self.mesh_faces = mesh.faces.copy()

        boundary_faces_indices = []

        for i in range(len(self.mesh_faces)):
            if len(np.intersect1d(self.boundary_vertices_indices, self.mesh_faces[i])) > 0:
                boundary_faces_indices.append(i)

        self.boundary_faces_indices = np.array(boundary_faces_indices)

    def compute_perimeter_gradient(self):
        gradient = np.zeros_like(self.boundary_vertices)

        for i in range(self.num_boundary_vertices):
            e1 = self.boundary_vertices[(i-1) % self.num_boundary_vertices] - self.boundary_vertices[i]
            e2 = self.boundary_vertices[(i+1) % self.num_boundary_vertices] - self.boundary_vertices[i]

            gradient[i] = - (e1 / np.linalg.norm(e1) + e2 / np.linalg.norm(e2))

        return gradient

    def compute_weighted_area_gradient(self, f):
        scheme = quadpy.c1.gauss_patterson(6)

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

        gradient1 = np.expand_dims(np.moveaxis(weights1, 0, -1), -1) * np.expand_dims(normals1, tuple(np.arange(weights1.ndim-1)))
        gradient2 = np.expand_dims(np.moveaxis(weights2, 0, -1), -1) * np.expand_dims(normals2, tuple(np.arange(weights2.ndim-1)))

        return gradient1 + gradient2

    def perform_gradient_descent(self, f, step_size, max_iter, eps_stop, num_points, max_tri_area):
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

            grad_norm_tab.append(np.linalg.norm(gradient))

            alpha = 0.1
            beta = 0.5
            t = step_size

            ag_condition = False

            former_obj = obj
            former_boundary_vertices = self.boundary_vertices

            while not ag_condition:
                self.boundary_vertices = former_boundary_vertices - t * gradient
                self.mesh_vertices[self.boundary_vertices_indices] = self.boundary_vertices

                areas[self.boundary_faces_indices] = integrate_on_triangles(f, self.mesh_vertices[self.mesh_faces[self.boundary_faces_indices]])

                area = np.sum(areas)
                perimeter = self.compute_perimeter()
                obj = perimeter / np.abs(area)

                ag_condition = (obj <= former_obj - alpha * t * np.linalg.norm(gradient) ** 2)
                t = beta * t

            n_iter += 1
            obj_tab.append(obj)

            convergence = np.linalg.norm(gradient) / self.num_boundary_vertices <= eps_stop

            if n_iter % 50 == 0:
                self.resample_boundary(num_points, max_tri_area)
                # plot_simple_set(self, eta=f, display_inner_mesh=True)
                areas = self.compute_weighted_areas(f)
                area = np.sum(areas)
                perimeter = self.compute_perimeter()
                obj = perimeter / np.abs(area)

        self.mesh(max_tri_area)

        print(n_iter)

        return obj_tab, grad_norm_tab


class Disk(SimpleSet):
    def __init__(self, center, radius, num_vertices=20, max_tri_area=0.005):
        t = np.linspace(0, 2 * np.pi, num_vertices + 1)[:-1]
        complex_vertices = center[0] + 1j * center[1] + radius * np.exp(1j * t)
        vertices = np.stack([np.real(complex_vertices), np.imag(complex_vertices)], axis=1)

        SimpleSet.__init__(self, vertices, max_tri_area)

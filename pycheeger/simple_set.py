import numpy as np
import quadpy

from .tools import winding, integrate_on_triangles, triangulate
from .plot_utils import plot_simple_set


class SimpleSet:
    def __init__(self, boundary_vertices, max_area=0.005):
        self.boundary_vertices = boundary_vertices
        self.num_boundary_vertices = len(boundary_vertices)
        self.boundary_vertices_indices = np.arange(self.num_boundary_vertices)

        rolled_boundary_vertices = np.roll(boundary_vertices, -1, axis=0)
        self.is_clockwise = (np.sum((rolled_boundary_vertices[:, 0] - boundary_vertices[:, 0]) *
                                    (rolled_boundary_vertices[:, 1] + boundary_vertices[:, 1])) > 0)

        self.mesh_vertices = None
        self.mesh_faces = None
        self.boundary_faces_indices = None

        self.mesh(max_area)

    def contains(self, x):
        return winding(x, self.boundary_vertices) != 0

    def compute_perimeter(self):
        rolled_boundary_vertices = np.roll(self.boundary_vertices, -1, axis=0)
        res = np.sum(np.linalg.norm(rolled_boundary_vertices - self.boundary_vertices, axis=1))
        return res

    def compute_weighted_areas(self, eta):
        triangles = self.mesh_vertices[self.mesh_faces]
        return integrate_on_triangles(eta, triangles)

    def compute_weighted_area(self, eta):
        return np.sum(self.compute_weighted_areas(eta))

    def mesh(self, max_area):
        mesh = triangulate(self.boundary_vertices, max_area=max_area)

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

    def compute_weighted_area_gradient(self, eta):
        scheme = quadpy.c1.gauss_patterson(6)
        gradient = np.zeros_like(self.boundary_vertices)

        for i in range(self.num_boundary_vertices):
            previous = self.boundary_vertices[(i-1) % self.num_boundary_vertices]
            current = self.boundary_vertices[i]
            next = self.boundary_vertices[(i+1) % self.num_boundary_vertices]

            weight1 = scheme.integrate(lambda t: eta(np.outer(previous, 1-t) + np.outer(current, t)) * t, [0.0, 1.0])
            weight2 = scheme.integrate(lambda t: eta(np.outer(current, 1-t) + np.outer(next, t)) * (1-t), [0.0, 1.0])

            if self.is_clockwise:
                rot = np.array([[0, -1], [1, 0]])
            else:
                rot = np.array([[0, 1], [-1, 0]])

            # /!\ normals do not have unit length (length of the segment in change of variable) /!\
            normal1 = rot.dot(current - previous)
            normal2 = rot.dot(next - current)

            gradient[i] = weight1 * normal1 + weight2 * normal2

        return gradient

    def perform_gradient_descent(self, eta, step_size, max_iter, eps_stop):
        obj_tab = []
        grad_norm_tab = []

        convergence = False
        n_iter = 0

        areas = self.compute_weighted_areas(eta)
        perimeter = self.compute_perimeter()
        area = np.sum(areas)

        obj = perimeter / np.abs(area)
        obj_tab.append(obj)

        while not convergence and n_iter < max_iter:
            perimeter_gradient = self.compute_perimeter_gradient()
            area_gradient = self.compute_weighted_area_gradient(eta)

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

                areas[self.boundary_faces_indices] = integrate_on_triangles(eta, self.mesh_vertices[self.mesh_faces[self.boundary_faces_indices]])

                area = np.sum(areas)
                perimeter = self.compute_perimeter()
                obj = perimeter / np.abs(area)

                ag_condition = (obj <= former_obj - alpha * t * np.linalg.norm(gradient) ** 2)
                t = beta * t

            n_iter += 1
            obj_tab.append(obj)

            convergence = np.max(np.linalg.norm(self.boundary_vertices - former_boundary_vertices, axis=1)
                                 / np.linalg.norm(former_boundary_vertices, axis=1)) <= eps_stop

            if n_iter % 100 == 0:
                self.mesh(0.005)
                areas = self.compute_weighted_areas(eta)
                area = np.sum(areas)
                plot_simple_set(self, eta)

        return obj_tab, grad_norm_tab


class Disk(SimpleSet):
    def __init__(self, center, radius, num_vertices=20, max_tri_area=0.005):
        t = np.linspace(0, 2 * np.pi, num_vertices + 1)[:-1]
        complex_vertices = center[0] + 1j * center[1] + radius * np.exp(1j * t)
        vertices = np.stack([np.real(complex_vertices), np.imag(complex_vertices)], axis=1)

        SimpleSet.__init__(self, vertices, max_tri_area)

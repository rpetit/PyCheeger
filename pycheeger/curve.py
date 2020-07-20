import numpy as np
import quadpy

from pymesh import form_mesh

from .tools import integrate_on_triangle


class SimpleClosedCurve:
    def __init__(self, vertices, inner_mesh):
        self.vertices = vertices
        self.inner_mesh = inner_mesh

        self.corresponding_indices = np.zeros(len(vertices), dtype='int')

        for i in range(len(vertices)):
            j = 0

            while not np.all(inner_mesh.vertices[j] == vertices[i]):
                j += 1

            self.corresponding_indices[i] = j

        self.num_vertices = len(vertices)

        rolled_vertices = np.roll(vertices, -1, axis=0)
        self.perimeter = np.sum(np.linalg.norm(rolled_vertices - vertices, axis=1))
        self.is_clockwise = (np.sum((rolled_vertices[:, 0] - vertices[:, 0]) * (rolled_vertices[:, 1] + vertices[:, 1])) > 0)

    def compute_weighted_area(self, eta):
        res = 0

        for face in self.inner_mesh.faces:
            res += integrate_on_triangle(eta, self.inner_mesh.vertices[face])

        return res

    def compute_perimeter_gradient(self):
        gradient = np.zeros_like(self.vertices)

        for i in range(self.num_vertices):
            e1 = self.vertices[(i-1) % self.num_vertices] - self.vertices[i]
            e2 = self.vertices[(i+1) % self.num_vertices] - self.vertices[i]

            gradient[i] = - (e1 / np.linalg.norm(e1) + e2 / np.linalg.norm(e2))

        return gradient

    def compute_weighted_area_gradient(self, eta):
        scheme = quadpy.line_segment.gauss_patterson(5)
        gradient = np.zeros_like(self.vertices)

        for i in range(self.num_vertices):
            previous = self.vertices[(i-1) % self.num_vertices]
            current = self.vertices[i]
            next = self.vertices[(i+1) % self.num_vertices]

            weight1 = scheme.integrate(lambda t: eta(np.outer(previous, 1-t) + np.outer(current, t)) * t, [0.0, 1.0])
            weight2 = scheme.integrate(lambda t: eta(np.outer(current, 1-t) + np.outer(next, t)) * (1-t), [0.0, 1.0])

            if self.is_clockwise:
                rot = np.array([[0, -1], [1, 0]])
            else:
                rot = np.array([[0, 1], [-1, 0]])

            normal1 = rot.dot(current - previous)
            normal1 *= 1 / np.linalg.norm(normal1)

            normal2 = rot.dot(next - current)
            normal2 *= 1 / np.linalg.norm(normal2)

            gradient[i] = weight1 * normal1 + weight2 * normal2

        return gradient

    def perform_gradient_step(self, eta, step_size):
        perimeter = self.perimeter
        area = self.compute_weighted_area(eta)

        perimeter_gradient = self.compute_perimeter_gradient()
        area_gradient = self.compute_weighted_area_gradient(eta)

        gradient = np.sign(area) * (perimeter_gradient * area - area_gradient * perimeter) / area ** 2
        new_vertices = self.vertices - step_size * gradient

        new_inner_mesh_vertices = self.inner_mesh.vertices.copy()
        new_inner_mesh_vertices[self.corresponding_indices] = new_vertices

        new_inner_mesh = form_mesh(new_inner_mesh_vertices, self.inner_mesh.faces)

        self.__init__(new_vertices, new_inner_mesh)

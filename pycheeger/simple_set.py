import numpy as np
import quadpy

from pymesh import form_mesh

from .tools import integrate_on_triangle, triangulate, proj_unit_square, plot_set_boundary


class SimpleSet:
    def __init__(self, boundary_vertices, mesh):
        self.boundary_vertices = boundary_vertices
        self.mesh = mesh

        self.boundary_indices = np.zeros(len(boundary_vertices), dtype='int')

        for i in range(len(boundary_vertices)):
            j = 0

            while not np.all(mesh.vertices[j] == boundary_vertices[i]):
                j += 1

            self.boundary_indices[i] = j

        self.num_boundary_vertices = len(boundary_vertices)

        rolled_boundary_vertices = np.roll(boundary_vertices, -1, axis=0)
        self.perimeter = np.sum(np.linalg.norm(rolled_boundary_vertices - boundary_vertices, axis=1))
        self.is_clockwise = (np.sum((rolled_boundary_vertices[:, 0] - boundary_vertices[:, 0]) *
                                    (rolled_boundary_vertices[:, 1] + boundary_vertices[:, 1])) > 0)

    def compute_weighted_area(self, eta):
        res = 0

        for face in self.mesh.faces:
            res += integrate_on_triangle(eta, self.mesh.vertices[face])

        return res

    def remesh(self, max_area):
        new_mesh = triangulate(self.boundary_vertices, max_area)
        self.__init__(self.boundary_vertices, new_mesh)

    def compute_perimeter_gradient(self):
        gradient = np.zeros_like(self.boundary_vertices)

        for i in range(self.num_boundary_vertices):
            e1 = self.boundary_vertices[(i-1) % self.num_boundary_vertices] - self.boundary_vertices[i]
            e2 = self.boundary_vertices[(i+1) % self.num_boundary_vertices] - self.boundary_vertices[i]

            gradient[i] = - (e1 / np.linalg.norm(e1) + e2 / np.linalg.norm(e2))

        return gradient

    def compute_weighted_area_gradient(self, eta):
        scheme = quadpy.line_segment.gauss_patterson(5)
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

    # TODO: implement line search
    def perform_gradient_descent(self, eta, step_size, max_iter, eps_stop):
        obj_tab = []
        grad_norm_tab = []

        convergence = False
        n_iter = 0

        while not convergence and n_iter < max_iter:
            perimeter = self.perimeter
            area = self.compute_weighted_area(eta)

            obj = perimeter / np.abs(area)
            obj_tab.append(obj)

            perimeter_gradient = self.compute_perimeter_gradient()
            area_gradient = self.compute_weighted_area_gradient(eta)

            gradient = np.sign(area) * (perimeter_gradient * area - area_gradient * perimeter) / area ** 2

            grad_norm_tab.append(np.linalg.norm(gradient))

            alpha = 0.1
            beta = 0.5
            t = step_size

            ag_condition = False

            while not ag_condition:
                new_boundary_vertices = proj_unit_square(self.boundary_vertices - t * gradient)

                new_mesh_vertices = self.mesh.vertices.copy()
                new_mesh_vertices[self.boundary_indices] = new_boundary_vertices

                new_mesh = form_mesh(new_mesh_vertices, self.mesh.faces)

                new_curve = SimpleSet(new_boundary_vertices, new_mesh)
                new_area = new_curve.compute_weighted_area(eta)
                new_perimeter = new_curve.perimeter
                new_obj = new_perimeter / np.abs(new_area)

                ag_condition = (new_obj <= obj - alpha * t * np.linalg.norm(gradient) ** 2)
                t = beta * t

            n_iter += 1
            convergence = (np.linalg.norm(new_boundary_vertices - self.boundary_vertices) / np.linalg.norm(self.boundary_vertices)) <= eps_stop

            if n_iter % 100 == 0:
                self.boundary_vertices = new_boundary_vertices
                self.remesh(0.005)
                plot_set_boundary(self, eta)
            else:
                self.__init__(new_boundary_vertices, new_mesh)

        return obj_tab, grad_norm_tab

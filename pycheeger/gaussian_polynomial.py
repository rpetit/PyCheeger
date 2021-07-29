import numpy as np
import quadpy

from numpy import exp
from numba import jit, prange


def generate_eval_aux(grid, weights, std):
    scale = - 1 / (2 * std ** 2)

    @jit(nopython=True, parallel=True)
    def aux(x, res):
        for i in prange(x.shape[0]):
            for j in range(weights.size):
                squared_norm = (x[i, 0] - grid[j, 0]) ** 2 + (x[i, 1] - grid[j, 1]) ** 2
                res[i] += weights[j] * exp(scale * squared_norm)

    return aux


def generate_square_aux(grid, weights, std):
    scheme = quadpy.c2.get_good_scheme(3)
    scheme_weights = scheme.weights
    scheme_points = (1 + scheme.points.T) / 2
    scale = - 1 / (2 * std ** 2)

    @jit(nopython=True, parallel=True)
    def aux(grid_size, res):
        h = 2 / grid_size
        for i in prange(grid_size):
            for j in prange(grid_size):
                for k in range(scheme_weights.size):
                    x = -1 + h * (scheme_points[k, 0] + i)
                    y = -1 + h * (scheme_points[k, 1] + j)
                    for l in range(grid.shape[0]):
                        squared_norm = (x - grid[l, 0]) ** 2 + (y - grid[l, 1]) ** 2
                        res[i, j] += weights[l] * scheme_weights[k] * exp(scale * squared_norm)

                res[i, j] *= h ** 2

    return aux


def generate_triangle_aux(grid, weights, std):
    scheme = quadpy.t2.get_good_scheme(5)
    scheme_weights = scheme.weights
    scheme_points = scheme.points.T
    scale = - 1 / (2 * std ** 2)

    @jit(nopython=True, parallel=True)
    def aux(triangles, res):
        for i in prange(len(triangles)):
            a = np.sqrt((triangles[i, 1, 0] - triangles[i, 0, 0]) ** 2 + (triangles[i, 1, 1] - triangles[i, 0, 1]) ** 2)
            b = np.sqrt((triangles[i, 2, 0] - triangles[i, 1, 0]) ** 2 + (triangles[i, 2, 1] - triangles[i, 1, 1]) ** 2)
            c = np.sqrt((triangles[i, 2, 0] - triangles[i, 0, 0]) ** 2 + (triangles[i, 2, 1] - triangles[i, 0, 1]) ** 2)
            p = (a + b + c) / 2
            area = np.sqrt(p * (p - a) * (p - b) * (p - c))

            for k in range(scheme_weights.size):
                x = scheme_points[k, 0] * triangles[i, 0, 0] + \
                    scheme_points[k, 1] * triangles[i, 1, 0] + \
                    scheme_points[k, 2] * triangles[i, 2, 0]
                y = scheme_points[k, 0] * triangles[i, 0, 1] + \
                    scheme_points[k, 1] * triangles[i, 1, 1] + \
                    scheme_points[k, 2] * triangles[i, 2, 1]
                for j in range(grid.shape[0]):
                    squared_norm = (x - grid[j, 0]) ** 2 + (y - grid[j, 1]) ** 2
                    res[i] += scheme_weights[k] * weights[j] * exp(scale * squared_norm)

            res[i] *= area

    return aux


def generate_line_aux(grid, weights, std):
    scheme = quadpy.c1.gauss_patterson(3)
    scheme_weights = scheme.weights
    scheme_points = (1 + scheme.points) / 2
    scale = - 1 / (2 * std ** 2)

    @jit(nopython=True, parallel=True)
    def aux(vertices, res):
        for i in prange(len(vertices)):
            edge_length = np.sqrt((vertices[(i + 1) % len(vertices), 0] - vertices[i, 0]) ** 2 +
                                  (vertices[(i + 1) % len(vertices), 1] - vertices[i, 1]) ** 2)

            for k in range(scheme_weights.size):
                x = scheme_points[k] * vertices[i] + (1 - scheme_points[k]) * vertices[(i + 1) % len(vertices)]

                for j in range(grid.shape[0]):
                    squared_norm = (x[0] - grid[j, 0]) ** 2 + (x[1] - grid[j, 1]) ** 2
                    res[i, 0] += scheme_weights[k] * weights[j] * scheme_points[k] * exp(scale * squared_norm)

            res[i, 0] *= edge_length / 2

            edge_length = np.sqrt((vertices[i, 0] - vertices[i - 1, 0]) ** 2 +
                                  (vertices[i, 1] - vertices[i - 1, 1]) ** 2)

            for k in range(scheme_weights.size):
                x = scheme_points[k] * vertices[i] + (1 - scheme_points[k]) * vertices[i - 1]

                for j in range(grid.shape[0]):
                    squared_norm = (x[0] - grid[j, 0]) ** 2 + (x[1] - grid[j, 1]) ** 2
                    res[i, 1] += scheme_weights[k] * weights[j] * scheme_points[k] * exp(scale * squared_norm)

            res[i, 1] *= edge_length / 2

    return aux


class GaussianPolynomial:
    def __init__(self, grid, weights, std, normalization=False):
        self.grid = grid
        self.weights = weights
        self.std = std
        self.normalization = normalization

        self._eval_aux = generate_eval_aux(self.grid, self.weights, self.std)
        self._square_aux = generate_square_aux(self.grid, self.weights, self.std)
        self._triangle_aux = generate_triangle_aux(self.grid, self.weights, self.std)
        self._line_aux = generate_line_aux(self.grid, self.weights, self.std)

    @property
    def grid_size(self):
        return len(self.grid)

    def __call__(self, x):
        if x.ndim == 1:
            tmp = np.zeros(1)
            self._eval_aux(np.reshape(x, (1, 2)), tmp)
            res = tmp[0]
        else:
            res = np.zeros(x.shape[0])
            self._eval_aux(x, res)
        if self.normalization:
            res = res / (2*np.pi * self.std**2)
        return res

    def integrate_on_pixel_grid(self, grid_size):
        res = np.zeros((grid_size, grid_size))
        self._square_aux(grid_size, res)
        return res

    def integrate_on_triangles(self, triangles):
        res = np.zeros(len(triangles))
        self._triangle_aux(triangles, res)
        return res

    def integrate_on_polygonal_curve(self, vertices):
        res = np.zeros((len(vertices), 2))
        self._line_aux(vertices, res)
        return res
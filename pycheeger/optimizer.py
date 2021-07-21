import numpy as np

from .tools import resample
from .plot_utils import plot_simple_set
from .simple_set import SimpleSet

import matplotlib.pyplot as plt


class CheegerOptimizerState:
    def __init__(self, initial_set, f):
        self.set = None
        self.weighted_area_tab = None
        self.weighted_area = None
        self.perimeter = None
        self.obj = None

        self.update_set(initial_set, f)

    def update_obj(self):
        self.weighted_area = np.sum(self.weighted_area_tab)
        self.perimeter = self.set.compute_perimeter()

        self.obj = self.perimeter / np.abs(self.weighted_area)

    def update_boundary_vertices(self, new_boundary_vertices, f):
        self.set.boundary_vertices = new_boundary_vertices

        boundary_weighted_area_tab = self.set.compute_weighted_area_tab(f, boundary_faces_only=True)
        self.weighted_area_tab[self.set.mesh_boundary_faces_indices] = boundary_weighted_area_tab

        self.update_obj()

    def update_set(self, new_set, f):
        self.set = new_set

        weighted_area_tab = self.set.compute_weighted_area_tab(f)
        self.weighted_area_tab = weighted_area_tab

        self.update_obj()

    def compute_gradient(self, f):
        perimeter_gradient = self.set.compute_perimeter_gradient()
        area_gradient = self.set.compute_weighted_area_gradient(f)
        gradient = (perimeter_gradient * self.weighted_area - area_gradient * self.perimeter) / self.weighted_area ** 2

        return np.sign(self.weighted_area) * gradient


class CheegerOptimizer:
    def __init__(self, step_size, max_iter, eps_stop, num_points, max_tri_area, num_iter_resampling, alpha, beta):
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps_stop = eps_stop
        self.num_points = num_points
        self.max_tri_area = max_tri_area
        self.num_iter_resampling = num_iter_resampling
        self.alpha = alpha
        self.beta = beta

        self.state = None

    def perform_linesearch(self, f, gradient):
        t = self.step_size

        ag_condition = False

        former_obj = self.state.obj
        former_boundary_vertices = self.state.set.boundary_vertices

        iteration = 0

        while not ag_condition:
            new_boundary_vertices = former_boundary_vertices - t * gradient
            self.state.update_boundary_vertices(new_boundary_vertices, f)
            new_obj = self.state.obj

            ag_condition = (new_obj <= former_obj - self.alpha * t * np.linalg.norm(gradient) ** 2)
            t = self.beta * t

            iteration += 1

        return iteration

    def run(self, f, initial_set, verbose=True):
        convergence = False
        obj_tab = []
        grad_norm_tab = []

        iteration = 0

        self.state = CheegerOptimizerState(initial_set, f)

        while not convergence and iteration < self.max_iter:
            gradient = self.state.compute_gradient(f)
            grad_norm_tab.append(np.max(np.linalg.norm(gradient, axis=1)))
            obj_tab.append(self.state.obj)

            n_iter_linesearch = self.perform_linesearch(f, gradient)

            iteration += 1
            convergence = False

            if verbose:
                print("iteration {}: {} linesearch steps".format(iteration, n_iter_linesearch))

        if self.num_iter_resampling is not None and iteration % self.num_iter_resampling == 0:
            new_boundary_vertices = resample(self.state.set.boundary_vertices, num_points=self.num_points)
            new_set = SimpleSet(new_boundary_vertices, max_tri_area=self.max_tri_area)
            self.state.update_set(new_set, f)

        return self.state.set, obj_tab, grad_norm_tab

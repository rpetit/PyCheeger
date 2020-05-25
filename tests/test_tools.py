import numpy as np

from pytest import approx
from pymesh import form_mesh
from pycheeger import *


TOL = 1e-10


def test_find_threshold():
    for x in [[1, 2, 3, 4], [-6, 10, -15, 12]]:
        thresh = find_threshold(x)
        assert approx(np.sum(np.maximum(np.abs(x) - thresh, 0)), 1, TOL)


def test_prox_two_inf_norm():
    x = np.array([[1, 2], [-1, 1], [4, 6], [8, -10]])
    y = prox_two_inf_norm(x)

    assert np.sum(np.linalg.norm(y, axis=1)) <= 1 + TOL

    x = np.array([[0, 0], [1, 1], [0, 0]])
    y = prox_two_inf_norm(x)

    assert approx(np.linalg.norm(y[0]), 0, TOL)
    assert approx(np.linalg.norm(y[1]), 1, TOL)
    assert approx(np.linalg.norm(y[2]), 0, TOL)


def test_divergence():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2], [2, 0, 3]])
    mesh = form_mesh(vertices, faces)

    div_mat, edges = build_divergence_matrix(mesh)

    assert np.array_equal(div_mat.toarray(), np.array([[-1, 1, 0, -1, 0], [0, -1, 1, 0, -1]]))

    x = 2 * np.random.rand(5) - 1
    proj_x = project_div_constraint(x, np.zeros(2), div_mat)

    assert np.allclose(div_mat.dot(proj_x), 0)


def test_proj_piecewise_constant():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2], [2, 0, 3]])
    mesh = form_mesh(vertices, faces)

    phi = lambda t: np.array([t[0], 0])

    proj = project_piecewise_constant(phi, mesh)

    assert approx(proj[0], 1, TOL)
    assert approx(proj[1], 1, TOL)

    phi = lambda t: np.array([0, t[0] * t[1]])

    assert approx(proj[0], 1/6, TOL)
    assert approx(proj[1], 1/3, TOL)


def test_eval_vector_field():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2], [2, 0, 3]])
    mesh = form_mesh(vertices, faces)

    div_mat, edges = build_divergence_matrix(mesh)

    fluxes = [0, 0, 0, 0, 1]
    # TODO: write

import numpy as np

from pytest import approx
from pymesh import form_mesh
from pycheeger import *


TOL = 1e-10


def test_find_threshold():
    for x in [[1, 2, 3, 4], [-6, 10, -15, 12]]:
        thresh = find_threshold(x)
        assert approx(np.sum(np.maximum(np.abs(x) - thresh, 0)), 1, TOL)


def test_proj_one_unit_ball():
    x = np.array([-1, 9, 1, 8, -2, 10])
    y = proj_one_unit_ball(x)

    assert np.sum(np.abs(y)) <= 1 + TOL

    x = np.array([0, 2, 0])
    y = proj_one_unit_ball(x)

    assert approx(y[0], 0, TOL)
    assert approx(y[1], 0, TOL)
    assert approx(y[2], 0, TOL)

    x = np.array([1, 1])
    y = proj_one_unit_ball(x)

    assert approx(y[0], 1/2, TOL)
    assert approx(y[1], 1/2, TOL)


def test_grad():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2], [2, 0, 3]])
    mesh = form_mesh(vertices, faces)

    grad_mat = build_grad_matrix(mesh).toarray()

    assert np.allclose(grad_mat, np.array([[1, 0], [-np.sqrt(2), np.sqrt(2)], [0, -1], [1, 0], [0, 1]]))


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

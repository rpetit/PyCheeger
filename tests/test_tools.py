import numpy as np

from pytest import approx
from pymesh import triangle
from pycheeger import find_threshold, prox_two_inf_norm, build_divergence_matrix


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


def test_build_divergence_matrix():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    tri = triangle()
    tri.points = vertices
    tri.max_area = 0.25
    tri.split_boundary = True
    tri.verbosity = 0
    tri.run()

    mesh = tri.mesh

    div_mat, edges = build_divergence_matrix(mesh)

    assert div_mat.shape[0] == mesh.num_faces
    assert div_mat.shape[1] == len(edges)

    # TODO: write more tests...

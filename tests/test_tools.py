import numpy as np

from pytest import approx
from pycheeger.tools import find_threshold, proj_unit_l1_ball


TOL = 1e-10


def test_find_threshold():
    for x in [[1, 2, 3, 4], [-6, 10, -15, 12]]:
        thresh = find_threshold(np.array(x))
        assert approx(np.sum(np.maximum(np.abs(x) - thresh, 0)), 1, TOL)


def test_proj_one_unit_ball():
    x = np.array([-1, 9, 1, 8, -2, 10])
    y = proj_unit_l1_ball(x)

    assert np.sum(np.abs(y)) <= 1 + TOL

    x = np.array([0, 2, 0])
    y = proj_unit_l1_ball(x)

    assert approx(y[0], 0, TOL)
    assert approx(y[1], 0, TOL)
    assert approx(y[2], 0, TOL)

    x = np.array([1, 1])
    y = proj_unit_l1_ball(x)

    assert approx(y[0], 1/2, TOL)
    assert approx(y[1], 1/2, TOL)

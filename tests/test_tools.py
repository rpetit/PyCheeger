from pytest import approx
from pycheeger.tools import *


TOL = 1e-10


def test_winding():
    # test polygon is a triangle
    vertices = np.array([[-1, 0], [0, 1], [1, 0]])

    # winding number should be zero if and only if the point is outside the polygon
    for (x, is_inside) in [([0, 0.5], True), ([0, 2], False), ([0, -1], False), ([-0.5, 0.1], True)]:
        assert winding(np.array(x), vertices) == 0 and not is_inside or \
               winding(np.array(x), vertices) != 0 and is_inside


def test_find_threshold():
    # test the threshold satisfies the condition given by KKT
    for x in [[1, 2, 3, 4], [-6, 10, -15, 12]]:
        thresh = find_threshold(np.array(x))
        assert approx(np.sum(np.maximum(np.abs(x) - thresh, 0)), 1, TOL)


def test_proj_one_unit_ball():
    x = np.array([-1, 9, 1, 8, -2, 10])
    y = proj_unit_l1_ball(x)

    # test the projection belongs to the unit ball
    assert np.linalg.norm(y, ord=1) <= 1 + TOL

    x = np.array([0, 2, 0])
    y = proj_unit_l1_ball(x)

    assert approx(y[0], 0, TOL)
    assert approx(y[1], 0, TOL)
    assert approx(y[2], 0, TOL)

    x = np.array([1, 1])
    y = proj_unit_l1_ball(x)

    assert approx(y[0], 1/2, TOL)
    assert approx(y[1], 1/2, TOL)

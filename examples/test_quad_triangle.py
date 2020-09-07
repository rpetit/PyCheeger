import time
import numpy as np
import quadpy

from math import sqrt

SCHEME = quadpy.t2.get_good_scheme(6)


def f(x, a=1):
    return np.exp(- a * np.linalg.norm(x, axis=-1))


scheme = quadpy.t2.get_good_scheme(6)


def integrate_on_triangles(g, triangles):
    a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)

    p = (a + b + c) / 2
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))

    x = np.tensordot(triangles, SCHEME.points, axes=([1], [0]))
    x = np.swapaxes(x, 1, 2)

    return area * np.dot(g(x), SCHEME.weights)


def integrate_on_triangle(g, vertices):
    x1, x2, x3 = vertices
    a = sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)
    b = sqrt((x3[0] - x2[0]) ** 2 + (x3[1] - x2[1]) ** 2)
    c = sqrt((x3[0] - x1[0]) ** 2 + (x3[1] - x1[1]) ** 2)

    p = (a + b + c) / 2
    area = sqrt(p * (p - a) * (p - b) * (p - c))

    x = np.dot(SCHEME.points.T, vertices)

    return area * np.dot(g(x), SCHEME.weights)


a_tab = np.random.rand(200)
triangles = np.random.random((1000, 3, 2))

integrate_on_triangles(lambda x: f(x), triangles)

start = time.time()
integrate_on_triangles(lambda x: f(x), triangles)
end = time.time()

print(end - start)

start = time.time()
res2 = 0
for i in range(len(triangles)):
    res2 += integrate_on_triangle(lambda x: f(x), triangles[i])
end = time.time()

print(end - start)

import numpy as np
import quadpy

from math import sqrt
from scipy.special import erf

from pycheeger import Disk


E = Disk(np.array([0, 0]), 1, num_vertices=100)
c = np.zeros(2)
std = 0.1


def aux(x):
    return np.exp(-np.linalg.norm(x - c[:, np.newaxis], axis=0) ** 2 / (2 * std ** 2))


def phi(x):
    res = np.zeros((x.shape[0], 2))

    res[:, 0] = np.exp(-(x[:, 1] - c[1]) ** 2) * std * sqrt(np.pi / 2) * erf((x[:, 0] - c[0]) / (sqrt(2) * std))
    res[:, 0] = np.exp(-(x[:, 0] - c[0]) ** 2) * std * sqrt(np.pi / 2) * erf((x[:, 1] - c[1]) / (sqrt(2) * std))

    return 0.5 * res


x = np.random.random((1, 2))
t = 1e-5
print("check finite diff")
print(phi(x + t * np.array([[1, 0]]))[0, 0] - phi(x)[0, 0] - t * 0.5 * aux(x.T)[0])
print(phi(x + t * np.array([[0, 1]]))[0, 1] - phi(x)[0, 1] - t * 0.5 * aux(x.T)[0])

print("\nanalytic value")
print(2 * np.pi * std**2 * (1 - np.exp(-1 / (2 * std ** 2))))

scheme = quadpy.s2.get_good_scheme(17)
print("\ndisk quadrature")
print(scheme.integrate(aux, [0.0, 0.0], 1.0))


res1 = 0

scheme = quadpy.c1.gauss_patterson(8)

for i in range(E.num_boundary_vertices):
    current = E.boundary_vertices[i]
    next = E.boundary_vertices[(i+1) % E.num_boundary_vertices]

    if E.is_clockwise:
        rot = np.array([[0, -1], [1, 0]])
    else:
        rot = np.array([[0, 1], [-1, 0]])

    normal = rot.dot(next - current)

    def lala(t):
        return np.dot(phi(np.outer(1 - t, current) + np.outer(t, next)), normal)

    res1 += scheme.integrate(lala, [0.0, 1.0])

print("\nline quadrature")
print(res1)

res2 = 0

scheme = quadpy.t2.get_good_scheme(12)

for i in range(len(E.mesh_faces)):
    res2 += scheme.integrate(aux, E.mesh_vertices[E.mesh_faces[i]])

print("\ntriangle quadrature")
print(res2)

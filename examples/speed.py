import time
import numpy as np

from numba import njit

std = 0.2
coeffs = np.random.rand(3)
means = np.random.random((3, 2))


def eta1(x):
    res = coeffs[0] * np.exp(-np.linalg.norm(x - means[0]) ** 2 / (2 * std ** 2))

    for i in range(1, len(coeffs)):
        res += coeffs[i] * np.exp(-np.linalg.norm(x - means[i]) ** 2 / (2 * std ** 2))

    return res


@njit
def eta2(x):
    res = coeffs[0] * np.exp(-np.linalg.norm(x - means[0]) ** 2 / (2 * std ** 2))

    for i in range(1, len(coeffs)):
        res += coeffs[i] * np.exp(-np.linalg.norm(x - means[i]) ** 2 / (2 * std ** 2))

    return res

@njit
def eta3(x):
    squared_norm = (x[0] - means[0, 0]) ** 2 + (x[1] - means[0, 1]) ** 2
    res = coeffs[0] * np.exp(-squared_norm / (2 * std ** 2))

    for i in range(1, len(coeffs)):
        squared_norm = (x[0] - means[i, 0]) ** 2 + (x[1] - means[i, 1]) ** 2
        res += coeffs[i] * np.exp(-squared_norm / (2 * std ** 2))

    return res


@njit
def eta4(x, res):
    for i in range(len(coeffs)):
        squared_norm = (x[0] - means[i, 0]) ** 2 + (x[1] - means[i, 1]) ** 2
        res += coeffs[i] * np.exp(-squared_norm / (2 * std ** 2))


x = np.random.rand(2)
lala = eta1(x)
lala = eta2(x)
lala = eta3(x)

res = 0.0
lala = eta4(x, res)

start = time.time()

for i in range(1000):
    x = np.random.rand(2)
    lala = eta1(x)

end = time.time()

print("%s" % (end - start))

start = time.time()

for i in range(1000):
    x = np.random.rand(2)
    lala = eta2(x)

end = time.time()

print("%s" % (end - start))

start = time.time()

for i in range(1000):
    x = np.random.rand(2)
    lala = eta3(x)

end = time.time()

print("%s" % (end - start))

start = time.time()

for i in range(1000):
    x = np.random.rand(2)
    res = 0.0
    lala = eta4(x, res)

end = time.time()

print("%s" % (end - start))


def eta1_vec(x):
    res = coeffs[0] * np.exp(-np.linalg.norm(x - means[0, :, np.newaxis], axis=0) ** 2 / (2 * std ** 2))

    for i in range(1, len(coeffs)):
        res += coeffs[i] * np.exp(-np.linalg.norm(x - means[i, :, np.newaxis], axis=0) ** 2 / (2 * std ** 2))

    return res

@njit
def lala(x, res):
    for j in range(x.shape[1]):
        for i in range(len(coeffs)):
            norm_squared = (x[0, j] - means[0, i]) ** 2 + (x[1, j] - means[1, i]) ** 2
            res[j] += coeffs[i] * np.exp(-norm_squared / (2 * std ** 2))


def eta2_vec(x):
    res = np.zeros(x.shape[1])
    lala(x, res)
    return res


x = np.random.random((2, 100))
lilou = eta1_vec(x)
lilou = eta2_vec(x)

start = time.time()

for i in range(1000):
    x = np.random.random((2, 100))
    lilou = eta1_vec(x)

end = time.time()

print("%s" % (end - start))

start = time.time()

for i in range(1000):
    x = np.random.random((2, 100))
    lilou = eta2_vec(x)

end = time.time()

print("%s" % (end - start))

import time
from pycheeger import *

np.random.seed(0)

x = np.random.rand(10)
print(proj_one_unit_ball(x))

x = np.random.rand(5000)

tic = time.perf_counter()

for i in range(500):
    lala = proj_one_unit_ball(x)

toc = time.perf_counter()

print(f"{toc - tic:0.4f} seconds")
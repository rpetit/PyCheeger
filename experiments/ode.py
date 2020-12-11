from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, z = np.meshgrid(np.linspace(-1, 1, 10),
                      np.linspace(-1, 1, 10),
                      np.linspace(0, 2*np.pi, 20))

std = 0.5

u = np.cos(z)
v = np.sin(z)
w = np.exp(- (x**2 + y**2) / (2*std**2))

ax.quiver(x, y, z, u, v, w, length=0.3)
plt.show()

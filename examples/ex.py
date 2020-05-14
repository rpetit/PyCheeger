import numpy as np
import pymesh
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
tri = pymesh.triangle()
tri.points = vertices
tri.max_area = 0.01
tri.split_boundary = True
tri.verbosity = 0
tri.run()
mesh = tri.mesh

triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces)
plt.triplot(triangulation, color='black')
plt.axis('equal')
plt.show()

from pycheeger import build_divergence_matrix

mat, E = build_divergence_matrix(mesh)
print(mesh.faces)
print(mat.toarray())
print(E)

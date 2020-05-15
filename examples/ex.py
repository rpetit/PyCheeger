import numpy as np
import pymesh

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from pycheeger import build_divergence_matrix, project_div_constraint

vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
tri = pymesh.triangle()
tri.points = vertices
tri.max_area = 0.25
tri.split_boundary = True
tri.verbosity = 0
tri.run()
mesh = tri.mesh

triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces)
plt.triplot(triangulation, color='black')
plt.axis('equal')
plt.show()

div_mat, edges = build_divergence_matrix(mesh)
x = np.zeros(div_mat.shape[1])
proj_x = project_div_constraint(x, np.zeros(div_mat.shape[0]), div_mat)
print(div_mat.dot(proj_x))

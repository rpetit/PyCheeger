import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# TODO: clean, comment, write docstrings

def plot_primal_dual_results(mesh, u, eta_bar):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 14))

    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.faces)

    eta_avg = eta_bar / np.array([mesh.get_face_area(face_index) for face_index in range(mesh.num_faces)])

    v_abs_max = max(np.max(np.abs(u)), np.max(np.abs(eta_avg)))

    axs[0].triplot(triangulation, color='black', alpha=0.1)
    axs[0].axis('equal')
    axs[0].axis('off')
    im = axs[0].tripcolor(triangulation, facecolors=eta_avg, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
    fig.colorbar(im, ax=axs[0])

    axs[1].triplot(triangulation, color='black', alpha=0.1)
    axs[1].axis('equal')
    axs[1].axis('off')
    im = axs[1].tripcolor(triangulation, facecolors=u, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
    fig.colorbar(im, ax=axs[1])

    plt.show()


def plot_simple_set(simple_set, eta=None, display_inner_mesh=False, boundary_color='black'):
    fig, ax = plt.subplots(figsize=(7, 7))

    if eta is not None:
        x = np.arange(-1.0, 1.0, 0.01)
        y = np.arange(-1.0, 1.0, 0.01)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.zeros_like(x_grid)

        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                z_grid[i, j] = eta(np.array([x_grid[i, j], y_grid[i, j]]))

        v_abs_max = np.max(np.abs(z_grid))

        im = ax.contourf(x_grid, y_grid, z_grid, levels=30, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
        fig.colorbar(im, ax=ax)

    x_curve = np.append(simple_set.boundary_vertices[:, 0], simple_set.boundary_vertices[0, 0])
    y_curve = np.append(simple_set.boundary_vertices[:, 1], simple_set.boundary_vertices[0, 1])

    ax.plot(x_curve, y_curve, color=boundary_color)

    if display_inner_mesh:
        triangulation = Triangulation(simple_set.mesh_vertices[:, 0],
                                      simple_set.mesh_vertices[:, 1],
                                      simple_set.mesh_faces)

        ax.triplot(triangulation, color='black', alpha=0.3)

    ax.axis('equal')
    ax.axis('off')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    plt.show()


def plot_simple_functions(u1, u2, display_inner_mesh=False, boundary_color='black', save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(2):
        if i == 0:
            u = u1
            color = 'black'
        else:
            u = u2
            color = 'red'
            boundary_color = 'red'

        for atom in u.atoms:
            simple_set = atom.support

            x_curve = np.append(simple_set.boundary_vertices[:, 0], simple_set.boundary_vertices[0, 0])
            y_curve = np.append(simple_set.boundary_vertices[:, 1], simple_set.boundary_vertices[0, 1])

            ax.plot(x_curve, y_curve, color=boundary_color)

            if display_inner_mesh:
                triangulation = Triangulation(simple_set.mesh_vertices[:, 0],
                                              simple_set.mesh_vertices[:, 1],
                                              simple_set.mesh_faces)

                ax.triplot(triangulation, color=color, alpha=0.3)

    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

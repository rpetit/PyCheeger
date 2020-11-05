import pytest

from pycheeger.mesh import *
from pycheeger.tools import triangulate


def test():
    vertices = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    raw_mesh = triangulate(vertices)
    mesh = Mesh(raw_mesh)

    # test the two possible representations of an edge yield the same result
    assert mesh.get_edge_index([0, 1]) == mesh.get_edge_index([1, 0])

    # test edge belongs to all of its adjacent faces
    adjacent_faces = mesh.faces[mesh.get_edge_adjacent_faces([0, 1])]
    for face in adjacent_faces:
        assert 0 in face and 1 in face

    # test orientation
    assert mesh.get_orientation(0, [1, 2]) == 0
    assert mesh.get_orientation(0, [1, 3]) == - mesh.get_orientation(1, [1, 3])

    # test length
    assert mesh.get_edge_length([0, 1]) == pytest.approx(1)

    # test get face edges
    face_edges = mesh.get_face_edges(0)
    assert len(face_edges) == 3
    assert [0, 1] in face_edges and [0, 3] in face_edges and [1, 3] in face_edges

    # test face area
    assert mesh.get_face_area(0) == pytest.approx(0.5, abs=1e-5)

    # test integration on the mesh
    # f must handle array inputs of shape (N, 2), see the documentation
    def f(x):
        if x.ndim == 1:
            return 1
        else:
            return np.ones(x.shape[0])

    # assert the integral of the constant function equal to one yields the total area of the mesh
    assert np.sum(mesh.integrate(f)) == pytest.approx(np.sum([mesh.get_face_area(i) for i in range(mesh.num_faces)]))

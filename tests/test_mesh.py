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

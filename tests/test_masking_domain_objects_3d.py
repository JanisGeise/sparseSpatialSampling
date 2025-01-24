"""
    Here: tests for masking out cells outside of domains for all 2D geometry objects available,
    S^3 has to return invalid = 'True' if a cell is outside the domain
"""
import pytest
import torch as pt

from sparseSpatialSampling.geometry import CubeGeometry, SphereGeometry, GeometrySTL3D, CylinderGeometry3D


def test_spherical_domain_3d():
    cylinder = SphereGeometry("sphere", True, [0.0, 0.0, 0.0], 0.5)

    # generate cell completely inside the sphere
    cell_inside = pt.tensor([[-0.25, -0.25, -0.25], [-0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, -0.25, -0.25],
                             [-0.25, -0.25, 0.25], [-0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, -0.25, 0.25]])

    # generate cell completely outside the sphere
    cell_outside = pt.tensor([[1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 1, 1],
                              [1, 1, 2], [1, 2, 2], [2, 2, 2], [2, 1, 2]])

    # generate cell partially inside the sphere
    cell_part = pt.tensor([[-0.25, -0.25, -0.25], [-0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, -0.25, -0.25],
                           [-0.25, -0.25, 5.25], [-0.25, 0.25, 5.25], [0.25, 0.25, 5.25], [0.25, -0.25, 5.25]])

    # invalid if point is outside the domain
    assert cylinder.check_cell(cell_outside) is True

    # valid if point is inside the domain
    assert cylinder.check_cell(cell_inside) is False

    # valid if point is partially inside the domain
    assert cylinder.check_cell(cell_part) is False


def test_cubic_domain_3d():
    domain = CubeGeometry("domain", True, [0, 0, 0], [1, 1, 1])

    # generate cell completely inside the domain
    cell_inside = pt.tensor([[0.25, 0.25, 0.25], [0.25, 0.75, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.25],
                             [0.25, 0.25, 0.75], [0.25, 0.75, 0.75], [0.75, 0.75, 0.75], [0.75, 0.25, 0.75]])

    # generate cell completely outside the domain
    cell_outside = pt.tensor([[2, 2, 2], [2, 3, 2], [3, 3, 2], [3, 2, 2], [2, 2, 3], [2, 3, 3], [3, 3, 3], [3, 2, 3]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [1.5, 1.5, 0.5], [1.5, 0.5, 0.5],
                           [0.5, 0.5, 1.5], [0.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 0.5, 1.5]])

    # invalid if point is outside the domain
    assert domain.check_cell(cell_outside) is True

    # valid if point is inside the domain
    assert domain.check_cell(cell_inside) is False

    # valid if point is partially inside the domain
    assert domain.check_cell(cell_part) is False


def test_domain_3d_stl():
    # load the STL file of the cube and create a geometry object from it
    cube = GeometrySTL3D("cube", True, "cube.stl")

    # generate cell completely inside the cube
    cell_inside = pt.tensor([[3.75, 4.25, 0.25], [3.75, 4.75, 0.25], [3.75, 4.25, 0.75], [3.75, 4.75, 0.75],
                             [4.25, 4.25, 0.25], [4.25, 4.75, 0.25], [4.25, 4.25, 0.75], [4.25, 4.75, 0.75]])

    # generate cell completely outside the cube
    cell_outside = pt.tensor([[5.25, 4.25, 0.25], [5.25, 4.75, 0.25], [5.25, 4.25, 0.75], [5.25, 4.75, 0.75],
                              [5.75, 4.25, 0.25], [5.75, 4.75, 0.25], [5.75, 4.25, 0.75], [5.75, 4.75, 0.75]])

    # generate cell partially inside the cube
    cell_part = pt.tensor([[3.25, 4.25, 0.25], [3.25, 4.75, 0.25], [3.25, 4.25, 0.75], [3.25, 4.75, 0.75],
                           [3.75, 4.25, 0.25], [3.75, 4.75, 0.25], [3.75, 4.25, 0.75], [3.75, 4.75, 0.75]])

    # invalid if point is outside the domain
    assert cube.check_cell(cell_outside) is True

    # valid if point is inside the domain
    assert cube.check_cell(cell_inside) is False

    # valid if point is partially inside the domain
    assert cube.check_cell(cell_part) is False


def test_cylindrical_domain_3d():
    cylinder = CylinderGeometry3D("cylinder", True, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 0.5)

    # generate cell completely inside the cylinder
    cell_inside = pt.tensor([[0.25, -0.25, -0.25], [0.25, 0.25, -0.25], [0.5, 0.25, -0.25], [0.5, -0.25, -0.25],
                             [0.25, -0.25, 0.25], [0.25, 0.25, 0.25], [0.5, 0.25, -0.25], [0.5, -0.25, 0.25]])

    # generate cell completely outside the cylinder
    cell_outside = pt.tensor([[2, 1, 1], [2, 2, 1], [3, 2, 1], [3, 1, 1],
                              [2, 1, 2], [2, 2, 2], [3, 2, 2], [3, 1, 2]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.25, -0.25, -0.25], [0.25, 0.25, -0.25], [0.5, 0.25, -0.25], [0.5, -0.25, -0.25],
                           [0.25, -0.25, 5.25], [0.25, 0.25, 5.25], [0.5, 0.25, -5.25], [0.5, -0.25, 5.25]])

    # invalid if point is outside the domain
    assert cylinder.check_cell(cell_outside) is True

    # valid if point is inside the domain
    assert cylinder.check_cell(cell_inside) is False

    # valid if point is partially inside the domain
    assert cylinder.check_cell(cell_part) is False


"""
    Here: tests for masking out cells inside geometries for all 3D geometry objects available,
    S^3 has to return invalid = 'True' if a cell is inside the geometry
"""
import pytest
import torch as pt
from os.path import join

from ..geometry import CubeGeometry, SphereGeometry, GeometrySTL3D, CylinderGeometry3D


def test_cubic_geometry_3d():
    # create cube
    cube = CubeGeometry("cube", False, [0, 0, 0], [1, 1, 1])

    # generate cell completely inside the cube
    cell_inside = pt.tensor([[0.25, 0.25, 0.25], [0.25, 0.75, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.25],
                             [0.25, 0.25, 0.75], [0.25, 0.75, 0.75], [0.75, 0.75, 0.75], [0.75, 0.25, 0.75]])

    # generate cell completely outside the cube
    cell_outside = pt.tensor([[2, 2, 2], [2, 3, 2], [3, 3, 2], [3, 2, 2], [2, 2, 3], [2, 3, 3], [3, 3, 3], [3, 2, 3]])

    # generate cell partially inside the cube
    cell_part = pt.tensor([[0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [1.5, 1.5, 0.5], [1.5, 0.5, 0.5],
                           [0.5, 0.5, 1.5], [0.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 0.5, 1.5]])

    # valid if point is outside the geometry
    assert cube.check_cell(cell_outside).item() is False

    # invalid if point is inside the geometry
    assert cube.check_cell(cell_inside).item() is True

    # valid if point is partially inside the geometry
    assert cube.check_cell(cell_part).item() is False


def test_geometry_3d_stl():
    # load the STL file of the cube and create a geometry object from it
    cube = GeometrySTL3D("cube", False, join("sparseSpatialSampling", "tests", "cube.stl"))

    # generate cell completely inside the cube
    cell_inside = pt.tensor([[3.75, 4.25, 0.25], [3.75, 4.75, 0.25], [3.75, 4.25, 0.75], [3.75, 4.75, 0.75],
                             [4.25, 4.25, 0.25], [4.25, 4.75, 0.25], [4.25, 4.25, 0.75], [4.25, 4.75, 0.75]])

    # generate cell completely outside the cube
    cell_outside = pt.tensor([[5.25, 4.25, 0.25], [5.25, 4.75, 0.25], [5.25, 4.25, 0.75], [5.25, 4.75, 0.75],
                              [5.75, 4.25, 0.25], [5.75, 4.75, 0.25], [5.75, 4.25, 0.75], [5.75, 4.75, 0.75]])

    # generate cell partially inside the cube
    cell_part = pt.tensor([[3.25, 4.25, 0.25], [3.25, 4.75, 0.25], [3.25, 4.25, 0.75], [3.25, 4.75, 0.75],
                           [3.75, 4.25, 0.25], [3.75, 4.75, 0.25], [3.75, 4.25, 0.75], [3.75, 4.75, 0.75]])

    # valid if point is outside the geometry
    assert cube.check_cell(cell_outside).item() is False

    # invalid if point is inside the geometry
    assert cube.check_cell(cell_inside).item() is True

    # valid if point is partially inside the geometry
    assert cube.check_cell(cell_part).item() is False


def test_spherical_geometry_3d():
    cylinder = SphereGeometry("sphere", False, [0.0, 0.0, 0.0], 0.5)

    # generate cell completely inside the sphere
    cell_inside = pt.tensor([[-0.25, -0.25, -0.25], [-0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, -0.25, -0.25],
                             [-0.25, -0.25, 0.25], [-0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, -0.25, 0.25]])

    # generate cell completely outside the sphere
    cell_outside = pt.tensor([[1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 1, 1],
                              [1, 1, 2], [1, 2, 2], [2, 2, 2], [2, 1, 2]]).float()

    # generate cell partially inside the sphere
    cell_part = pt.tensor([[-0.25, -0.25, -0.25], [-0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.25, -0.25, -0.25],
                           [-0.25, -0.25, 5.25], [-0.25, 0.25, 5.25], [0.25, 0.25, 5.25], [0.25, -0.25, 5.25]])

    # valid if point is outside the geometry
    assert cylinder.check_cell(cell_outside).item() is False

    # invalid if point is inside the geometry
    assert cylinder.check_cell(cell_inside).item() is True

    # valid if point is partially inside the geometry
    assert cylinder.check_cell(cell_part).item() is False


def test_cylindrical_geometry_3d():
    cylinder = CylinderGeometry3D("cylinder", False, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 0.5)

    # generate cell completely inside the cylinder
    cell_inside = pt.tensor([[0.25, -0.25, -0.25], [0.25, 0.25, -0.25], [0.5, 0.25, -0.25], [0.5, -0.25, -0.25],
                             [0.25, -0.25, 0.25], [0.25, 0.25, 0.25], [0.5, 0.25, -0.25], [0.5, -0.25, 0.25]])

    # generate cell completely outside the cylinder
    cell_outside = pt.tensor([[2, 1, 1], [2, 2, 1], [3, 2, 1], [3, 1, 1],
                              [2, 1, 2], [2, 2, 2], [3, 2, 2], [3, 1, 2]]).float()

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.25, -0.25, -0.25], [0.25, 0.25, -0.25], [0.5, 0.25, -0.25], [0.5, -0.25, -0.25],
                           [0.25, -0.25, 5.25], [0.25, 0.25, 5.25], [0.5, 0.25, -5.25], [0.5, -0.25, 5.25]])

    # valid if point is outside the geometry
    assert cylinder.check_cell(cell_outside).item() is False

    # invalid if point is inside the geometry
    assert cylinder.check_cell(cell_inside).item() is True

    # valid if point is partially inside the geometry
    assert cylinder.check_cell(cell_part).item() is False

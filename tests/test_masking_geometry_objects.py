"""
    Here: tests for masking out cells inside geometries and outside of domains for 2D & 3D with and without STL files
"""
import pytest
import torch as pt

from s_cube.geometry import CubeGeometry, SphereGeometry, GeometryCoordinates2D, GeometrySTL3D


def test_domain_2d():
    domain = CubeGeometry("cube", True, [0, 0], [1, 1])

    # generate cell completely inside the domain
    cell_inside = pt.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.75], [0.75, 0.25]])

    # generate cell completely outside the domain
    cell_outside = pt.tensor([[2, 2], [2, 3], [3, 3], [3, 2]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])

    # valid if point is outside the domain
    assert domain.check_cell(cell_outside) is True

    # invalid if point is inside the domain
    assert domain.check_cell(cell_inside) is False

    # valid if point is partially inside the domain
    assert domain.check_cell(cell_part) is False


def test_domain_2d_stl():
    rectangle = GeometryCoordinates2D("rectangle", True, [[0, 0], [0, 1], [1, 1], [1, 0]])

    # generate cell completely inside the domain
    cell_inside = pt.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.75], [0.75, 0.25]])

    # generate cell completely outside the domain
    cell_outside = pt.tensor([[2, 2], [2, 3], [3, 3], [3, 2]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])

    # invalid if point is outside the domain
    assert rectangle.check_cell(cell_outside) is True

    # valid if point is inside the domain
    assert rectangle.check_cell(cell_inside) is False

    # valid if point is partially inside the domain
    assert rectangle.check_cell(cell_part) is False


def test_domain_3d():
    domain = CubeGeometry("domain", True, [0, 0, 0], [1, 1, 1])

    # generate cell completely inside the domain
    cell_inside = pt.tensor([[0.25, 0.25, 0.25], [0.25, 0.75, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.25],
                             [0.25, 0.25, 0.75], [0.25, 0.75, 0.75], [0.75, 0.75, 0.75], [0.75, 0.25, 0.75]])

    # generate cell completely outside the domain
    cell_outside = pt.tensor([[2, 2, 2], [2, 3, 2], [3, 3, 2], [3, 2, 2], [2, 2, 3], [2, 3, 3], [3, 3, 3], [3, 2, 3]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [1.5, 1.5, 0.5], [1.5, 0.5, 0.5],
                           [0.5, 0.5, 1.5], [0.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 0.5, 1.5]])

    # valid if point is outside the domain
    assert domain.check_cell(cell_outside) is True

    # invalid if point is inside the domain
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


def test_geometry_2d():
    cylinder = SphereGeometry("cylinder", False, [0.2, 0.2], 0.05)

    # generate cell completely inside the cylinder
    cell_inside = pt.tensor([[0.175, 0.175], [0.175, 0.225], [0.225, 0.225], [0.225, 0.175]])

    # generate cell completely outside the cylinder
    cell_outside = pt.tensor([[0.5, -0.05], [0.5, 0.05], [1, 0.05], [1, -0.05]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.2, 0.2], [0.2, 0.5], [0.5, 0.5], [0.5, 0.2]])

    # valid if point is outside the geometry
    assert cylinder.check_cell(cell_outside) is False

    # invalid if point is inside the geometry
    assert cylinder.check_cell(cell_inside) is True

    # valid if point is partially inside the geometry
    assert cylinder.check_cell(cell_part) is False


def test_geometry_2d_stl():
    rectangle = GeometryCoordinates2D("rectangle", False, [[0, 0], [0, 1], [1, 1], [1, 0]])

    # generate cell completely inside the domain
    cell_inside = pt.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.75], [0.75, 0.25]])

    # generate cell completely outside the domain
    cell_outside = pt.tensor([[2, 2], [2, 3], [3, 3], [3, 2]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])

    # valid if point is outside the geometry
    assert rectangle.check_cell(cell_outside) is False

    # invalid if point is inside the geometry
    assert rectangle.check_cell(cell_inside) is True

    # valid if point is partially inside the geometry
    assert rectangle.check_cell(cell_part) is False


def test_geometry_3d():
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
    assert cube.check_cell(cell_outside) is False

    # invalid if point is inside the geometry
    assert cube.check_cell(cell_inside) is True

    # valid if point is partially inside the geometry
    assert cube.check_cell(cell_part) is False


def test_geometry_3d_stl():
    # load the STL file of the cube and create a geometry object from it
    cube = GeometrySTL3D("cube", False, "cube.stl")

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
    assert cube.check_cell(cell_outside) is False

    # invalid if point is inside the geometry
    assert cube.check_cell(cell_inside) is True

    # valid if point is partially inside the geometry
    assert cube.check_cell(cell_part) is False

"""
    Here: tests for masking out cells inside geometries for all 2D geometry objects available,
    S^3 has to return invalid = 'True' if a cell is inside the geometry
"""
import pytest
import torch as pt

from ..geometry import CubeGeometry, SphereGeometry, GeometryCoordinates2D, TriangleGeometry


def test_spherical_geometry_2d():
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


def test_cubic_geometry_2d():
    domain = CubeGeometry("cube", False, [0, 0], [1, 1])

    # generate cell completely inside the domain
    cell_inside = pt.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.75], [0.75, 0.25]])

    # generate cell completely outside the domain
    cell_outside = pt.tensor([[2, 2], [2, 3], [3, 3], [3, 2]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])

    # valid if point is outside the geometry
    assert domain.check_cell(cell_outside) is False

    # invalid if point is inside the geometry
    assert domain.check_cell(cell_inside) is True

    # valid if point is partially inside the geometry
    assert domain.check_cell(cell_part) is False


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


def test_triangle_geometry_2d():
    triangle = TriangleGeometry("triangle", False, [(0.5, 1), (1, 0), (0, 0)])

    # generate cell completely inside the cylinder
    cell_inside = pt.tensor([[0.175, 0.175], [0.175, 0.225], [0.225, 0.225], [0.225, 0.175]])

    # generate cell completely outside the cylinder
    cell_outside = pt.tensor([[2, 2], [3, 2], [3, 3], [2, 3]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.2, 0.2], [0.2, 0.5], [0.5, 0.5], [0.5, 0.2]])

    # valid if point is outside the geometry
    assert triangle.check_cell(cell_outside) is False

    # invalid if point is inside the geometry
    assert triangle.check_cell(cell_inside) is True

    # valid if point is partially inside the geometry
    assert triangle.check_cell(cell_part) is False
"""
    Here: tests for masking out cells outside of domains for all 2D geometry objects available,
    S^3 has to return invalid = 'True' if a cell is outside the domain
"""
import pytest
import torch as pt

from ..geometry import CubeGeometry, SphereGeometry, GeometryCoordinates2D


def test_cubic_domain_2d():
    domain = CubeGeometry("cube", True, [0, 0], [1, 1])

    # generate cell completely inside the domain
    cell_inside = pt.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.75], [0.75, 0.25]])

    # generate cell completely outside the domain
    cell_outside = pt.tensor([[2, 2], [2, 3], [3, 3], [3, 2]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])

    # valid if point is outside the domain
    assert domain.check_cell(cell_outside).item() is True

    # invalid if point is inside the domain
    assert domain.check_cell(cell_inside).item() is False

    # valid if point is partially inside the domain
    assert domain.check_cell(cell_part).item() is False


def test_spherical_domain_2d():
    cylinder = SphereGeometry("cylinder", True, [0.2, 0.2], 0.05)

    # generate cell completely inside the cylinder
    cell_inside = pt.tensor([[0.175, 0.175], [0.175, 0.225], [0.225, 0.225], [0.225, 0.175]])

    # generate cell completely outside the cylinder
    cell_outside = pt.tensor([[0.5, -0.05], [0.5, 0.05], [1, 0.05], [1, -0.05]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.2, 0.2], [0.2, 0.5], [0.5, 0.5], [0.5, 0.2]])

    # valid if point is outside the geometry
    assert cylinder.check_cell(cell_outside).item() is True

    # invalid if point is inside the geometry
    assert cylinder.check_cell(cell_inside).item() is False

    # valid if point is partially inside the geometry
    assert cylinder.check_cell(cell_part).item() is False


def test_domain_2d_stl():
    rectangle = GeometryCoordinates2D("rectangle", True, [[0, 0], [0, 1], [1, 1], [1, 0]])

    # generate cell completely inside the domain
    cell_inside = pt.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.75], [0.75, 0.25]])

    # generate cell completely outside the domain
    cell_outside = pt.tensor([[2, 2], [2, 3], [3, 3], [3, 2]])

    # generate cell partially inside the cylinder
    cell_part = pt.tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])

    # invalid if point is outside the domain
    assert rectangle.check_cell(cell_outside).item() is True

    # valid if point is inside the domain
    assert rectangle.check_cell(cell_inside).item() is False

    # valid if point is partially inside the domain
    assert rectangle.check_cell(cell_part).item() is False

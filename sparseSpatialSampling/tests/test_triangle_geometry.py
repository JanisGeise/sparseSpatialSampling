"""
Unit tests for ``TriangleGeometry`` in 2D using the ``DummyCells`` class.
"""
import pytest
from torch import tensor

from ..geometry import TriangleGeometry
from .import DummyCells


class TestTriangleGeometry2D:
    @pytest.fixture
    def valid_points(self):
        return [(-1, -0.5), (0.25, 4), (1.5, -0.5)]

    def test_valid_triangle_does_not_raise(self, valid_points):
        """
        A valid triangle should pass without AssertionError.
        """
        TriangleGeometry("triangle", keep_inside=False, points=valid_points)

    def test_too_few_points_raises(self):
        """
        Less than 3 points should raise an AssertionError.
        """
        with pytest.raises(AssertionError, match="Expected 3 points"):
            TriangleGeometry("triangle", keep_inside=False, points=[(0, 0), (1, 0)])

    def test_too_many_points_raises(self):
        """
        More than 3 points should raise an AssertionError.
        """
        with pytest.raises(AssertionError, match="Expected 3 points"):
            TriangleGeometry("triangle", keep_inside=False, points=[(0, 0), (1, 0), (0, 1), (1, 1)])

    def test_points_with_wrong_dimension_raises(self):
        """
        Each point must have exactly 2 coordinates.
        """
        with pytest.raises(AssertionError, match="have to contain exactly 2 entries"):
            TriangleGeometry("triangle", keep_inside=False, points=[(0, 0), (1, 1, 5), (0, 1)])

    def test_zero_area_triangle_raises(self):
        """
        Collinear points should raise AssertionError due to zero area.
        This assertion is also raised if we have the same point multiple times.
        """
        with pytest.raises(AssertionError, match="area of the triangle has to be larger than zero"):
            TriangleGeometry("triangle", keep_inside=False, points=[(0, 0), (1, 1), (2, 2)])

    def test_points_as_tensors_allowed(self):
        """
        Passing torch tensors instead of tuples should work.
        """
        pts = [tensor([0.0, 0.0]), tensor([1.0, 0.0]), tensor([0.0, 1.0])]
        tri = TriangleGeometry("triangle", keep_inside=False, points=pts)
        assert tri.type == "triangle"

    @pytest.fixture
    def triangle_keep_inside_false(self):
        """
        Fixture providing a TriangleGeometry with vertices and ``keep_inside=False``:
        [(-1, -0.5), (0.25, 4), (1.5, -0.5)]
        """
        return TriangleGeometry("triangle", keep_inside=False, points=[(-1, -0.5), (0.25, 4), (1.5, -0.5)])

    @pytest.fixture
    def triangle_keep_inside_true(self):
        """
        Fixture providing a TriangleGeometry with vertices and ``keep_inside=True``.
        """
        return TriangleGeometry("triangle", keep_inside=True, points=[(-1, -0.5), (0.25, 4), (1.5, -0.5)])

    @pytest.fixture
    def dummy_cells(self):
        """
        Fixture providing reusable 2D/3D test cells.
        """
        return DummyCells()

    @pytest.mark.parametrize(
        "triangle_fixture, cell_attr, expected",
        [
            # keep_inside=False
            ("triangle_keep_inside_false", "cell_outside_2D", False),
            ("triangle_keep_inside_false", "cell_inside_2D", True),
            ("triangle_keep_inside_false", "cell_partially_2D", False),

            # keep_inside=True
            ("triangle_keep_inside_true", "cell_outside_2D", True),
            ("triangle_keep_inside_true", "cell_inside_2D", False),
            ("triangle_keep_inside_true", "cell_partially_2D", False),
        ]
    )
    def test_check_cell(self, request, dummy_cells, triangle_fixture, cell_attr, expected):
        """
        Test ``check_cell`` with both ``keep_inside=False`` and ``keep_inside=True``.
        """
        triangle = request.getfixturevalue(triangle_fixture)
        cell = getattr(dummy_cells, cell_attr)
        result = triangle.check_cell(cell)
        assert result is expected, (f"Expected '{expected}' for '{triangle_fixture}' with '{cell_attr}' "
                                    f"(coords={cell.tolist()}), got '{result}'.")

if __name__ == "__main__":
    pass

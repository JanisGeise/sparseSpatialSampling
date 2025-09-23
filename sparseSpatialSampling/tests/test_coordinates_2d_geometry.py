"""
Unit tests for ``GeometryCoordinates2D`` in 2D using the ``DummyCells`` class.
"""
import pytest
from torch import tensor

from ..geometry import GeometryCoordinates2D
from .import DummyCells


class TestGeometryCoordinates2D:
    @pytest.fixture
    def square_keep_inside_false(self):
        """
        Fixture providing GeometryCoordinates2D with ``keep_inside=False``.
        Coordinates form a simple square: [(-1, -1), (-1, 1.25), (1.25, 1.25), (1.25, -1)]
        """
        return GeometryCoordinates2D("square", keep_inside=False,
                                     coordinates=[(-1, -1), (-1, 1.25), (1.25, 1.25), (1.25, -1)])

    @pytest.fixture
    def square_keep_inside_true(self):
        """
        Fixture providing GeometryCoordinates2D with ``keep_inside=True``.
        The coordinates are the same as used in ``square_keep_inside_false``
        """
        return GeometryCoordinates2D("square", keep_inside=True,
                                     coordinates=[(-1, -1), (-1, 1.25), (1.25, 1.25), (1.25, -1)])

    @pytest.fixture
    def dummy_cells(self):
        """
        Fixture providing reusable 2D/3D test cells.
        """
        return DummyCells()

    @pytest.mark.parametrize(
        "square_fixture, cell_attr, expected",
        [
            # keep_inside=False
            ("square_keep_inside_false", "cell_outside_2D", False),
            ("square_keep_inside_false", "cell_inside_2D", True),
            ("square_keep_inside_false", "cell_partially_2D", False),

            # keep_inside=True
            ("square_keep_inside_true", "cell_outside_2D", True),
            ("square_keep_inside_true", "cell_inside_2D", False),
            ("square_keep_inside_true", "cell_partially_2D", False),
        ]
    )
    def test_check_cell(self, request, dummy_cells, square_fixture, cell_attr, expected):
        """
        Test ``check_cell`` with both ``keep_inside=False`` and ``keep_inside=True``.
        """
        square = request.getfixturevalue(square_fixture)
        cell = getattr(dummy_cells, cell_attr)
        result = square.check_cell(cell)
        assert result is expected, (f"Expected '{expected}' for '{square_fixture}' with '{cell_attr}' "
                                    f"(coords={cell.tolist()}), got '{result}'.")

    def test_pre_check_cell(self, square_keep_inside_false):
        """
        Test that ``pre_check_cell`` works correctly.
        """
        # point inside bounding box
        inside = tensor([[0.0, 0.0]])
        assert square_keep_inside_false.pre_check_cell(inside) is True

        # point outside bounding box
        outside = tensor([[5.0, 5.0]])
        assert square_keep_inside_false.pre_check_cell(outside) is False

if __name__ == "__main__":
    pass

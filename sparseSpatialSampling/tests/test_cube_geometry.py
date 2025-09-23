"""
Unit tests for ``CubeGeometry`` in 2D using the ``DummyCells`` class.
"""
import pytest
from torch import zeros

from ..geometry import CubeGeometry
from .import DummyCells


class TestCubeGeometry:
    @pytest.fixture
    def cube_2d_keep_inside_false(self):
        """
        2D rectangle ``CubeGeometry`` with ``keep_inside=False``.
        """
        return CubeGeometry("cube2D", keep_inside=False, lower_bound=[0.0, 0.0], upper_bound=[1.0, 1.0])

    @pytest.fixture
    def cube_2d_keep_inside_true(self):
        """
        2D rectangle ``CubeGeometry`` with ``keep_inside=True``.
        """
        return CubeGeometry("cube2D", keep_inside=True, lower_bound=[0.0, 0.0], upper_bound=[1.0, 1.0])

    @pytest.fixture
    def cube_3d_keep_inside_false(self):
        """
        3D cube ``CubeGeometry`` with ``keep_inside=False``.
        """
        return CubeGeometry("cube3D", keep_inside=False, lower_bound=[0.0, 0.0, 0.0], upper_bound=[1.0, 1.0, 1.0])

    @pytest.fixture
    def cube_3d_keep_inside_true(self):
        """
        3D cube ``CubeGeometry`` with ``keep_inside=True``.
        """
        return CubeGeometry("cube3D", keep_inside=True, lower_bound=[0.0, 0.0, 0.0], upper_bound=[1.0, 1.0, 1.0])

    @pytest.fixture
    def dummy_cells(self):
        """
        Reusable 2D/3D test cells.
        """
        return DummyCells()

    @pytest.mark.parametrize(
        "cube_fixture, cell_attr, expected",
        [
            # 2D keep_inside=False
            ("cube_2d_keep_inside_false", "cell_outside_2D", False),
            ("cube_2d_keep_inside_false", "cell_inside_2D", True),
            ("cube_2d_keep_inside_false", "cell_partially_2D", False),

            # 2D keep_inside=True
            ("cube_2d_keep_inside_true", "cell_outside_2D", True),
            ("cube_2d_keep_inside_true", "cell_inside_2D", False),
            ("cube_2d_keep_inside_true", "cell_partially_2D", False),

            # 3D keep_inside=False
            ("cube_3d_keep_inside_false", "cell_outside_3D", False),
            ("cube_3d_keep_inside_false", "cell_inside_3D", True),
            ("cube_3d_keep_inside_false", "cell_partially_3D", False),

            # 3D keep_inside=True
            ("cube_3d_keep_inside_true", "cell_outside_3D", True),
            ("cube_3d_keep_inside_true", "cell_inside_3D", False),
            ("cube_3d_keep_inside_true", "cell_partially_3D", False),
        ]
    )
    def test_check_cell(self, request, dummy_cells, cube_fixture, cell_attr, expected):
        """
        Test ``check_cell`` with both ``keep_inside=False`` and ``keep_inside=True``.
        """
        cube = request.getfixturevalue(cube_fixture)
        cell = getattr(dummy_cells, cell_attr)
        result = cube.check_cell(cell)
        assert result is expected, (f"Expected '{expected}' for '{cube_fixture}' with '{cell_attr}' "
                                    f"(coords={cell.tolist()}), got '{result}'.")

    def test_check_cell_wrong_dimensions(self, cube_2d_keep_inside_false):
        """
        check_cell should raise AssertionError if dimensions don't match bounds.
        """
        wrong_dim_cell = zeros((1, 3))
        with pytest.raises(AssertionError):
            cube_2d_keep_inside_false.check_cell(wrong_dim_cell)

    def test_bounds_assertions(self):
        """
        Test that ``CubeGeometry`` raises AssertionError for invalid bounds.
        """
        # Empty bounds
        with pytest.raises(AssertionError):
            CubeGeometry("empty_lower", keep_inside=True, lower_bound=[], upper_bound=[1, 1])
        with pytest.raises(AssertionError):
            CubeGeometry("empty_upper", keep_inside=True, lower_bound=[0, 0], upper_bound=[])

        # Mismatched lengths
        with pytest.raises(AssertionError):
            CubeGeometry("mismatch", keep_inside=True, lower_bound=[0, 0], upper_bound=[1, 1, 1])

        # Lower >= upper
        with pytest.raises(AssertionError):
            CubeGeometry("invalid", keep_inside=True, lower_bound=[0, 2], upper_bound=[1, 1])

if __name__ == "__main__":
    pass

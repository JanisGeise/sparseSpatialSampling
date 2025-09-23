"""
Unit tests for ``SphereGeometry`` in 2D using the ``DummyCells`` class.
"""
import pytest

from ..geometry import SphereGeometry
from .import DummyCells


class TestSphereGeometry:
    @pytest.fixture
    def circle_2d_keep_inside_false(self):
        """2D circle with ``keep_inside=False``.
        """
        return SphereGeometry("circle2D", keep_inside=False, position=[0.5, 0.5], radius=0.9)

    @pytest.fixture
    def circle_2d_keep_inside_true(self):
        """
        2D circle with ``keep_inside=True``.
        """
        return SphereGeometry("circle2D", keep_inside=True, position=[0.5, 0.5], radius=0.9)

    @pytest.fixture
    def sphere_3d_keep_inside_false(self):
        """
        3D sphere with ``keep_inside=False``.
        """
        return SphereGeometry("sphere3D", keep_inside=False, position=[0.5, 0.5, 0.5], radius=0.9)

    @pytest.fixture
    def sphere_3d_keep_inside_true(self):
        """
        3D sphere with ``keep_inside=True``.
        """
        return SphereGeometry("sphere3D", keep_inside=True, position=[0.5, 0.5, 0.5], radius=0.9)

    @pytest.fixture
    def dummy_cells(self):
        """Reusable 2D/3D test cells."""
        return DummyCells()

    @pytest.mark.parametrize(
        "sphere_fixture, cell_attr, expected",
        [
            # 2D keep_inside=False
            ("circle_2d_keep_inside_false", "cell_outside_2D", False),
            ("circle_2d_keep_inside_false", "cell_inside_2D", True),
            ("circle_2d_keep_inside_false", "cell_partially_2D", False),

            # 2D keep_inside=True
            ("circle_2d_keep_inside_true", "cell_outside_2D", True),
            ("circle_2d_keep_inside_true", "cell_inside_2D", False),
            ("circle_2d_keep_inside_true", "cell_partially_2D", False),

            # 3D keep_inside=False
            ("sphere_3d_keep_inside_false", "cell_outside_3D", False),
            ("sphere_3d_keep_inside_false", "cell_inside_3D", True),
            ("sphere_3d_keep_inside_false", "cell_partially_3D", False),

            # 3D keep_inside=True
            ("sphere_3d_keep_inside_true", "cell_outside_3D", True),
            ("sphere_3d_keep_inside_true", "cell_inside_3D", False),
            ("sphere_3d_keep_inside_true", "cell_partially_3D", False),
        ]
    )
    def test_check_cell(self, request, dummy_cells, sphere_fixture, cell_attr, expected):
        """
        Test ``check_cell`` with both ``keep_inside=False`` and ``keep_inside=True``.
        """
        sphere = request.getfixturevalue(sphere_fixture)
        cell = getattr(dummy_cells, cell_attr)
        result = sphere.check_cell(cell)
        assert result is expected, (f"Expected '{expected}' for '{sphere_fixture}' with '{cell_attr}' "
                                    f"(coords={cell.tolist()}), got '{result}'.")

    def test_check_cell_wrong_dimensions(self, circle_2d_keep_inside_false):
        """
        check_cell should raise AssertionError if dimensions don't match the center position.
        """
        import torch
        wrong_dim_cell = torch.zeros((1, 3))  # 2D circle expects 2D points
        with pytest.raises(AssertionError):
            circle_2d_keep_inside_false.check_cell(wrong_dim_cell)

    def test_geometry_assertions(self):
        """
        Test that ``SphereGeometry`` raises AssertionError for invalid inputs.
        """

        # Empty position
        with pytest.raises(AssertionError):
            SphereGeometry("empty_pos", keep_inside=True, position=[], radius=1.0)

        # Invalid radius type
        with pytest.raises(AssertionError):
            SphereGeometry("invalid_radius_type", keep_inside=True, position=[0, 0], radius="1")

        # Non-positive radius
        with pytest.raises(AssertionError):
            SphereGeometry("zero_radius", keep_inside=True, position=[0, 0], radius=0)

if __name__ == "__main__":
    pass

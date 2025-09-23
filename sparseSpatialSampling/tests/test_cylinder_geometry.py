"""
Unit tests for ``CylinderGeometry3D`` in 3D using the ``DummyCells`` class.
"""
import pytest

from ..geometry import CylinderGeometry3D
from .const import DummyCells


class TestCylinderGeometry3D:
    @pytest.fixture
    def cylinder_keep_inside_false(self):
        """
        Cylinder along z-axis with ``radius = 0.5``, ``keep_inside=False``.
        """
        position = [[0.5, 0.5, -0.5], [0.5, 0.5, 1.25]]
        radius = 0.8
        return CylinderGeometry3D("cylinder", keep_inside=False, position=position, radius=radius)

    @pytest.fixture
    def cone_keep_inside_true(self):
        """
        Cylinder along z-axis with ``radius = 0.5``, ``keep_inside=True``.
        """
        position = [[0.5, 0.5, -0.5], [0.5, 0.5, 1.25]]
        radius = 0.8
        return CylinderGeometry3D("cylinder", keep_inside=True, position=position, radius=radius)

    @pytest.fixture
    def dummy_cells(self):
        return DummyCells()

    @pytest.mark.parametrize(
        "geom_fixture, cell_attr, expected",
        [
            # Cylinder keep_inside=False
            ("cylinder_keep_inside_false", "cell_outside_3D", False),
            ("cylinder_keep_inside_false", "cell_inside_3D", True),
            ("cylinder_keep_inside_false", "cell_partially_3D", False),

            # Cone keep_inside=True
            ("cone_keep_inside_true", "cell_outside_3D", True),
            ("cone_keep_inside_true", "cell_inside_3D", False),
            ("cone_keep_inside_true", "cell_partially_3D", False),
        ]
    )

    def test_check_cell(self, request, dummy_cells, geom_fixture, cell_attr, expected):
        """
        Test ``check_cell`` with both ``keep_inside=False`` and ``keep_inside=True``.
        """
        geom = request.getfixturevalue(geom_fixture)
        cell = getattr(dummy_cells, cell_attr)
        result = geom.check_cell(cell)
        assert result is expected, (f"Expected '{expected}' for '{geom_fixture}' with '{cell_attr}' "
                                    f"(coords={cell.tolist()}), got '{result}'.")

    def test_invalid_radius(self):
        """
        Ensure invalid radius definitions raise AssertionError.
        """
        # negative radius
        with pytest.raises(AssertionError):
            CylinderGeometry3D("bad_cylinder", True, position=[[0,0,0],[0,0,1]], radius=-1)

        # zero-length cylinder
        with pytest.raises(AssertionError):
            CylinderGeometry3D("bad_cylinder2", True, position=[[0,0,0],[0,0,0]], radius=0.5)

        # two radii both zero
        with pytest.raises(AssertionError):
            CylinderGeometry3D("bad_cone", True, position=[[0,0,0],[0,0,1]], radius=[0,0])

if __name__ == "__main__":
    pass

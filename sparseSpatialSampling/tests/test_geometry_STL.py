import pytest
from os.path import join

from ..geometry import GeometrySTL3D
from .const import DummyCells

class TestGeometrySTL3D:
    @pytest.fixture
    def cube_keep_inside_false(self):
        """
        Fixture providing a cube geometry from an STL file with ``keep_inside=False``.
        """
        return GeometrySTL3D("cube", keep_inside=False, path_stl_file=join("sparseSpatialSampling", "tests",
                                                                           "cube.stl"))

    @pytest.fixture
    def cube_keep_inside_true(self):
        """
        Fixture providing a cube geometry from an STL file with ``keep_inside=True``.
        """
        return GeometrySTL3D("cube", keep_inside=True, path_stl_file=join("sparseSpatialSampling", "tests",
                                                                          "cube.stl"))

    @pytest.fixture
    def dummy_cells(self):
        """
        Fixture providing reusable 3D test cells.
        """
        return DummyCells()

    @pytest.mark.parametrize(
        "cube_fixture, cell_attr, expected",
        [
            # keep_inside=False
            ("cube_keep_inside_false", "cell_outside_3D", False),
            ("cube_keep_inside_false", "cell_inside_3D", True),
            ("cube_keep_inside_false", "cell_partially_3D", False),

            # keep_inside=True
            ("cube_keep_inside_true", "cell_outside_3D", True),
            ("cube_keep_inside_true", "cell_inside_3D", False),
            ("cube_keep_inside_true", "cell_partially_3D", False),
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

    def test_pre_check_cell(self, cube_keep_inside_false, dummy_cells):
        """
        Test that `pre_check_cell` works correctly for STL files.
        """
        inside = dummy_cells.cell_inside_3D
        outside = dummy_cells.cell_outside_3D

        assert cube_keep_inside_false.pre_check_cell(inside) is True
        assert cube_keep_inside_false.pre_check_cell(outside) is False

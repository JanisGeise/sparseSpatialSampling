"""
Unit tests for ``TetrahedronGeometry3D`` in 3D using the ``DummyCells`` class.
"""
import pytest

from ..geometry import TetrahedronGeometry3D
from .const import DummyCells


class TestTetrahedronGeometry3D:
    @pytest.fixture
    def tetra_keep_inside_false(self):
        """
        Tetrahedron with ``keep_inside=False``.
        """
        positions = [[-1.5, 0.5, -0.1], [1.5, -1.5, -0.1], [1.5, 2.5, -0.1], [0.5, 0.5, 3]]
        return TetrahedronGeometry3D("tetra", keep_inside=False, positions=positions)

    @pytest.fixture
    def tetra_keep_inside_true(self):
        """
        Tetrahedron with ``keep_inside=True``.
        """
        positions = [[-1.5, 0.5, -0.1], [1.5, -1.5, -0.1], [1.5, 2.5, -0.1], [0.5, 0.5, 3]]
        return TetrahedronGeometry3D("tetra", keep_inside=True, positions=positions)

    @pytest.fixture
    def dummy_cells(self):
        return DummyCells()

    @pytest.mark.parametrize(
        "tetra_fixture, cell_attr, expected",
        [
            # keep_inside=False
            ("tetra_keep_inside_false", "cell_outside_3D", False),
            ("tetra_keep_inside_false", "cell_inside_3D", True),
            ("tetra_keep_inside_false", "cell_partially_3D", False),

            # keep_inside=True
            ("tetra_keep_inside_true", "cell_outside_3D", True),
            ("tetra_keep_inside_true", "cell_inside_3D", False),
            ("tetra_keep_inside_true", "cell_partially_3D", False),
        ]
    )
    def test_check_cell(self, request, dummy_cells, tetra_fixture, cell_attr, expected):
        """
        Test ``check_cell`` with both ``keep_inside=False`` and ``keep_inside=True``.
        """
        tetra = request.getfixturevalue(tetra_fixture)
        cell = getattr(dummy_cells, cell_attr)
        result = tetra.check_cell(cell)
        assert result is expected, (f"Expected '{expected}' for '{tetra_fixture}' with '{cell_attr}' "
                                    f"(coords={cell.tolist()}), got '{result}'.")

    def test_invalid_positions(self):
        """
        Ensure invalid tetrahedron definitions raise AssertionError.
        """
        # empty list
        with pytest.raises(AssertionError):
            TetrahedronGeometry3D("bad_tetra", True, positions=[])

        # wrong number of vertices
        with pytest.raises(AssertionError):
            TetrahedronGeometry3D(
                "bad_tetra2", True,
                positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]  # only 3 vertices
            )

        # vertex with wrong number of coordinates
        with pytest.raises(AssertionError):
            TetrahedronGeometry3D(
                "bad_tetra3", True,
                positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5]]  # last vertex has 2 coordinates
            )

if __name__ == "__main__":
    pass

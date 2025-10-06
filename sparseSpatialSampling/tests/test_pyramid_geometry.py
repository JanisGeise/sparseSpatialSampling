"""
Unit tests for ``PyramidGeometry3D`` in 3D using the ``DummyCells`` class.
"""
import pytest

from ..geometry import PyramidGeometry3D
from .const import DummyCells


class TestPyramidGeometry3D:
    @pytest.fixture
    def pyramid_keep_inside_false(self):
        """
        Pyramid with ``keep_inside=False``.
        """
        nodes = [
            [-1, -1, -0.25],
            [2, -1, -0.25],
            [2, 2, -0.25],
            [-1, 2, -0.25],
            [0.5, 0.5, 3],
        ]
        return PyramidGeometry3D("pyramid", keep_inside=False, nodes=nodes)

    @pytest.fixture
    def pyramid_keep_inside_true(self):
        """
        Pyramid with ``keep_inside=True``.
        """
        nodes = [
            [-1, -1, -0.25],
            [2, -1, -0.25],
            [2, 2, -0.25],
            [-1, 2, -0.25],
            [0.5, 0.5, 3],
        ]
        return PyramidGeometry3D("pyramid", keep_inside=True, nodes=nodes)

    @pytest.fixture
    def dummy_cells(self):
        return DummyCells()

    @pytest.mark.parametrize(
        "pyramid_fixture, cell_attr, expected",
        [
            # keep_inside=False
            ("pyramid_keep_inside_false", "cell_outside_3D", False),
            ("pyramid_keep_inside_false", "cell_inside_3D", True),
            ("pyramid_keep_inside_false", "cell_partially_3D", False),

            # keep_inside=True
            ("pyramid_keep_inside_true", "cell_outside_3D", True),
            ("pyramid_keep_inside_true", "cell_inside_3D", False),
            ("pyramid_keep_inside_true", "cell_partially_3D", False),
        ]
    )
    def test_check_cell(self, request, dummy_cells, pyramid_fixture, cell_attr, expected):
        """
        Test ``check_cell`` with both ``keep_inside=False`` and ``keep_inside=True``.
        """
        pyramid = request.getfixturevalue(pyramid_fixture)
        cell = getattr(dummy_cells, cell_attr)
        result = pyramid.check_cell(cell)
        assert result is expected, (
            f"Expected '{expected}' for '{pyramid_fixture}' with '{cell_attr}' "
            f"(coords={cell.tolist()}), got '{result}'."
        )

    def test_invalid_nodes(self):
        """
        Ensure invalid pyramid definitions raise AssertionError.
        """
        # empty list
        with pytest.raises(AssertionError):
            PyramidGeometry3D("bad_pyramid", True, nodes=[])

        # wrong number of vertices
        with pytest.raises(AssertionError):
            PyramidGeometry3D(
                "bad_pyramid2", True,
                nodes=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 1]]
            )

        # vertex with wrong number of coordinates
        with pytest.raises(AssertionError):
            PyramidGeometry3D(
                "bad_pyramid3", True,
                nodes=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 1], [0.5, 0.5]]
            )


if __name__ == "__main__":
    pass

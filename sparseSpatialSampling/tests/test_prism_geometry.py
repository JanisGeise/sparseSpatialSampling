"""
Unit tests for ``PrismGeometry3D`` in 3D using the ``DummyCells`` class.
"""
import pytest

from ..geometry import PrismGeometry3D
from .const import DummyCells


class TestPrismGeometry3D:
    @pytest.fixture
    def prism_keep_inside_false(self):
        """
        Fixture providing a prism with ``keep_inside=False``.

        Prism defined by two triangles along the z-axis:
            - bottom triangle: [(-1, -0.5, -0.5), (0.25, 4, -0.5), (1.5, -0.5, -0.5)]
            - top triangle:    [(-1, -0.5, 1.25), (0.25, 4, 1.25), (1.5, -0.5, 1.25)]
        """
        positions = [[(-1, -0.5, -0.5), (0.25, 4, -0.5), (1.5, -0.5, -0.5)],
                     [(-1, -0.5, 1.25), (0.25, 4, 1.25), (1.5, -0.5, 1.25)]]
        return PrismGeometry3D("prism", keep_inside=False, positions=positions)

    @pytest.fixture
    def prism_keep_inside_true(self):
        """
        Fixture providing a prism with ``keep_inside=True``.
        The location of the prism is the same as in ``prism_keep_inside_false``
        """
        positions = [[(-1, -0.5, -0.5), (0.25, 4, -0.5), (1.5, -0.5, -0.5)],
                     [(-1, -0.5, 1.25), (0.25, 4, 1.25), (1.5, -0.5, 1.25)]]
        return PrismGeometry3D("prism", keep_inside=True, positions=positions)

    @pytest.fixture
    def dummy_cells(self):
        return DummyCells()

    @pytest.mark.parametrize(
        "prism_fixture, cell_attr, expected",
        [
            # keep_inside=False
            ("prism_keep_inside_false", "cell_outside_3D", False),
            ("prism_keep_inside_false", "cell_inside_3D", True),
            ("prism_keep_inside_false", "cell_partially_3D", False),

            # keep_inside=True
            ("prism_keep_inside_true", "cell_outside_3D", True),
            ("prism_keep_inside_true", "cell_inside_3D", False),
            ("prism_keep_inside_true", "cell_partially_3D", False),
        ]
    )
    def test_check_cell(self, request, dummy_cells, prism_fixture, cell_attr, expected):
        """
        Test ``check_cell`` with both ``keep_inside=False`` and ``keep_inside=True``.
        """
        prism = request.getfixturevalue(prism_fixture)
        cell = getattr(dummy_cells, cell_attr)
        result = prism.check_cell(cell)
        assert result is expected, (f"Expected '{expected}' for '{prism_fixture}' with '{cell_attr}' "
                                    f"(coords={cell.tolist()}), got '{result}'.")

    def test_invalid_positions(self):
        """
        Ensure that invalid prism definitions raise AssertionError.
        """
        # Not two triangles
        with pytest.raises(AssertionError):
            PrismGeometry3D("bad_prism", True, positions=[[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])

        # Triangle with wrong number of vertices
        with pytest.raises(AssertionError):
            PrismGeometry3D(
                "bad_prism2", True,
                positions=[
                    [[0, 0, 0], [1, 0, 0]],   # only 2 vertices
                    [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
                ]
            )

if __name__ == "__main__":
    pass

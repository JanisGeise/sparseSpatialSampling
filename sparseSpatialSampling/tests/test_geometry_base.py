"""
Unit tests for the GeometryObject base class.

This module provides tests to ensure that the base geometry class behaves as expected.
It includes:

- Validation of constructor arguments (e.g., ``name``, ``keep_inside``, ``min_refinement_level``).
- Automatic adjustment of the refine flag when a minimum refinement level is provided.
- Correct behavior of the `_apply_mask` method for both ``keep_inside=True`` and ``keep_inside=False``.
"""
import pytest
from torch import tensor

from ..geometry.geometry_base import GeometryObject

class DummyGeometry(GeometryObject):
    """
    A minimal subclass of GeometryObject for unit testing.

    Implements abstract methods with pass statements so that
    the base class functionality can be tested without specific geometry logic.
    """

    def check_cell(self, cell_nodes, refine_geometry: bool = False):
        """
        Dummy implementation of check_cell (does nothing).
        """
        pass

    def _check_geometry(self):
        """
        Dummy implementation of _check_geometry (does nothing).
        """
        pass

    def type(self):
        """
        Dummy implementation of type (does nothing).
        """
        pass

    @property
    def main_width(self):
        """
        Dummy implementation of type (does nothing).
        """
        return None

    @property
    def center(self):
        return None

    def _compute_main_width(self):
        """
        Dummy implementation of type (does nothing).
        """
        pass

    def _compute_center(self):
        """
        Dummy implementation of type (does nothing).
        """
        pass

class TestGeometryBase:
    """
    Unit tests for the GeometryObject base class.

    This test class validates:
        - Constructor input checks
        - Automatic refine flag behavior
        - Minimum refinement level validation
        - Correct ``_apply_mask`` logic for both ``keep_inside=True`` and ``keep_inside=False``
    """
    @pytest.fixture
    def check_keep_inside(self) -> DummyGeometry:
        """
        Fixture returning a ``DummyGeometry`` instance with ``keep_inside=True``.

        :return: Dummy geometry object with points kept inside
        :rtype: DummyGeometry
        """
        return DummyGeometry("inside", keep_inside=True, refine=False)

    @pytest.fixture
    def check_keep_outside(self) -> DummyGeometry:
        """
        Fixture returning a ``DummyGeometry`` instance with ``keep_inside=False``.

        :return: Dummy geometry object with points masked outside
        :rtype: DummyGeometry
        """
        return DummyGeometry("outside", keep_inside=False, refine=True, min_refinement_level=1)

    def test_name_cannot_be_empty(self) -> None:
        """
        Ensure that creating a geometry with an empty name raises an AssertionError.

        :raises AssertionError: if the geometry name is empty
        """
        with pytest.raises(AssertionError):
            DummyGeometry("", True)

    def test_keep_inside_must_be_bool(self) -> None:
        """
        Ensure that the ``keep_inside`` parameter must be a boolean.

        :raises AssertionError: if ``keep_inside`` is not a bool
        """
        with pytest.raises(AssertionError):
            DummyGeometry("issue", "not_bool")

    def test_refine_auto_set_if_min_refinement_given(self) -> None:
        """
        Ensure that the refine flag is automatically set to ``True`` if ``min_refinement_level`` is provided.
        """
        dummy = DummyGeometry("set_ref_level", True, refine=False, min_refinement_level=1)
        assert dummy.refine is True

    def test_min_refinement_less_one(self) -> None:
        with pytest.raises(AssertionError):
            DummyGeometry("check_ref_level", True, refine=True, min_refinement_level=0)

    @pytest.mark.parametrize(
        "test_name, refine_geometry, mask, expected",
        [
            # since we are testing the mask directly, each node is assigned to a bool
            # test if a cell is masked correctly as invalid if completely outside a domain
            ("check_keep_inside", False, tensor([False, False, False, False]), True),

            # test if a cell is masked correctly as valid if completely inside a domain
            ("check_keep_inside", False, tensor([True, True, True, True]), False),

            # test if a cell is masked correctly as valid if partially inside a domain
            ("check_keep_inside", False, tensor([True, False, False, True]), False),

            # test if a cell is masked correctly as invalid if completely inside a geometry
            ("check_keep_outside", False, tensor([True, True, True, True]), True),

            # test if a cell is masked correctly as valid if partially inside a geometry
            ("check_keep_outside", False, tensor([True, False, False, True]), False),

            # test if a cell is masked correctly as valid if outside a geometry
            ("check_keep_outside", False, tensor([False, False, False, False]), False),
        ]
    )

    def test_apply_mask_logic(self, request, test_name, refine_geometry, mask, expected) -> None:
        """
        Parameterized test to verify ``_apply_mask`` behavior for different masks and ``keep_inside`` settings.

        :param request: pytest fixture request object used to retrieve other fixtures
        :type request: FixtureRequest
        :param test_name: Name of the fixture to retrieve (``check_keep_inside`` or ``check_keep_outside``)
        :type test_name: str
        :param refine_geometry: Flag indicating whether the grid is being refined
        :type refine_geometry: bool
        :param mask: Boolean tensor representing which vertices are inside the geometry
        :type mask: torch.Tensor
        :param expected: Expected boolean result after applying the mask
        :type expected: bool
        """
        geom = request.getfixturevalue(test_name)
        result = geom._apply_mask(mask, refine_geometry)
        assert result == expected, (
            f"_apply_mask returned '{result}' for geometry '{test_name}' "
            f"with mask={mask.tolist()} and refine_geometry={refine_geometry}, "
            f"but expected '{expected}'."
        )


if __name__ == "__main__":
    pass

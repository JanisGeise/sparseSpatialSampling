"""
Implements a common base class for geometry objects from which all other geometry objects should be derived.

"""
import logging

from torch import Tensor
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)


class GeometryObject(ABC):
    def __init__(self, name: str, keep_inside: bool, refine: bool = False, min_refinement_level: int = None):
        """
        Implement the base class for geometry objects from which all other geometry objects should be derived.

        :param name: Name of the geometry object.
        :type name: str
        :param keep_inside: If ``True``, the points inside the object are kept; if ``False``, they are masked out.
        :type keep_inside: bool
        :param refine: If ``True``, the mesh around the geometry object is refined after :math:`S^3` generates the mesh.
        :type refine: bool
        :param min_refinement_level: Minimum refinement level for resolving the geometry. If ``None`` and
            ``refine=True``, the geometry will be resolved with the maximum refinement level present at its surface
            after :math:`S^3` has generated the grid.
        :type min_refinement_level: int | None
        """
        self._name = name
        self._keep_inside = keep_inside
        self._refine = refine
        self._min_refinement_level = min_refinement_level

        # check the arguments which should be present for all geometry objects
        self._check_common_arguments()

    def _apply_mask(self, mask: Tensor, refine_geometry: bool) -> bool:
        """
        Check if a given cell is invalid based on a provided mask and settings.

        This method returns ``False`` if the cell is valid, and ``True`` if it is invalid.

        Note:
            It is expected that the mask passed into the `_apply_mask` method is always ``False`` outside
            the mask and always ``True`` inside it (regardless of whether it is a geometry or domain).

        :param mask: Mask created by the geometry object.
        :type mask: pt.Tensor
        :param refine_geometry: If ``False``, cells are masked out while generating the grid.
            If ``True``, checks whether a cell is located in the vicinity of the geometry surface
            to refine it subsequently. This parameter is provided by :math:`S^3`.
        :type refine_geometry: bool
        :return: ``True`` if the cell is invalid, ``False`` if the cell is valid.
        :rtype: bool
        """
        if not refine_geometry:
            # any(~mask), because mask returns 'False' if we are outside, but we want 'True' if we are outside
            if not self._keep_inside:
                invalid = mask.all(0)

            # if we are outside the domain, we want to return 'False'
            else:
                invalid = ~mask.any(0)

        # otherwise, we want to refine all cells that have at least one node in the geometry / outside the domain.
        else:
            # add all cells that have at least one node inside the geometry
            if not self._keep_inside:
                invalid = mask.any(0)
            else:
                invalid = ~mask.all(0)

        return invalid.item()

    def _check_common_arguments(self) -> None:
        """
        Check the user input for correctness.

        This method validates common arguments of the geometry object and raises
        errors if any input is invalid.

        :return: None
        :rtype: None
        """

        # check if name is empty string
        assert self._name != "", "Found empty string for the geometry object name. Please provide a name."

        # check if keep_inside is bool
        assert isinstance(self._keep_inside, bool), (f"Invalid type for argument keep_inside. Expected bool but "
                                                 f"{type(self._keep_inside)} was given.")

        # in case a min_refinement_level is defined, but refine is kept False, we assume that the geometry should be
        # refined and set refine automatically to True
        if not self._refine and self._min_refinement_level is not None:
            logger.warning(f"Found value refine={self._refine} while a min_refinement_level of "
                           f"{self._min_refinement_level} was provided for geometry {self._name}. Changing refine from"
                           f" {self._refine} to refine=True.")
            self._refine = True

        # make sure the refinement level is >= 1
        if self._refine and self._min_refinement_level is not None:
            assert self._min_refinement_level > 0, (f"Expected min_refinement_level > 0 but found "
                                                    f"min_refinement_level={self.min_refinement_level}.")

    @property
    def keep_inside(self):
        """
        Get the `keep_inside` flag for the geometry object.

        :return: ``True`` if points inside the object are kept; ``False`` if they are masked out.
        :rtype: bool
        """
        return self._keep_inside

    @property
    def name(self):
        """
        Get the name of the geometry object.

        :return: Name of the geometry object.
        :rtype: str
        """
        return self._name

    @property
    def refine(self):
        """
        Get the `refine` flag for the geometry object.

        :return: ``True`` if the mesh around the geometry object should be refined; ``False`` otherwise.
        :rtype: bool
        """
        return self._refine

    @property
    def min_refinement_level(self):
        """
        Get the minimum refinement level for resolving the geometry.

        :return: Minimum refinement level, or ``None`` if not explicitly set.
        :rtype: int | None
        """
        return self._min_refinement_level

    @abstractmethod
    def check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> bool:
        """
        Check if a cell is valid or invalid based on the specified settings.

        :param cell_nodes: Vertices of the cell to be checked.
        :type cell_nodes: pt.Tensor
        :param refine_geometry: If ``False``, cells are masked out while generating the grid.
            If ``True``, checks whether a cell is located in the vicinity of the geometry surface
            to refine it subsequently. This parameter is provided by :math:`S^3`.
        :type refine_geometry: bool
        :return: ``True`` if the cell is invalid, ``False`` if the cell is valid.
        :rtype: bool
        """
        pass

    @abstractmethod
    def _check_geometry(self) -> None:
        """
        Check the user input for correctness.
        """
        pass

    @abstractmethod
    def type(self) -> str:
        """
        Return the name of the geometry object.

        :return: Name of the geometry object.
        :rtype: str
        """
        pass


if __name__ == "__main__":
    pass

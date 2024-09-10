"""
    implements the base class for geometry objects from which all geometry objects should be derived

    Note: it is expected that the mask passed into the _apply_mask method is always False outside the geometry and
          always True inside it (independently if it is a geometry or domain)
"""
import logging

from torch import Tensor
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GeometryObject(ABC):
    def __init__(self, name: str, keep_inside: bool, refine: bool = False, min_refinement_level: int = None):
        """
        implements the base class for geometry objects from which all geometry objects should be derived

        :param name: name of the geometry object
        :type name: str
        :param keep_inside: flag if the points inside the object should be masked out (False) or kept (True)
        :type keep_inside: bool
        :param refine: flag if the mesh around the geometry object should be refined after S^3 generated the mesh
        :type refine: bool
        :param min_refinement_level: option to define a min. refinement level with which the geometry should be
                                     resolved; if 'None' and 'refine = True' the geometry will be resolved with the max.
                                     refinement level present at its surface after S^3 has generated the grid
        :type min_refinement_level: int
        """
        self._name = name
        self._keep_inside = keep_inside
        self._refine = refine
        self._min_refinement_level = min_refinement_level

        # check the arguments which should be present for all geometry objects
        self._check_common_arguments()

    def _apply_mask(self, mask: Tensor, refine_geometry: bool) -> bool:
        """
        check if a given cell is valid or invalid based onn a given mask and settings

        Note: it is expected that the mask passed into the _apply_mask method is always False outside the geometry and
              always True inside it (independently if it is a geometry or domain)

        :param mask: mask created by the geometry object
        :type mask: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: bool
        :return: flag if the cell is valid or invalid based on the defined settings
        :rtype: bool
        """
        if not refine_geometry:
            # any(~mask), because mask returns False if we are outside, but we want True if we are outside
            if not self._keep_inside:
                if any(~mask):
                    invalid = False
                else:
                    invalid = True

            # if we are outside the domain, we want to return False
            else:
                if any(mask):
                    invalid = False
                else:
                    invalid = True

        # otherwise, we want to refine all cells that have at least one node in the geometry / outside the domain.
        else:
            # add all cells that have at least one node inside the geometry
            if not self._keep_inside:
                if any(mask):
                    invalid = True
                else:
                    invalid = False
            else:
                if all(~mask):
                    invalid = True
                else:
                    invalid = False

        return invalid

    def _check_common_arguments(self) -> None:
        """
        method to check the user input for correctness
        """
        # check if name is empty string
        assert self._name != "", "Found emtpy string for the geometry object name. Please provide a name."

        # check if keep_inside is bool
        assert type(self._keep_inside) is bool, (f"Invalid type for argument keep_inside. Expected bool but "
                                                 f"{type(self._keep_inside)} was given.")

        # in case a min_refinement_level is defined, but refine is kept False, we assume that the geometry should be
        # refined and set refine automatically to True
        if not self._refine and self._min_refinement_level is not None:
            logger.warning(f"Found value refine={self._refine} while a min_refinement_level of "
                           f"{self._min_refinement_level} was provided for geometry {self._name}. Changing refine from"
                           f" {self._refine} to refine=True.")

    @property
    def keep_inside(self):
        return self._keep_inside

    @property
    def name(self):
        return self._name

    @property
    def refine(self):
        return self._refine

    @property
    def min_refinement_level(self):
        return self._min_refinement_level

    @abstractmethod
    def check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> bool:
        """
        method to check if a cell is valid or invalid based on the specified settings

        :param cell_nodes: vertices of the cell which should be checked
        :type cell_nodes: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: bool
        :return: flag if the cell is valid or invalid based on the specified settings
        :rtype: bool
        """
        pass

    @abstractmethod
    def _check_geometry(self) -> None:
        """
        method to check the user input for correctness
        """
        pass


if __name__ == "__main__":
    pass

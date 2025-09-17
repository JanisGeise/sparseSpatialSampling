"""
 implements a class for using rectangles (2D) or cubes (3D) as geometry object
"""
from torch import Tensor
from flowtorch.data import mask_box

from .geometry_base import GeometryObject


class CubeGeometry(GeometryObject):
    __short_description__ = "rectangles (2D) or cubes (3D)"

    def __init__(self, name: str, keep_inside: bool, lower_bound: list, upper_bound: list, refine: bool = False,
                 min_refinement_level: int = None):
        """
        implements a class for using rectangles (2D) or cubes (3D) as geometry objects representing the numerical
        domain or geometries inside the domain

        :param name: name of the geometry object
        :type name: str
        :param keep_inside: flag if the points inside the object should be masked out (False) or kept (True)
        :type name: bool
        :param lower_bound: lower boundaries of the rectangle or cube, sorted as [x_min, y_min, z_min]
        :type lower_bound: list
        :param upper_bound: upper boundaries of the rectangle or cube, sorted as [x_max, y_max, z_max]
        :type upper_bound: list
        :param refine: flag if the mesh around the geometry object should be refined after S^3 generated the mesh
        :type refine: bool
        :param min_refinement_level: option to define a min. refinement level with which the geometry should be
                                     resolved; if 'None' and 'refine = True' the geometry will be resolved with the max.
                                     refinement level present at its surface after S^3 has generated the grid
        :type min_refinement_level: int
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._type = "cube"

        # check the user input based on the specified settings
        self._check_geometry()

    def check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> Tensor:
        """
        method to check if a cell is valid or invalid based on the specified settings

        :param cell_nodes: vertices of the cell which should be checked
        :type cell_nodes: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: bool
        :return: flag if the cell is valid ('False') or invalid ('True') based on the specified settings
        :rtype: bool
        """
        # check if the number of boundaries matches the number of physical dimensions;
        # this can't be done beforehand because we don't know the number of physical dimensions yet
        assert cell_nodes.size(-1) == len(self._lower_bound), (f"Number of dimensions of the cell does not match the "
                                                               f"number of given bounds. Expected "
                                                               f"{cell_nodes.size(-1)} values, found "
                                                               f"{len(self._lower_bound)} for geometry {self.name}.")
        # create a mask, the mask is expected to be always 'False' outside the geometry and always 'True' inside it
        # (independently if it is a geometry or domain)
        mask = mask_box(cell_nodes, self._lower_bound, self._upper_bound)

        # check if the cell is valid or invali
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _check_geometry(self) -> None:
        """
        method to check the user input for correctness

        :return: None
        :rtype: None
        """
        # check is boundaries are empty list
        assert self._lower_bound, "Found empty list for the lower bound. Please provide values for the lower bound."
        assert self._upper_bound, "Found empty list for the upper bound. Please provide values for the upper bound."

        # check if the number of values for the lower boundary is the same as for the upper boundary
        assert len(self._lower_bound) == len(self._upper_bound), (f"The number of provided boundaries for the lower "
                                                                  f"bound does not match the number of boundaries for "
                                                                  f"the upper bound. Found {len(self._lower_bound)} "
                                                                  f"values for the lower bound but "
                                                                  f"{len(self._upper_bound)} values for the upper "
                                                                  f"bound for geometry {self.name}.")

        # check if the lower boundary is smaller than the upper boundary
        for i, v in enumerate(zip(self._lower_bound, self._upper_bound)):
            assert v[0] < v[1], (f"Value of {v[0]} for the lower bound at position {i} is larger or equal than the "
                                 f"value of {v[1]} for the upper bound for geometry {self.name}. The the lower bound "
                                 f"must be smaller than the upper bound!")

    @property
    def type(self) -> str:
        """
        returns name of the geometry object

        :return: name of the geometry object
        :rtype: str
        """
        return self._type
"""
Implement a class for using rectangles (2D) or cubes (3D) as geometry object.
"""
from torch import Tensor
from flowtorch.data import mask_box

from .geometry_base import GeometryObject


class CubeGeometry(GeometryObject):
    __short_description__ = "rectangles (2D) or cubes (3D)"

    def __init__(self, name: str, keep_inside: bool, lower_bound: list, upper_bound: list, refine: bool = False,
                 min_refinement_level: int = None):
        """
        Implement a class for using rectangles (2D) or cubes (3D) as geometry objects
        representing the numerical domain or geometries inside the domain.

        :param name: Name of the geometry object.
        :type name: str
        :param keep_inside: If ``True``, the points inside the object are kept; if ``False``,
            they are masked out.
        :type keep_inside: bool
        :param lower_bound: Lower boundaries of the rectangle or cube, sorted as
            ``[x_min, y_min, z_min]``.
        :type lower_bound: list
        :param upper_bound: Upper boundaries of the rectangle or cube, sorted as
            ``[x_max, y_max, z_max]``.
        :type upper_bound: list
        :param refine: If ``True``, the mesh around the geometry object is refined after
            :math:`S^3` generates the mesh.
        :type refine: bool
        :param min_refinement_level: Minimum refinement level for resolving the geometry.
            If ``None`` and ``refine=True``, the geometry is resolved with the maximum
            refinement level present at its surface after :math:`S^3` has generated the grid.
        :type min_refinement_level: int or None
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._type = "cube"

        # check the user input based on the specified settings
        self._check_geometry()

    def check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> Tensor:
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
        Check the user input for correctness.

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
        Return the name of the geometry object.

        :return: Name of the geometry object.
        :rtype: str
        """
        return self._type
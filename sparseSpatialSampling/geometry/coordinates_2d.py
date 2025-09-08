"""
    implements a class for using 2D coordinates as geometry object
"""
from numpy import ndarray
from typing import Union
from shapely import Point, Polygon
from torch import Tensor, tensor

from flowtorch.data import mask_box

from .geometry_base import GeometryObject


class GeometryCoordinates2D(GeometryObject):
    """
    implements a class for using 2D coordinates
    """
    def __init__(self, name: str, keep_inside: bool, coordinates: Union[list, ndarray], refine: bool = False,
                 min_refinement_level: int = None):
        """
        implements a class for using coordinates as geometry objects representing the numerical
        domain or geometries inside the domain for a 2D case only

        Note: The coordinates need to form an enclosed area

        :param name: name of the geometry object
        :type name: str
        :param keep_inside: flag if the points inside the object should be masked out (False) or kept (True)
        :type keep_inside: bool
        :param coordinates: coordinates of the geometry or domain; they need to form an enclosed area
        :type coordinates: any
        :param refine: flag if the mesh around the geometry object should be refined after S^3 generated the mesh
        :type refine: bool
        :param min_refinement_level: option to define a min. refinement level with which the geometry should be
                                     resolved; if 'None' and 'refine = True' the geometry will be resolved with the max.
                                     refinement level present at its surface after S^3 has generated the grid
        :type keep_inside: int
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._coordinates = Polygon(coordinates)
        self._type = "coord_2D"

        # xmin, ymin, zmin, xmax, ymax, zmax
        self._lower_bound = list(self._coordinates.bounds)[:2]
        self._upper_bound = list(self._coordinates.bounds)[2:]

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
        # Create a mask. We can't compute this for all nodes at once, because within() method only returns a single
        # bool, but we need to have a bool for each node. The mask is expected to be always 'False' outside the geometry
        # and always 'True' inside it (independently if it is a geometry or domain)
        mask = tensor([Point(cell_nodes[i, :]).within(self._coordinates) for i in range(cell_nodes.size(0))])

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def pre_check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> Tensor:
        """
        method to pre-check if a cell is within a rectangular bounding box of the geometry object
        -> much faster than check the polygon directly if it is expected to generate large numbers of cells outside
        the bounding box

        :param cell_nodes: vertices of the cell which should be checked
        :type cell_nodes: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: bool
        :return: flag if the cell is valid ('False') or invalid ('True') based on the specified settings
        :rtype: bool
        """
        mask = mask_box(cell_nodes, self._lower_bound, self._upper_bound)

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _check_geometry(self) -> None:
        """
        method to check the user input for correctness
        """
        # check if an enclosed area is provided (is_closed property only available for line strings in shapely)
        assert self._coordinates.boundary.is_closed, (f"Expected an enclosed area formed by the provided coordinates "
                                                      f"for geometry {self.name}.")

    @property
    def type(self) -> str:
        return self._type
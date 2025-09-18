"""
Implement a class for using 2D coordinates as a geometry object.
"""
from numpy import ndarray
from typing import Union
from shapely import Point, Polygon
from torch import Tensor, tensor

from flowtorch.data import mask_box

from .geometry_base import GeometryObject


class GeometryCoordinates2D(GeometryObject):
    __short_description__ = "2D coordinates for geometries"

    def __init__(self, name: str, keep_inside: bool, coordinates: Union[list, ndarray], refine: bool = False,
                 min_refinement_level: int = None):
        """
        Implements a class for using coordinates as geometry objects representing the numerical
        domain or geometries inside the domain (2D case only).

        Note:
            The coordinates need to form an enclosed area.

        :param name: Name of the geometry object.
        :type name: str
        :param keep_inside: If True, the points inside the object are kept; if False, they are masked out.
        :type keep_inside: bool
        :param coordinates: Coordinates of the geometry or domain. They need to form an enclosed area.
        :type coordinates: list | numpy.ndarray
        :param refine: If True, the mesh around the geometry object is refined after :math:`S^3` generates the mesh.
        :type refine: bool
        :param min_refinement_level: Minimum refinement level for resolving the geometry. If ``None`` and
            ``refine=True``, the geometry is resolved with the maximum refinement level present at its surface
            after S³ has generated the grid.
        :type min_refinement_level: int | None
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
        Check if a cell is valid or invalid based on the specified settings.

        :param cell_nodes: Vertices of the cell to be checked.
        :type cell_nodes: pt.Tensor
        :param refine_geometry: If ``False``, cells are masked out while generating the grid.
            If ``True``, checks whether a cell is located in the vicinity of the geometry surface
            to refine it subsequently. This parameter is provided by  :math:`S^3`.
        :type refine_geometry: bool
        :return: ``True`` if the cell is invalid, ``False`` if the cell is valid.
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
        Pre-check if a cell is within the rectangular bounding box of the geometry object.

        This method is faster than checking the polygon directly and is especially useful
        when generating large numbers of cells outside the bounding box.

        :param cell_nodes: Vertices of the cell to be checked.
        :type cell_nodes: pt.Tensor
        :param refine_geometry: If ``False``, cells are masked out while generating the grid.
            If ``True``, checks whether a cell is located in the vicinity of the geometry surface
            to refine it subsequently. This parameter is provided by :math:`S^3`.
        :type refine_geometry: bool
        :return: ``True`` if the cell is invalid, ``False`` if the cell is valid.
        :rtype: bool
        """
        mask = mask_box(cell_nodes, self._lower_bound, self._upper_bound)

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _check_geometry(self) -> None:
        """
        Check the user input for correctness.

        :return: None
        :rtype: None
        """
        # check if an enclosed area is provided (is_closed property only available for line strings in shapely)
        assert self._coordinates.boundary.is_closed, (f"Expected an enclosed area formed by the provided coordinates "
                                                      f"for geometry {self.name}.")

    @property
    def type(self) -> str:
        """
        Return the name of the geometry object.

        :return: Name of the geometry object.
        :rtype: str
        """
        return self._type
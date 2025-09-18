"""
 Implements a class for using circles (2D) or spheres (3D) as geometry object.
"""
from typing import Union
from torch import Tensor

from flowtorch.data import mask_sphere

from .geometry_base import GeometryObject

class SphereGeometry(GeometryObject):
    __short_description__ = "circles (2D) or spheres (3D)"

    def __init__(self, name: str, keep_inside: bool, position: list, radius: Union[int, float], refine: bool = False,
                 min_refinement_level: int = None):
        """
        Implement a class for using circles (2D) or spheres (3D) as geometry objects
        representing the numerical domain or geometries inside the domain.

        :param name: Name of the geometry object.
        :type name: str
        :param keep_inside: If ``True``, the points inside the object are kept; if ``False``, they are masked out.
        :type keep_inside: bool
        :param position: Position of the circle or sphere as ``[x, y, z]`` (center coordinates).
        :type position: list
        :param radius: Radius of the circle or sphere.
        :type radius: Union[int, float]
        :param refine: If ``True``, the mesh around the geometry object is refined after :math:`S^3` generates the mesh.
        :type refine: bool
        :param min_refinement_level: Minimum refinement level for resolving the geometry. If ``None`` and
            ``refine=True``, the geometry will be resolved with the maximum refinement level present at its surface
            after :math:`S^3` has generated the grid.
        :type min_refinement_level: int | None
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._position = position
        self._radius = radius
        self._type = "sphere"

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
        assert cell_nodes.size(1) == len(self._position), (f"Number of dimensions of the cell does not match the "
                                                           f"number of dimensions for the position. Expected "
                                                           f"{cell_nodes.size(-1)} values, found "
                                                           f"{len(self._position)} for geometry {self.name}.")

        # create a mask, the mask is expected to be always 'False' outside the geometry and always 'True' inside it
        # (independently if it is a geometry or domain)
        mask = mask_sphere(cell_nodes, self._position, self._radius)

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _check_geometry(self) -> None:
        """
        Check the user input for correctness.

        :return: None
        :rtype: None
        """
        # check if position is an empty list
        assert self._position, "Found empty list for the position. Please provide values for the position."

        # check if the radius is an int or float
        assert isinstance(self._radius, Union[int, float]), (f"Expected the type of radius to be Union[int, float], "
                                                             f"got {type(self._radius)} for geometry {self.name} "
                                                             f"instead.")

        # make sure the radius is larger than zero
        assert self._radius > 0, f"Expected a radius larger than zero but found a value of {self._radius}."

    @property
    def type(self) -> str:
        """
        Return the name of the geometry object.

        :return: Name of the geometry object.
        :rtype: str
        """
        return self._type


"""
    implements a class for using circles (2D) or spheres (3D) as geometry objects
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
        implements a class for using circles (2D) or spheres (3D) as geometry objects representing the numerical
        domain or geometries inside the domain

        :param name: name of the geometry object
        :type name: str
        :param keep_inside: flag if the points inside the object should be masked out (False) or kept (True)
        :type keep_inside: bool
        :param position: position of the circle or sphere as [x, y, z] (center of sphere)
        :type position: list
        :param radius: radius of the circle or sphere
        :type radius: Union[int, float]
        :param refine: flag if the mesh around the geometry object should be refined after S^3 generated the mesh
        :type refine: bool
        :param min_refinement_level: option to define a min. refinement level with which the geometry should be
                                     resolved; if 'None' and 'refine = True' the geometry will be resolved with the max.
                                     refinement level present at its surface after S^3 has generated the grid
        :type min_refinement_level: int
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._position = position
        self._radius = radius
        self._type = "sphere"

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
        method to check the user input for correctness

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
        returns name of the geometry object

        :return: name of the geometry object
        :rtype: str
        """
        return self._type


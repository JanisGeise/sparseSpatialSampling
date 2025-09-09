"""
implements a class for using cylinders (3D) as geometry objects
"""
from typing import Union, List
from torch import Tensor, tensor, cross, logical_and, where, float64

from .geometry_base import GeometryObject


class CylinderGeometry3D(GeometryObject):
    """
    implements a class for using cylinders (3D)
    """
    def __init__(self, name: str, keep_inside: bool, position: List[Union[list, tuple]], radius: Union[int, float],
                 refine: bool = False, min_refinement_level: int = None):
        """
        implements a class for using cylinders (3D) as geometry objects representing the numerical
        domain or geometries inside the domain

        Note: the length and orientation of the cylinder is inferred by two circles representing the start and end point
              of the cylinder.

        :param name: name of the geometry object
        :type name: str
        :param keep_inside: flag if the points inside the object should be masked out (False) or kept (True)
        :type keep_inside: bool
        :param position: position of the two circles [x, y, z] (center coordinates) spanning the cylinder (start and
                         end of the cylinder)
        :type position: List[Union[list, tuple]]
        :param radius: radius of the cylinder
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
        self._type = "cylinder"

        # check the user input based on the specified settings
        self._check_geometry()

        # convert the position to tensor of floats (otherwise the cross-product can't be computed)
        self._position = tensor(self._position).float()

        # compute direction vector of cylinder centerline
        self._axis = (self._position[1, :] - self._position[0, :]).type(float64)

        # compute the norm of the axis vector
        self._norm = self._axis.norm()

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
        # create a mask, the mask is expected to be always 'False' outside the geometry and always 'True' inside it
        # (independently if it is a geometry or domain);
        # cast cell to tensor of floats just to make sure that the cross-product works
        mask = self._mask_cylinder(cell_nodes)

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _check_geometry(self) -> None:
        """
        method to check the user input for correctness

        :return: None
        :rtype: None
        """
        # check if position is an empty list
        assert self._position, "Found empty list for the position. Please provide values for the positions."

        # check that the cylinder is exactly made up of two position coordinates
        assert len(self._position) == 2, (f"Expected exactly two entries for the position but found "
                                          f"{len(self._position)} entries.")

        # make sure that the positions of the circles are different
        assert self._position[0] != self._position[1], ("Expected two different positions, a cylinder of length zero is"
                                                        "invalid.")

        # make sure that the radius is an int or float
        assert type(self._radius) is int or type(self._radius) is float, (f"Expected the type of radius to be "
                                                                          f"Union[int, float], got {type(self._radius)}"
                                                                          f" for geometry {self.name} instead.")

        # make sure the radius is larger than zero
        assert self._radius > 0, f"Expected a radius larger than zero but found a value of {self._radius}."

    def _mask_cylinder(self, vertices: Tensor) -> Tensor:
        """
        select all vertices, which are located inside a cylinder or on its surface

        :param vertices: tensor of vertices where each column corresponds to a coordinate
        :type vertices: boolean mask that's 'True' for every vertex inside the cylinder or on the cylinder's surface
        :rtype: pt.Tensor
        """
        # computer the normal distance of the point to an arbitrary (here starting point) point on the centerline
        direction_vec = (vertices - self._position[0, :]).type(self._axis.dtype)
        normal_distance = (cross(self._axis.expand_as(direction_vec), direction_vec, 1)).norm(dim=1) / self._norm

        # project the vertices onto the cylinder centerline and scale with the cylinder length to get information about
        # the direction of the distance relative to the start of the cylinder
        projection = (direction_vec * self._axis.expand_as(direction_vec)).sum(-1) / self._norm

        # create mask -> needs to return 'True' if point is inside, therefore it needs to yield 'True' if:
        # projection >= 0 and projection <= pt.norm(axis) and normal_distance <= radius
        return logical_and(logical_and(where(0 <= projection, True, False),
                                       where(projection <= self._norm, True, False)
                                       ),
                           where(normal_distance <= self._radius, True, False))

    @property
    def type(self) -> str:
        """
        returns name of the geometry object

        :return: name of the geometry object
        :rtype: str
        """
        return self._type

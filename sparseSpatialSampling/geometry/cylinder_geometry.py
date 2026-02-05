"""
Implements a class for using cylinders, cones and conical objects (3D) as geometry object.
"""
from typing import Union, List
from torch import Tensor, tensor, cross, logical_and, where, float64

from .geometry_base import GeometryObject

class CylinderGeometry3D(GeometryObject):
    __short_description__ = "cylinders, conical objects and cones (3D)"

    def __init__(self, name: str, keep_inside: bool, position: List[Union[list, tuple]],
                 radius: Union[int, float, list, tuple], refine: bool = False, min_refinement_level: int = None):
        """
        Implement a class for using cylinders with a constant radius, cones, and conical objects (3D)
        as geometry objects representing the numerical domain or geometries inside the domain.

        Note:
            The length and orientation of the cylinder is inferred from two circles representing the start and end
            points of the cylinder. The circles do not have to be aligned, allowing the creation of *oblique*
            cylinders along arbitrary directions, as long as both circles are defined in the same coordinate plane.

        :param name: Name of the geometry object.
        :type name: str
        :param keep_inside: If ``True``, the points inside the object are kept; if ``False``, they are masked out.
        :type keep_inside: bool
        :param position: Coordinates of the circles center ``[(x1, y1, z1), (x2, y2, z2)]``
            representing the start and end points of the cylinder
        :type position: list[Union[list, tuple]]
        :param radius: Radius/radii of the cylinder(s):

            - If only one radius is given, it is assumed to be constant along the extrusion axis of the cylinder.
            - For conical objects, two radii are required (one for each position).
            - For cones, the radius associated with the tip must be set to zero.
        :type radius: Union[int, float, list, tuple]
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
        self._type = "cylinder"

        # check the user input based on the specified settings
        self._check_geometry()

        # convert the position to tensor of floats (otherwise the cross-product can't be computed)
        self._position = tensor(self._position).float()

        # compute direction vector of cylinder centerline
        self._axis = (self._position[1, :] - self._position[0, :]).type(float64)

        # compute the norm of the axis vector
        self._norm = self._axis.norm()

        # we have to compute the main dimension and the midpoint if the name of the GeometryObject is domain
        self._main_width = None if not keep_inside else self._compute_main_width()
        self._center = None if not keep_inside else self._compute_center()

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
        # create a mask, the mask is expected to be always 'False' outside the geometry and always 'True' inside it
        # (independently if it is a geometry or domain);
        # cast cell to tensor of floats just to make sure that the cross-product works
        mask = self._mask_cylinder(cell_nodes)

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _check_geometry(self) -> None:
        """
        Check the user input for correctness.

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

        # make sure that the radius is an int or float, or  in case of two radii a list or tuple
        assert isinstance(self._radius, Union[int, float, list, tuple]), (f"Expected the type of radius to be "
                                                                          f"Union[int, float, list, tuple], "
                                                                          f"got {type(self._radius)}"
                                                                          f" for geometry {self.name} instead.")

        # perform more checks on the radius/radii
        if isinstance(self._radius, Union[int, float]):
            # make sure the radius is larger than zero
            assert self._radius > 0, f"Expected a radius larger than zero but found a value of {self._radius}."
        else:
            # make sure that we have either one or two radii
            assert len(self._radius) == 2, f"Expected two values for the radii but found {len(self._radius)}."

            # make sure they are >=0
            assert self._radius[0] >= 0 and self._radius[1] >= 0, (f"Expected all radii >= 0 but found a values of "
                                                                   f"{self._radius}.")

            # ensure that max. one radius is zero
            assert (self._radius[0] == self._radius[1]) == 0, (f"Both values for the radii can't be zero. At least one "
                                                               f"radius has to be > 0 but found values of "
                                                               f"{self._radius}.")

    def _mask_cylinder(self, vertices: Tensor) -> Tensor:
        """
        Select all vertices that are located inside a cylinder or on its surface.

        :param vertices: Tensor of vertices, where each column corresponds to a coordinate.
        :type vertices: pt.Tensor
        :return: Boolean mask that is ``True`` for every vertex inside the cylinder or on its surface.
        :rtype: pt.Tensor
        """
        # compute the normal distance of the point to an arbitrary (here starting point) point on the centerline
        direction_vec = (vertices - self._position[0, :]).type(self._axis.dtype)
        normal_distance = (cross(self._axis.expand_as(direction_vec), direction_vec, 1)).norm(dim=1) / self._norm

        # project the vertices onto the cylinder centerline and scale with the cylinder length to get information about
        # the direction of the distance relative to the start of the cylinder
        projection = (direction_vec * self._axis.expand_as(direction_vec)).sum(-1) / self._norm

        # create mask -> needs to return 'True' if point is inside, therefore it needs to yield 'True' if:
        # projection >= 0 and projection <= pt.norm(axis) and normal_distance <= radius
        within_height = logical_and(where(0 <= projection, True, False), where(projection <= self._norm, True, False))

        # check if the cell is within the local radius
        if isinstance(self._radius, Union[float, int]):
            _radius = self._radius
        else:
            # if we have two different radii then linearly interpolate the local radius at our cell
            # (since we only have start and end of the cone), we also need to normalize it to the range of [0, 1] to
            # obtain the relative position along the axis
            _radius = self._radius[0] + projection / self._norm * (self._radius[1] - self._radius[0])

        within_radius = where(normal_distance <= _radius, True, False)
        return logical_and(within_height, within_radius)

    @property
    def type(self) -> str:
        """
        Return the name of the geometry object.

        :return: Name of the geometry object.
        :rtype: str
        """
        return self._type

    @property
    def main_width(self) -> float:
        """
        Return the width of the main dimension of the cylinder.

        :return: Main width of the cylinder.
        :rtype: float
        """
        return self._main_width

    @property
    def center(self) -> Tensor:
        """
        Return the center coordinates based on the main width of the cylinder.

        :return: center coordinates of the cylinder.
        :rtype: pt.Tensor
        """
        return self._center

    def _compute_main_width(self) -> float:
        """
        Compute the center coordinates based on the main width of the cylinder.

        :return: center coordinates of the cylinder.
        :rtype: pt.Tensor
        """
        return max(max(self._radius) if isinstance(self._radius, list) else self._radius, self._axis.norm().item())

    def _compute_center(self) -> Tensor:
        """
        Compute the geometric center coordinates based on the main dimension of the cylinder.

        :return: center coordinates of the cylinder.
        :rtype: pt.Tensor
        """
        return self._position.mean(0)

if __name__ == "__main__":
    pass
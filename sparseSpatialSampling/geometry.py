"""
    Implements different geometry objects, which can be used to represent the numerical domain or geometries inside the
    domain.
    Currently implemented are:
        - CubeGeometry: rectangles (2D), cubes (3D)
        - SphereGeometry: circles (2D), spheres (3D)
        - CylinderGeometry3D: cylinders (3D)
        - GeometryCoordinates2D: arbitrary 2D geometries; coordinates must be provided
        - GeometrySTL3D: arbitrary 3D geometries; an STL file must be provided
"""
import logging

from numpy import ndarray
from pyvista import PolyData, read
from typing import Union, List
from shapely import Point, Polygon
from torch import Tensor, tensor, cross, logical_and, where, float64

from flowtorch.data import mask_sphere, mask_box

from .geometry_base import GeometryObject

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CubeGeometry(GeometryObject):
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


class SphereGeometry(GeometryObject):
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
        """
        # check if position is an empty list
        assert self._position, "Found empty list for the position. Please provide values for the position."

        # check if the radius is an int or float
        assert type(self._radius) is int or type(self._radius) is float, (f"Expected the type of radius to be "
                                                                          f"Union[int, float], got "
                                                                          f"{type(self._radius)} for geometry "
                                                                          f"{self.name} instead.")

        # make sure the radius is larger than zero
        assert self._radius > 0, f"Expected a radius larger than zero but found a value of {self._radius}."


class CylinderGeometry3D(GeometryObject):
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


class GeometryCoordinates2D(GeometryObject):
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
        self._type = "coordinates 2D"

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

    def _check_geometry(self) -> None:
        """
        method to check the user input for correctness
        """
        # check if an enclosed area is provided (is_closed property only available for line strings in shapely)
        assert self._coordinates.boundary.is_closed, (f"Expected an enclosed area formed by the provided coordinates "
                                                      f"for geometry {self.name}.")


class GeometrySTL3D(GeometryObject):
    def __init__(self, name: str, keep_inside: bool, path_stl_file: str, refine: bool = False,
                 min_refinement_level: int = None, reduce_by: Union[int, float] = 0):
        """
        implements a class for using an STL file as geometry objects representing the numerical domain or geometries
        inside the domain for a 3D case

        Note: pyVista requires the STL file to have a closed surface

        :param name: name of the geometry object
        :type name: str
        :param keep_inside: flag if the points inside the object should be masked out (False) or kept (True)
        :type keep_inside: bool
        :param path_stl_file: path to the STL file
        :type path_stl_file: str
        :param refine: flag if the mesh around the geometry object should be refined after S^3 generated the mesh
        :type refine: bool
        :param min_refinement_level: option to define a min. refinement level with which the geometry should be
                                     resolved; if 'None' and 'refine = True' the geometry will be resolved with the max.
                                     refinement level present at its surface after S^3 has generated the grid
        :type min_refinement_level: int
        :param reduce_by: reduce the STL file by a factor, recommended for larger STL files since the number of points
                          within the STL file increases the runtime significantly.
                          A value of zero means no compression, a value of 0.9 ... 0.98 should work for most STL files.
                          Note: this factor has to be 0 <= reduce_by < 1
        :type min_refinement_level: Union[int, float]
        """
        # make sure the compression factor is in a valid range
        if reduce_by < 0:
            logger.warning(f"Found invalid negative value for 'reduce_by' of {reduce_by}. Disabling compression.")
            reduce_by = 0
        elif reduce_by >= 1:
            logger.warning(f"Found invalid value for 'reduce_by ' of {reduce_by}. Compression factor needs to be "
                           f"0 <= reduce_by < 1. Correcting 'reduce_by' to reduce_by=0.99")
            reduce_by = 0.99

        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._stl_file = read(path_stl_file).decimate(reduce_by)
        self._type = "STL"

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
        # for 3D geometries represented by STL files, we need to mask using pyVista; here we don't check for closed
        # surface since we already did that on initialization
        n = PolyData(cell_nodes.numpy())

        # create a mask, the mask is expected to be always 'False' outside the geometry and always 'True' inside it
        # (independently if it is a geometry or domain)
        mask = tensor(n.select_enclosed_points(self._stl_file, check_surface=False)["SelectedPoints"]).bool()

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _check_geometry(self) -> None:
        """
        method to check the user input for correctness
        """
        # check if the STL file is closed and manifold
        test_data = PolyData([(0.0, 0.0, 0.0)])
        try:
            # pyVista will throw a RuntimeError if the surface is not closed and manifold
            _ = test_data.select_enclosed_points(self._stl_file, check_surface=True)
        except RuntimeError:
            logger.critical(f"Expected an STL file with a closed and manifold surface for geometry {self.name}.")
            exit(0)


if __name__ == "__main__":
    pass

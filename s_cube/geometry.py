"""
    Implements different geometry objects, which can be used to represent the numerical domain or geometries inside the
    domain.
    Currently implemented are:
        - CubeGeometry: rectangles (2D), cubes (3D)
        - SphereGeometry: circles (2D), spheres (3D)
        - GeometryCoordinates2D: arbitrary 2D geometries; coordinates must be provided
        - GeometrySTL3D: arbitrary 3D geometries; STL file must be provided
"""
import logging

from typing import Union
from torch import Tensor, tensor
from pyvista import PolyData
from shapely import Point, Polygon
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

    def check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> bool:
        """
        method to check if a cell is valid or invalid based on the specified settings

        :param cell_nodes: vertices of the cell which should be checked
        :type cell_nodes: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: Bool
        :return: flag if the cell is valid or invalid based on the specified settings
        :rtype: bool
        """
        # check if the number of boundaries matches the number of physical dimensions;
        # this can't be done beforehand because we don't know the number of physical dimensions yet
        assert cell_nodes.size(-1) == len(self._lower_bound), (f"Number of dimensions of the cell does not match the "
                                                               f"number of given bounds. Expected "
                                                               f"{cell_nodes.size(-1)} values, found "
                                                               f"{len(self._lower_bound)}.")

        # create a mask
        mask = mask_box(cell_nodes, self._lower_bound, self._upper_bound)

        # check if the cell is valid or invalid
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
                                                                  f"bound.")

        # check if the lower boundary is smaller than the upper boundary
        for i, v in enumerate(zip(self._lower_bound, self._upper_bound)):
            assert v[0] < v[1], (f"Value of {v[0]} for the lower bound at position {i} is larger or equal than the "
                                 f"value of {v[1]} for the upper bound. The the lower bound must be smaller than the "
                                 f"upper bound!")


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

    def check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> bool:
        """
        method to check if a cell is valid or invalid based on the specified settings

        :param cell_nodes: vertices of the cell which should be checked
        :type cell_nodes: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: Bool
        :return: flag if the cell is valid or invalid based on the specified settings
        :rtype: bool
        """
        # check if the number of boundaries matches the number of physical dimensions;
        # this can't be done beforehand because we don't know the number of physical dimensions yet
        assert cell_nodes.size(-1) == len(self._position), (f"Number of dimensions of the cell does not match the "
                                                            f"number of dimensions for the position. Expected "
                                                            f"{cell_nodes.size(-1)} values, found "
                                                            f"{len(self._position)}.")

        # create a mask
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
        assert type(self._radius) is int or type(self._radius) is float, (f"Expected a type or radius to be "
                                                                          f"Union[int, float], got "
                                                                          f"{type(self._radius)}.")


class GeometryCoordinates2D(GeometryObject):
    def __init__(self, name: str, keep_inside: bool, coordinates: any, refine: bool = False,
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

    def check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> bool:
        """
        method to check if a cell is valid or invalid based on the specified settings

        :param cell_nodes: vertices of the cell which should be checked
        :type cell_nodes: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: Bool
        :return: flag if the cell is valid or invalid based on the specified settings
        :rtype: bool
        """
        # Create a mask. We can't compute this for all nodes at once, because within() method only returns a single
        # bool, but we need to have a bool for each node
        # TODO: make more efficient if possible
        mask = tensor([Point(cell_nodes[i, :]).within(self._coordinates) for i in range(cell_nodes.size(0))])

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _check_geometry(self) -> None:
        """
        method to check the user input for correctness
        """
        # TODO assert enclosed area -> test if works possible?
        pass


class GeometrySTL3D(GeometryObject):
    def __init__(self, name: str, keep_inside: bool, path_stl_file: str, refine: bool = False,
                 min_refinement_level: int = None):
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
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._stl_file = PolyData(path_stl_file)
        self._type = "STL"

        # check the user input based on the specified settings
        self._check_geometry()

    def check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> bool:
        """
        method to check if a cell is valid or invalid based on the specified settings

        :param cell_nodes: vertices of the cell which should be checked
        :type cell_nodes: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: Bool
        :return: flag if the cell is valid or invalid based on the specified settings
        :rtype: bool
        """
        # for 3D geometries represented by STL files, we need to mask using pyVista; here we don't check for closed
        # surface since we already did that on initialization
        n = PolyData(cell_nodes.tolist())
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
            logger.critical("Expected an STL file with a closed and manifold surface.")
            exit(0)


if __name__ == "__main__":
    pass

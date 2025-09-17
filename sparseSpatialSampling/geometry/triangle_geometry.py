"""
 implements a class for using triangles (2D) as geometry object
"""
import logging
from typing import Union

from torch import Tensor, tensor, float64
from .geometry_base import GeometryObject

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)


class TriangleGeometry(GeometryObject):
    __short_description__ = "triangles (2D)"

    def __init__(self, name: str, keep_inside: bool, points: Union[list, Tensor], refine: bool = False,
                 min_refinement_level: int = None):
        """
        implements a class for using triangles (2D) as geometry objects representing the numerical
        domain or geometries inside the domain

        :param name: name of the geometry object
        :type name: str
        :param keep_inside: flag if the points inside the object should be masked out (False) or kept (True)
        :type name: bool
        :param points: list containing the three points of the triangle, each point is made up of two coordinates
        :type points: list[tuple | list | pt.Tensor | np.ndarray] | pt.Tensor
        :param refine: flag if the mesh around the geometry object should be refined after S^3 generated the mesh
        :type refine: bool
        :param min_refinement_level: option to define a min. refinement level with which the geometry should be
                                     resolved; if 'None' and 'refine = True' the geometry will be resolved with the max.
                                     refinement level present at its surface after S^3 has generated the grid
        :type min_refinement_level: int
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._type = "triangle"

        # convert the coordinates to a tensor if that's not yet the case
        for i, p in enumerate(points):
            if type(p) != Tensor:
                try:
                    points[i] = tensor(p)
                except TypeError:
                    logger.error(f"Could not convert coordinate {i} of type {type(p)} to a tensor.")

            # since we use the cross product, we need to cast everything to float
            points[i].type(float64)

        self._points = points

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
        # create a mask, the mask is expected to be always 'False' outside the geometry and always 'True' inside it
        # (independently if it is a geometry or domain)
        mask = self._mask_triangle(cell_nodes)

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _mask_triangle(self, vertices: Tensor) -> Tensor:
        """
        select all vertices, which are located inside a triangle or on its surface

        :param vertices: tensor of vertices where each column corresponds to a coordinate
        :type vertices: pt.Tensor
        :return: boolean mask that's 'True' for every vertex inside the triangle or on the triangle's surface
        :rtype: pt.Tensor
        """
        # pytorch doesn't support cross product in 2D, so we have to do it manually
        # check for line between first and second point of the triangle
        d1 = self._cross_product_2d(self._points[1] - self._points[0], vertices - self._points[0])

        # check for line between last and second point of the triangle
        d2 = self._cross_product_2d(self._points[2] - self._points[1], vertices - self._points[1])

        # check for line between last and first point of the triangle
        d3 = self._cross_product_2d(self._points[0] - self._points[2], vertices - self._points[0])

        # check the signs of the cross product between triangle side vector and vector to the node
        _neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        _pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

        return ~(_neg & _pos)

    @staticmethod
    def _cross_product_2d(a: Tensor, b: Tensor) -> Tensor:
        # cross product a x b = a0*b1 - a1b0 in 2D
        return a[0] * b[:, 1] - a[1] * b[:, 0]


    def _check_geometry(self) -> None:
        """
        method to check the user input for correctness

        :return: None
        :rtype: None
        """
        # make sure the points are in some kind of list
        assert isinstance(self._points, Union[list, Tensor]), (f"Expected the points to be a list or pt.Tensor, "
                                                               f"but found type {type(self._points)} instead.")

        # make sure we have exactly 3 points
        assert len(self._points) == 3, f"Expected 3 points, but found {len(self._points)} points instead."

        # make sure each point has exactly 2 coordinates (x-y)
        assert all(len(p) == 2 for p in self._points), ("All given coordinates have to contain exactly 2 entries with "
                                                        "the x- and y-coordinates.")

        # make sure the coordinates are unique
        assert len(self._points) == len(set(self._points)), "The coordinates of the triangle hae to be unique."

        # make sure that the area of the triangle is larger than zero, pyTorch doesn't support cross product in 2D,
        # so do it manually
        _a = self._points[1] - self._points[0]
        _b = self._points[2] - self._points[0]
        _area = 0.5 * abs(_a[0] * _b[1] - _a[1] * _b[0])
        assert _area > 0, f"The area of the triangle has to be larger than zero. Found an area of {_area}."

    def check_triangle(self, vertices: Tensor) -> Tensor:
        """
        public method to check if given vertices are inside this triangle to access _mask_triangle method from other
        classes

        :param vertices: tensor of vertices where each column corresponds to a coordinate
        :type vertices: pt.Tensor
        :return: boolean mask that's 'True' for every vertex inside the triangle or on the triangle's surface
        :rtype: pt.Tensor
        """
        return self._mask_triangle(vertices)

    @property
    def type(self) -> str:
        """
        returns name of the geometry object

        :return: name of the geometry object
        :rtype: str
        """
        return self._type
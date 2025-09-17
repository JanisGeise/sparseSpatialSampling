"""
implements a class for using triangular prisms (3D) as geometry objects
"""
from typing import List, Union
from torch import Tensor, tensor, float64, logical_and, where, allclose

from .geometry_base import GeometryObject
from .triangle_geometry import TriangleGeometry

class PrismGeometry3D(GeometryObject):
    """
    Implements a class for using triangular prisms (3D) as geometry objects
    """

    def __init__(self, name: str, keep_inside: bool,
                 positions: List[List[Union[list, tuple]]],
                 refine: bool = False, min_refinement_level: int = None):
        """
        Implements a class for using triangular prisms (3D) as geometry objects representing the numerical
        domain or geometries inside the domain.

        Note: The prism is defined by two triangles, connected by an extrusion axis.
              The triangles have to be aligned along a coordinate axis.

        :param name: name of the geometry object
        :type name: str
        :param keep_inside: flag if the points inside the object should be masked out (False) or kept (True)
        :type keep_inside: bool
        :param positions: two lists of 3D coordinates, each containing the three vertices of the start and end triangles
                          e.g. [[[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]],
                                [[x1',y1',z1'], [x2',y2',z2'], [x3',y3',z3']]]
        :type positions: List[List[Union[list, tuple]]]
        :param refine: flag if the mesh around the geometry object should be refined after S^3 generated the mesh
        :type refine: bool
        :param min_refinement_level: option to define a min. refinement level with which the geometry should be
                                     resolved; if 'None' and 'refine = True' the geometry will be resolved with the max.
                                     refinement level present at its surface after S^3 has generated the grid
        :type min_refinement_level: int
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._type = "prism"
        self._positions = positions

        # check the user input based on the specified settings
        self._check_geometry()

        # convert positions to tensor
        self._positions = [tensor(tri, dtype=float64) for tri in self._positions]

        # axis of extrusion
        self._axis = (self._positions[1][0] - self._positions[0][0]).type(float64)

        # compute the norm of the axis vector
        self._norm = self._axis.norm()

        # determine in which plane the triangle lies and check if the two triangles are aligned
        # TODO: this changes if triangles are not aligned along a coordinate direction
        self._dim = where(self._axis == 0)[0]

        # the 2nd check is redundant since we don't have two zeros if the axis is not aligned along a coordinate
        # direction, but just in case I change something later
        assert len(self._dim) == 2, "The specified triangles are not aligned along a coordinate direction."
        assert allclose(self._positions[0][:, self._dim], self._positions[1][:, self._dim]), \
            "The specified triangles are not aligned along a coordinate direction."

        # create TriangleGeometry instances for start and end faces
        self._triangles = [
            TriangleGeometry(f"{name}_first", keep_inside=True, points=self._positions[0][:, self._dim]),
            TriangleGeometry(f"{name}_second", keep_inside=True, points=self._positions[1][:, self._dim])
        ]

    def check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> Tensor:
        """
        Method to check if a cell is valid or invalid based on the specified settings.

        :param cell_nodes: vertices of the cell which should be checked
        :type cell_nodes: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: bool
        :return: flag if the cell is valid ('False') or invalid ('True') based on the specified settings
        :rtype: bool
        """
        mask = self._mask_prism(cell_nodes)
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _mask_prism(self, vertices: Tensor) -> Tensor:
        """
        Select all vertices which are located inside a triangular prism or on its surface.

        :param vertices: tensor of vertices where each column corresponds to a coordinate
        :type vertices: pt.Tensor
        :return: boolean mask that's 'True' for every vertex inside the prism or on the prism's surface
        :rtype: pt.Tensor
        """
        # compute the normal distance of the point to the point on the centerline of the first triangle node
        direction_vec = (vertices - self._positions[0][0]).type(self._axis.dtype)

        # project the vertices onto prism axis
        projection = (direction_vec * self._axis.expand_as(direction_vec)).sum(-1) / self._norm

        # check if the nodes are within the height of the prism
        _within_height = logical_and(where(0 <= projection, True, False), where(projection <= self._norm, True, False))

        # use the 2D TriangleGeometry mask for inside check of the triangle
        _inside_triangle_1 = self._triangles[0].check_triangle(vertices[:, self._dim])

        # in case the triangles are not aligned along a coordinate axis, linearly interpolate a new triangle at the
        # requested position TODO: how to determine self._dim in these cases?
        if not allclose(self._positions[0][:, self._dim], self._positions[1][:, self._dim]):
            pass

        return logical_and(_within_height, _inside_triangle_1)

    def _check_geometry(self) -> None:
        """
        method to check the user input for correctness

        :return: None
        :rtype: None
        """
        # check if positions is an empty list
        assert self._positions, "Found empty list for the positions. Please provide values for the prism."

        # check that we have exactly two triangles
        assert len(self._positions) == 2, (f"Expected exactly two triangles for the prism but found "
                                           f"{len(self._positions)} entries.")

        # check that each triangle has exactly 3 vertices
        assert all(len(tri) == 3 for tri in self._positions), \
            "Each triangle must have exactly 3 vertices."

    @property
    def type(self) -> str:
        """
        Returns name of the geometry object.

        :return: name of the geometry object
        :rtype: str
        """
        return self._type

"""
Implements a class for using prisms (3D) as geometry object.
"""
from typing import List, Union
from torch import Tensor, tensor, float64, logical_and, where, allclose, cat, zeros

from .geometry_base import GeometryObject
from .triangle_geometry import TriangleGeometry

class PrismGeometry3D(GeometryObject):
    __short_description__ = "prisms (3D)"

    def __init__(self, name: str, keep_inside: bool, positions: List[List[Union[list, tuple]]], refine: bool = False,
                 min_refinement_level: int = None):
        """
        Implement a class for using prisms (3D) as geometry objects representing the numerical
        domain or geometries inside the domain.

        Note:
            The prism is defined by two triangles connected by an extrusion axis.
            The triangles must be aligned along a coordinate axis.

        :param name: Name of the geometry object.
        :type name: str
        :param keep_inside: If ``True``, the points inside the object are kept; if ``False``, they are masked out.
        :type keep_inside: bool
        :param positions: Two lists of 3D coordinates, each containing the three vertices of the start and end triangles, e.g.:

            ``[[[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], [[x1', y1', z1'], [x2', y2', z2'], [x3', y3', z3']]]``
        :type positions: list[list[Union[list, tuple]]]
        :param refine: If ``True``, the mesh around the geometry object is refined after :math:`S^3` generates the mesh.
        :type refine: bool
        :param min_refinement_level: Minimum refinement level for resolving the geometry. If ``None`` and
            ``refine=True``, the geometry will be resolved with the maximum refinement level present at its surface
            after :math:`S^3` has generated the grid.
        :type min_refinement_level: int | None
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
        mask = self._mask_prism(cell_nodes)
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _mask_prism(self, vertices: Tensor) -> Tensor:
        """
        Select all vertices that are located inside a triangular prism or on its surface.

        :param vertices: Tensor of vertices, where each column corresponds to a coordinate.
        :type vertices: pt.Tensor
        :return: Boolean mask that is ``True`` for every vertex inside the prism or on its surface.
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
        # requested position
        # TODO: how to determine self._dim in these cases? -> self._axis.nonzero()[0].item() not working since multiple entries
        if not allclose(self._positions[0][:, self._dim], self._positions[1][:, self._dim]):
            raise NotImplementedError("The triangles are not aligned along a coordinate axis, which is currently not"
                                      " supported.")

        return logical_and(_within_height, _inside_triangle_1)

    def _check_geometry(self) -> None:
        """
        Check the user input for correctness.

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
        Return the name of the geometry object.

        :return: Name of the geometry object.
        :rtype: str
        """
        return self._type

    @property
    def main_width(self) -> float:
        """
        Return the width of the main dimension of the prism.

        :return: Main width of the prism.
        :rtype: float
        """
        return self._main_width

    @property
    def center(self) -> Tensor:
        """
        Return the center coordinates based on the main width of the prism.

        :return: center coordinates of the prism.
        :rtype: pt.Tensor
        """
        return self._center

    def _compute_main_width(self) -> float:
        """
        Compute the center coordinates based on the main width of the prism.

        :return: center coordinates of the prism.
        :rtype: pt.Tensor
        """
        return max(self._axis.norm().item(), max([t.main_width for t in self._triangles]))

    def _compute_center(self) -> Tensor:
        """
        Compute the geometric center coordinates based on the main dimension of the prism.

        :return: center coordinates of the prism.
        :rtype: pt.Tensor
        """
        # get the 2D centers of the triangles
        _triangle_centers = cat([t.center.unsqueeze(-1) for t in self._triangles], -1).mean(1)

        # check the direction of the axis
        _ax_dim = self._axis.nonzero()[0]
        if len(_ax_dim) > 1:
            raise NotImplementedError("The triangles are not aligned along a coordinate axis, which is currently not"
                                      " supported.")
        else:
            # compute the mean position on the axis, then insert it into the correct dimension. Since the ax_dim point
            # is the same in all points per triangle, it is sufficient to take the first entry per triangle
            _ax_avg_pos = (self._positions[1][0, _ax_dim.item()] + self._positions[0][0, _ax_dim.item()]) / 2
            _ax_avg_pos_vec = zeros((3, ), dtype=self._axis.dtype)
            _ax_avg_pos_vec[_ax_dim.item()] = _ax_avg_pos
            _ax_avg_pos_vec[self._dim] = _triangle_centers
            return _ax_avg_pos_vec

if __name__ == "__main__":
    pass
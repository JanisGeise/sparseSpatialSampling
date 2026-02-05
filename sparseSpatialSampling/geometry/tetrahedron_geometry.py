"""
Implements a class for using tetrahedrons (3D) as geometry object.
"""
from typing import List, Union
from torch import Tensor, tensor, float64, where, det, ones, cat, cross, dot

from .geometry_base import GeometryObject


class TetrahedronGeometry3D(GeometryObject):
    __short_description__ = "tetrahedrons (3D)"

    def __init__(self, name: str, keep_inside: bool, positions: Union[List[Union[list, tuple]], Tensor],
                 refine: bool = False, min_refinement_level: int = None):
        """
        Implement a class for using tetrahedrons (3D) as geometry objects representing the numerical
        domain or geometries inside the domain.

        :param name: Name of the geometry object.
        :type name: str
        :param keep_inside: If ``True``, the points inside the object are kept; if ``False``, they are masked out.
        :type keep_inside: bool
        :param positions: List or tensor of four 3D coordinates, each representing one vertex of the tetrahedron.
        :type positions: list[list[float]] | list[tuple[float]] | pt.Tensor
        :param refine: If ``True``, the mesh around the geometry object is refined after :math:`S^3` generates the mesh.
        :type refine: bool
        :param min_refinement_level: Minimum refinement level for resolving the geometry. If ``None`` and
            ``refine=True``, the geometry will be resolved with the maximum refinement level present at its surface
            after :math:`S^3` has generated the grid.
        :type min_refinement_level: int | None
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._type = "tetrahedron"
        self._positions = positions
        self._normals = None

        # check the user input based on the specified settings
        self._check_geometry()

        # convert positions to tensor, otherwise make sure we use DP
        if not isinstance(self._positions, Tensor):
            self._positions = tensor(self._positions, dtype=float64)
        else:
            self._positions = self._positions.type(float64)

        # make sure the volume is larger zero -> v = 1/6 det(points)
        # since we have a 4x3 matrix, we have to add another column, compare: https://de.wikipedia.org/wiki/Tetraeder
        # abs() since we don't care about the orientation of the tetrahedron
        assert abs(1/6 * det(cat([self._positions, ones((4, 1), dtype=float64)], dim=1))) > 0, \
            "The tetrahedron provided has a volume of zero."

        # compute the normal vectors of each triangle within the tetrahedron
        self._compute_normals()

        # we have to compute the main dimension and the midpoint if the name of the GeometryObject is domain
        self._main_width = None if not keep_inside else self._compute_main_width()
        self._center = None if not keep_inside else self._compute_center()

    def _compute_normals(self) -> None:
        """
        Compute the inward-pointing face normals of the tetrahedron.

        The normals are calculated using the cross product of the edges forming
        each triangular face. The orientation is corrected by checking the
        direction relative to the centroid of the tetrahedron to ensure that
        all normals point inward.

        Reference:
            See `Math StackExchange <https://math.stackexchange.com/questions/3698021/how-to-find-if-a-3d-point-is-in-on-outside-of-tetrahedron>`_
            for details on point-in-tetrahedron checks.

        :return: None
        :rtype: None
        """
        # compute the centroid of the tetrahedron
        _centroid = self._positions.mean(dim=0)

        # compute the face normals of the three triangles making up the tetrahedron
        # n1 = (B - A) x (C - A)
        n1 = cross(self._positions[1, :] - self._positions[0, :], self._positions[2, :] - self._positions[0, :],
                   dim=0).unsqueeze(-1)

        # n2 = (B - A) x (D - A)
        n2 = cross(self._positions[1, :] - self._positions[0, :], self._positions[3, :] - self._positions[0, :],
                   dim=0).unsqueeze(-1)

        # n3 = (C - A) x (D - A)
        n3 = cross(self._positions[2, :] - self._positions[0, :], self._positions[3, :] - self._positions[0, :],
                   dim=0).unsqueeze(-1)

        # n4 = (C - B) x (D - B)
        n4 = cross(self._positions[2, :] - self._positions[1, :], self._positions[3, :] - self._positions[2, :],
                   dim=0).unsqueeze(-1)

        # vectorize
        normals = cat([n1, n2, n3, n4], dim=1)

        # check the direction of the face normals between centroid and all points (nodes) of the tetrahedron
        _check = [dot(_centroid - self._positions[p ,:], normals[:, p]) for p in range(4)]

        # reverse direction for all _check < 0 to ensure that all normals are pointing inwards
        normals[:, where(tensor(_check) < 0)[0]] *= -1
        self._normals = normals

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
        mask = self._mask_tetrahedron(cell_nodes)
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _mask_tetrahedron(self, vertices: Tensor) -> Tensor:
        """
        Select all vertices that are located inside a tetrahedron or on its surface.

        :param vertices: Tensor of vertices, where each column corresponds to a coordinate.
        :type vertices: pt.Tensor
        :return: Boolean mask that is ``True`` for every vertex inside the tetrahedron or on its surface.
        :rtype: pt.Tensor
        """
        # compute all vectors between all nodes of the cell and all nodes of the tetrahedron
        _vectors = vertices.unsqueeze(1) - self._positions.unsqueeze(0)

        # compute all dot products between all nodes of the tetrahedron and all normal vectors for each node of the cell
        # TODO: vectorize dot product
        _dots = tensor([[dot(_vectors[v, p, :], self._normals[:, p]) for p in range(4)] for v in range(vertices.size(0))])

        # if any of the dot product is < 0 (for each node of the cell to all nodes of the tetrahedron),
        # this means we are outside the tetrahedron.
        # Since we have to return False outside the geometry, we have to negate the tensor
        return ~(_dots < 0).bool().any(1)

    def check_tetrahedron(self, vertices: Tensor) -> Tensor:
        """
        Check if the given vertices are inside this tetrahedron. This method provides access
        to the `_mask_tetrahedron` method from other classes.

        :param vertices: Tensor of vertices, where each column corresponds to a coordinate.
        :type vertices: pt.Tensor
        :return: Boolean mask that is ``True`` for every vertex inside the tetrahedron or on its surface.
        :rtype: pt.Tensor
        """
        return self._mask_tetrahedron(vertices)

    def _check_geometry(self) -> None:
        """
        Check the user input for correctness.

        :return: None
        :rtype: None
        """
        # check if positions is an empty list
        if isinstance(self._positions, list):
            assert self._positions, "Found empty list for the positions. Please provide values for the tetrahedron."
        else:
            assert isinstance(self._positions, Tensor), (f"Expected the points to be either a list, tuple or pt.Tensor,"
                                                         f" but found type {type(self._positions)}.")

        # make sure each point is int, float or tensor type
        assert isinstance(self._positions, Union[list, Tensor]), (f"Expected the points to be a list or pt.Tensor, "
                                                               f"but found type {type(self._positions)} instead.")
        # make sure we have exactly 4 points
        assert len(self._positions) == 4, (f"Expected exactly four points for the tetrahedron but found "
                                           f"{len(self._positions)} entries.")

        # make sure each point has three coordinates
        assert all([len(p) == 3 for p in self._positions]), "Each point must have exactly 3 coordinates (x, y, z)."

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
        Return the width of the main dimension of the tetrahedron.

        :return: Main width of the tetrahedron.
        :rtype: float
        """
        return self._main_width

    @property
    def center(self) -> Tensor:
        """
        Return the center coordinates based on the main width of the tetrahedron.

        :return: center coordinates of the tetrahedron.
        :rtype: pt.Tensor
        """
        return self._center

    def _compute_main_width(self) -> float:
        """
        Compute the center coordinates based on the main width of the tetrahedron.

        :return: center coordinates of the tetrahedron.
        :rtype: pt.Tensor
        """
        return (self._positions.max(dim=0).values - self._positions.min(dim=0).values).max().item()

    def _compute_center(self) -> Tensor:
        """
        Compute the geometric center coordinates based on the main dimension of the tetrahedron.

        :return: center coordinates of the tetrahedron.
        :rtype: pt.Tensor
        """
        return self._positions.mean(dim=0)

if __name__ == "__main__":
    pass
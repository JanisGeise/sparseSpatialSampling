"""
Implements a class for using square pyramids (3D) as geometry object.
"""
from typing import List, Union
from torch import Tensor, cat, tensor, argmax, cross, nonzero

from .geometry_base import GeometryObject
from .tetrahedron_geometry import TetrahedronGeometry3D


class PyramidGeometry3D(GeometryObject):
    __short_description__ = "square pyramids (3D)"

    def __init__(self, name: str, keep_inside: bool, nodes: List[Union[list, tuple]], refine: bool = False,
                 min_refinement_level: int = None):
        """
        Implement a class for using square pyramids (3D) as geometry objects representing the numerical
        domain or geometries inside the domain.

        Note:
            It is expected that four out of the five nodes form a planar plane representing the base of the pyramid.
            The order of the nodes doesn't matter.

        :param name: Name of the geometry object.
        :type name: str
        :param keep_inside: If ``True``, the points inside the object are kept; if ``False``, they are masked out.
        :type keep_inside: bool
        :param nodes: List of five 3D coordinates, each representing one vertex of the pyramid.
        :type nodes: list[list[float]] | list[tuple[float]]
        :param refine: If ``True``, the mesh around the geometry object is refined after :math:`S^3` generates the mesh.
        :type refine: bool
        :param min_refinement_level: Minimum refinement level for resolving the geometry. If ``None`` and
            ``refine=True``, the geometry will be resolved with the maximum refinement level present at its surface
            after :math:`S^3` has generated the grid.
        :type min_refinement_level: int | None
        """
        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._type = "pyramid"
        self._nodes = nodes

        # check the user input
        self._check_geometry()
        self._nodes = tensor(self._nodes)

        # create two tetrahedrons from the pyramid
        self._create_tetrahedrons()

        # we have to compute the main dimension and the midpoint if the name of the GeometryObject is domain
        self._main_width = self._compute_main_width()
        self._center = self._compute_center()

    def _create_tetrahedrons(self) -> None:
        """
        Decompose the given pyramid into two tetrahedrons.

        :return: None
        :rtype: None
        """
        # determine the apex based on given nodes
        self._get_apex()

        # determine the main diagonal so we can get the node indices making up the two tetrahedrons
        self._get_main_diagonal()

        # get the node indices for each tetrahedron
        _idx1 = [self._diagonal_idx[0], self._off_diagonal[0], self._diagonal_idx[1], self._apex_idx]
        _idx2 = [self._diagonal_idx[1], self._off_diagonal[1], self._diagonal_idx[0], self._apex_idx]

        # create the two tetrahedrons making up the pyramid
        self._tets = [TetrahedronGeometry3D("tet0", self._keep_inside, self._nodes[_idx1]),
                      TetrahedronGeometry3D("tet1", self._keep_inside, self._nodes[_idx2])]

    def _get_apex(self) -> None:
        """
        Determine the apex of the pyramid.

        :return: None. Updates the apex index in ``self._apex_idx``.
        :rtype: None
        """
        # initialize the current largest number of points within a plane, the corresponding normal vector
        # and a point that lies on this plane
        best_inliers, base_normal, base_p = 0, None, None

        # we now check all possible combinations of three points that can make up a plane within self._nodes
        # since we only have 5 points making up our pyramid, we don't need to vectorize this
        for i in range(self._nodes.size(0)):
            for j in range(i + 1, self._nodes.size(0)):
                for k in range(j + 1, self._nodes.size(0)):
                    # now compute the plane of the chosen three points
                    n = cross(self._nodes[j] - self._nodes[i], self._nodes[k] - self._nodes[i], dim=0)

                    # skip collinear points -> they can't make up a plane
                    if n.norm() < 1e-12:
                        continue
                    n /= n.norm()

                    # compute perpendicular distances of all points to the plane using the plane equation:
                    # (x - p0) · n = 0, < tol to avoid numerical errors
                    inliers = (abs((self._nodes - self._nodes[i]) @ n) < 1e-6).sum()

                    # if more points than our current largest number satisfy the condition, update
                    if inliers > best_inliers:
                        best_inliers = inliers
                        base_normal = n
                        base_p = self._nodes[i]

        # make sure we have an apex and not just a plane or something weird
        if base_normal is None:
            raise RuntimeError("No valid plane detected: the vertices may be collinear.")

        # compute final distance with the plane found for the pyramid base
        dists = abs((self._nodes - base_p) @ base_normal)

        # apex = point farthest from base plane
        self._apex_idx = argmax(dists).item()

    def _get_main_diagonal(self) -> None:
        """
        Determine the main diagonal of the quadrilateral base.

        :return: None. Updates ``self._diagonal_idx`` with the two diagonal indices
                 and ``self._off_diagonal`` with the remaining two base indices.
        :rtype: None
        """
        # select all points which are not identified as apex
        _idx = [i for i in range(self._nodes.size(0)) if i != self._apex_idx]
        _points = self._nodes[_idx]

        # compute the pair-wise Euclidean distance to all points
        diff = ((_points[:, None, :] - _points[None, :, :])** 2).sum(-1)

        # make sure we omit the distance of a point to itself
        diff.fill_diagonal_(-float('inf'))

        # take the distance which is max. and assume that the first entry is our main diagonal
        i, j = nonzero(diff == diff.max(), as_tuple=True)
        self._diagonal_idx = (_idx[i[0].item()], _idx[j[0].item()])
        self._off_diagonal = [i for i in _idx if i not in self._diagonal_idx]

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
        mask = self._mask_pyramid(cell_nodes)
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _mask_pyramid(self, vertices: Tensor) -> Tensor:
        """
        Select all vertices that are located inside a pyramid or on its surface.

        :param vertices: Tensor of vertices, where each column corresponds to a coordinate.
        :type vertices: pt.Tensor
        :return: Boolean mask that is ``True`` for every vertex inside the pyramid or on its surface.
        :rtype: pt.Tensor
        """
        # mask each tetrahedron
        masks = cat([tet.check_tetrahedron(vertices).unsqueeze(-1) for tet in self._tets], dim=1)

        # now we have to check, if the cells are 'True' in one tetrahedron but 'False' in the other one
        # -> if so, then we have to mask the node as 'True' since it is invalid if it's in both tetrahedrons
        return masks.any(dim=1)

    def _check_geometry(self) -> None:
        """
        Check the user input for correctness.

        :return: None
        :rtype: None
        """
        # make sure we have exactly 5 vertices
        assert len(self._nodes) == 5, (f"The pyramid must have exactly five vertices but found {len(self._nodes)} "
                                       f"vertices.")

        # make sure each vertex has three components and is a list or tuple. The remaining checks, e.g. volume > 0 are
        # carried out when instantiating the tetrahedrons
        for i, v in enumerate(self._nodes):
            assert isinstance(v, Union[list, tuple]), (f"Expected each vertex to be of type list or tuple but found "
                                                       f"type {type(v)} for vertex no. {i}.")
            assert len(v) == 3, (f"Expected each vertex to have exactly 3 components but found {len(v)} components "
                                 f"for entry {i}.")

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
        Return the width of the main dimension of the pyramid.

        :return: Main width of the pyramid.
        :rtype: float
        """
        return self._main_width

    @property
    def center(self) -> Tensor:
        """
        Return the center coordinates based on the main width of the pyramid.

        :return: center coordinates of the pyramid.
        :rtype: pt.Tensor
        """
        return self._center

    def _compute_main_width(self) -> float:
        """
        Compute the center coordinates based on the main width of the pyramid.

        :return: center coordinates of the pyramid.
        :rtype: pt.Tensor
        """
        return max([t.main_width for t in self._tets])

    def _compute_center(self) -> Tensor:
        """
        Compute the geometric center coordinates based on the main dimension of the pyramid.

        :return: center coordinates of the pyramid.
        :rtype: pt.Tensor
        """
        return cat([t.center.unsqueeze(-1) for t in self._tets], -1).mean(1)

if __name__ == "__main__":
    pass

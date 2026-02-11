"""
Implements a class for using an STL file (3D) as geometry object.
"""
import logging

from pyvista import PolyData, read
from typing import Union
from torch import Tensor, tensor
from os.path import join
from pymeshfix import MeshFix

from flowtorch.data import mask_box

from .geometry_base import GeometryObject

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)


class GeometrySTL3D(GeometryObject):
    __short_description__ = "usage of STL files for geometries (3D)"

    def __init__(self, name: str, keep_inside: bool, path_stl_file: str, refine: bool = False,
                 min_refinement_level: int = None, reduce_by: Union[int, float] = 0):
        """
        Implement a class for using an STL file as a geometry object representing the numerical domain
        or geometries inside the domain (3D case).

        Note:
            pyVista requires the STL file to have a closed surface.

        :param name: Name of the geometry object.
        :type name: str
        :param keep_inside: If ``True``, the points inside the object are kept; if ``False``, they are masked out.
        :type keep_inside: bool
        :param path_stl_file: Path to the STL file.
        :type path_stl_file: str
        :param refine: If ``True``, the mesh around the geometry object is refined after :math:`S^3` generates the mesh.
        :type refine: bool
        :param min_refinement_level: Minimum refinement level for resolving the geometry. If ``None`` and
            ``refine=True``, the geometry will be resolved with the maximum refinement level present at its surface
            after :math:`S^3` has generated the grid.
        :type min_refinement_level: int or None
        :param reduce_by: Factor to reduce the STL file. Recommended for larger STL files, as the number of points
            increases runtime significantly. A value of 0 means no compression; values between 0.9 and 0.98 typically
            work for most STL files. Must satisfy ``0 <= reduce_by < 1``.
        :type reduce_by: Union[int, float]
        """
        # make sure the compression factor is in a valid range
        if reduce_by < 0:
            logger.warning(f"Found invalid negative value for 'reduce_by' of {reduce_by}. Disabling compression.")
            reduce_by = 0
        elif reduce_by >= 1:
            logger.warning(f"Found invalid value for 'reduce_by' of {reduce_by}. Compression factor needs to be "
                           f"0 <= reduce_by < 1. Correcting 'reduce_by' to reduce_by=0.99")
            reduce_by = 0.99

        super().__init__(name, keep_inside, refine, min_refinement_level)
        self._stl_file = read(path_stl_file).decimate(reduce_by)
        self._type = "STL"
        self._pwd = path_stl_file

        # save reduced STL file
        if reduce_by > 0:
            logger.info(f"Saving reduced STL file to disk. The name of the reduced STL file is "
                        f"{'.'.join([self._pwd.split('.stl')[0], '_reduced_by_Scube.stl'])}")
            self._stl_file.save(join(".".join([self._pwd.split(".stl")[0], "_reduced_by_Scube.stl"])))

        # xmin, xmax, ymin, ymax, zmin, zmax
        self._lower_bound = list(self._stl_file.bounds)[::2]
        self._upper_bound = list(self._stl_file.bounds)[1::2]

        # we have to compute the main dimension and the midpoint if the name of the GeometryObject is domain
        self._main_width = self._compute_main_width()
        self._center = self._compute_center()

        # check the user input based on the specified settings
        self._check_geometry()

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
        # for 3D geometries represented by STL files, we need to mask using pyVista; here we don't check for closed
        # surface since we already did that on initialization
        n = PolyData(cell_nodes.numpy())

        # create a mask, the mask is expected to be always 'False' outside the geometry and always 'True' inside it
        # (independently if it is a geometry or domain)
        mask = tensor(n.select_enclosed_points(self._stl_file, check_surface=False)["SelectedPoints"]).bool()

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def pre_check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> bool:
        """
        Pre-check if a cell is within the rectangular bounding box of the geometry object.

        This method is faster than checking the polygon directly and is especially useful
        when generating large numbers of cells outside the bounding box.

        :param cell_nodes: Vertices of the cell to be checked.
        :type cell_nodes: pt.Tensor
        :param refine_geometry: If ``False``, cells are masked out while generating the grid.
            If ``True``, checks whether a cell is located in the vicinity of the geometry surface
            to refine it subsequently. This parameter is provided by :math:`S^3`.
        :type refine_geometry: bool
        :return: ``True`` if the cell is invalid, ``False`` if the cell is valid.
        :rtype: bool
        """
        mask = mask_box(cell_nodes, self._lower_bound, self._upper_bound)

        # check if the cell is valid or invalid
        return self._apply_mask(mask, refine_geometry=refine_geometry)

    def _check_geometry(self) -> None:
        """
        Check the user input for correctness.

        :return: None
        :rtype: None
        """
        # check if the STL file is closed and manifold
        test_data = PolyData([(0.0, 0.0, 0.0)])
        try:
            # pyVista will throw a RuntimeError if the surface is not closed and manifold
            _ = test_data.select_enclosed_points(self._stl_file, check_surface=True)
        except RuntimeError:
            logger.warning(f"Expected an STL file with a closed and manifold surface for geometry {self.name}. "
                           f"Attempting to close the STL file. This may lead to unexpected behavior. The closed STL "
                           f"file will be saved to disk. The name of the closed STL file is "
                           f"{'.'.join([self._pwd.split('.stl')[0], '_closed_by_Scube.stl'])}.")

            # now try to close it using pv.MeshFix()
            mFix = MeshFix(self._stl_file)
            mFix.repair()
            mFix.save(join(".".join([self._pwd.split(".stl")[0], "_closed_by_Scube.stl"])))

            # replace the STL with the fixed one
            self._stl_file = mFix.mesh

        # check if the fix worked, if not exit
        try:
            # pyVista will throw a RuntimeError if the surface is not closed and manifold
            _ = test_data.select_enclosed_points(self._stl_file, check_surface=True)
        except RuntimeError:
            logger.critical("Was not able to close the STL file.")
            exit(0)


        # print a warning if the STL file contains many points, since it will slow down S^3 significantly
        if self._stl_file.n_points > 5e4:
            logger.warning(f"STL files contains {self._stl_file.n_points} points. This will slow down the grid "
                           f"generation process significantly. Consider passing a value for `reduce_by` on "
                           f"instantiation to decrease the number of points within the STL file. The number of points "
                           f"should be < 5e4 to achieve a good performance.")

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
        Return the width of the main dimension of the cube.

        :return: Main width of the cube.
        :rtype: float
        """
        return self._main_width

    @property
    def center(self) -> Tensor:
        """
        Return the center coordinates based on the main width of the cube.

        :return: center coordinates of the cube.
        :rtype: pt.Tensor
        """
        return self._center

    def _compute_main_width(self) -> float:
        """
        Compute the center coordinates based on the main width of the cube.

        :return: center coordinates of the cube.
        :rtype: pt.Tensor
        """
        return max([abs(u - l) for l, u in zip(self._lower_bound, self._upper_bound)])

    def _compute_center(self) -> Tensor:
        """
        Compute the geometric center coordinates based on the main dimension of the cube.

        :return: center coordinates of the cube.
        :rtype: pt.Tensor
        """
        return (tensor(self._lower_bound) + tensor(self._upper_bound)) / 2.0

if __name__ == "__main__":
    pass

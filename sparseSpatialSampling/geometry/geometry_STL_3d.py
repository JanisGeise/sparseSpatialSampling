"""
    implements a class for using an STL file (3D) as geometry objects
"""
import logging

from pyvista import PolyData, read
from typing import Union
from torch import Tensor, tensor

from flowtorch.data import mask_box

from .geometry_base import GeometryObject

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)


class GeometrySTL3D(GeometryObject):
    """
    implements a class for using an STL file (3D)
    """
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

        # xmin, xmax, ymin, ymax, zmin, zmax
        self._lower_bound = list(self._stl_file.bounds)[::2]
        self._upper_bound = list(self._stl_file.bounds)[1::2]

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

    def pre_check_cell(self, cell_nodes: Tensor, refine_geometry: bool = False) -> Tensor:
        """
        method to pre-check if a cell is within a rectangular bounding box of the geometry object
        ->  much faster than check the STL directly if it is expected to generate large numbers of cells outside
        the bounding box

        :param cell_nodes: vertices of the cell which should be checked
        :type cell_nodes: pt.Tensor
        :param refine_geometry: flag if we are currently generating the grid (and mask out cells, False) or if we want
                                to check if a cell is located in the vicinity of the geometry surface (True) to refine
                                it subsequently. S^3 will provide this parameter.
        :type refine_geometry: bool
        :return: flag if the cell is valid ('False') or invalid ('True') based on the specified settings
        :rtype: bool
        """
        mask = mask_box(cell_nodes, self._lower_bound, self._upper_bound)

        # check if the cell is valid or invali
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

    @property
    def type(self) -> str:
        return self._type


if __name__ == "__main__":
    pass

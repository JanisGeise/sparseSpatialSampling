"""
    Implements the :math:`S^3` algorithm to generate grids for CFD data.
"""
import inspect
import textwrap
import logging
import torch as pt

from os.path import join
from typing import Union
from os import path, makedirs

from .s_cube import SamplingTree

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)


class SparseSpatialSampling:
    def __init__(self, coordinates: pt.Tensor, metric: pt.Tensor, geometry_objects: list, save_path: str,
                 save_name: str, grid_name: str = "grid_s_cube", uniform_levels: int = 5,
                 n_cells_max: Union[int, float] = None,
                 min_metric: float = 0.75, max_delta_level: bool = False, write_times: Union[str, list] = None,
                 n_cells_iter_start: int = None, n_cells_iter_end: int = None, n_jobs: int = 1,
                 relTol: Union[int, float] = 1e-3, reach_at_least: float = 0.75, pre_select_cells: bool = False):
        """
        Class for executing the :math:`S^3` algorithm.

        Note:
            The parameter ``geometry_objects`` needs to have at least one entry
            containing information about the domain.

        :param coordinates: Coordinates of the original grid
        :type coordinates: pt.Tensor
        :param metric: Quantity used as an indicator for refinement of a cell
        :type metric: pt.Tensor
        :param geometry_objects: Information about the domain and geometries in it;
            each geometry is passed as instance of class within the ``geometry`` module
        :type geometry_objects: list
        :param save_path: Path where the interpolated grid and data should be saved
        :type save_path: str
        :param save_name: Base name of the files (grid & data)
        :type save_name: str
        :param grid_name: Name of the grid (used in XDMF file)
        :type grid_name: str
        :param uniform_levels: Number of uniform refinement cycles to perform
        :type uniform_levels: int
        :param n_cells_max: Maximum number of cells of the grid; if not set,
            early stopping based on captured variance will be used
        :type n_cells_max: int | float | None
        :param min_metric: Percentage of variance of the metric the generated grid should
            capture w.r.t. the original grid. If None, the max. number of cells will be used as
            stopping criterion. If n_cells_max is also provided, min_metric will be ignored.
        :type min_metric: float
        :param max_delta_level: Constraint that two adjacent cells should have a maximum
            level difference of one
        :type max_delta_level: bool
        :param write_times: Numerical time steps of the simulation; if None, time steps need
            to be passed when calling the export method
        :type write_times: str | list[int | float | str] | None
        :param n_cells_iter_start: Number of cells to refine per iteration at the
            beginning; if None, defaults to 1% of the number of vertices in the original grid
        :type n_cells_iter_start: int | None
        :param n_cells_iter_end: Number of cells to refine per iteration at the end;
            if None, defaults to 5% of n_cells_iter_start
        :type n_cells_iter_end: int | None
        :param n_jobs: Number of CPUs to use; if None, all available CPUs will be used
        :type n_jobs: int
        :param relTol: Minimum improvement between two consecutive iterations
        :type relTol: int | float
        :param reach_at_least: Minimum percentage of the target metric / number of cells
            to reach before activating the relTol stopping criterion
        :type reach_at_least: float
        :param pre_select_cells: Optimization for geometry objects (e.g., ``GeometrySTL3D``,
            ``GeometryCoordinates2D``) that reduces runtime when bounding box volume is close
            to geometry volume
        :type pre_select_cells: bool
        :return: None
        :rtype: None
        """
        self.n_jobs = n_jobs
        self.coordinates = coordinates
        self.metric = metric
        self.save_path = save_path
        self.save_name = save_name
        self.grid_name = grid_name

        self.write_times = write_times if isinstance(write_times, list) else [write_times]

        # results we get from SamplingTree
        self.centers = None
        self.vertices = None
        self.faces = None
        self.n_dimensions = coordinates.squeeze().size(-1)
        self.size_initial_cell = None
        self.levels = None

        # properties only required by SamplingTree
        self._geometries = geometry_objects
        self._pre_select_cells = pre_select_cells
        self._level_bounds = int(uniform_levels)
        self._n_cells_max = n_cells_max if n_cells_max is None else int(n_cells_max)
        self._min_metric = min_metric
        self._max_delta_level = max_delta_level
        self._n_cells_iter_start = n_cells_iter_start if n_cells_iter_start is None else int(n_cells_iter_start)
        self._n_cells_iter_end = n_cells_iter_end if n_cells_iter_end is None else int(n_cells_iter_end)
        self._relTol = relTol
        self._reach_at_least = reach_at_least

        # check if the dicts for the geometry objects are correct
        self._check_input()

        # create SamplingTree
        self._sampling = SamplingTree(self.coordinates, self.metric, self._geometries, n_cells=self._n_cells_max,
                                      uniform_level=self._level_bounds, min_metric=self._min_metric,
                                      max_delta_level=self._max_delta_level, n_cells_iter_end=self._n_cells_iter_end,
                                      n_cells_iter_start=self._n_cells_iter_start, n_jobs=self.n_jobs,
                                      relTol=self._relTol, reach_at_least=self._reach_at_least,
                                      pre_select=self._pre_select_cells)

    def execute_grid_generation(self) -> None:
        """
        Execute the :math:`S^3` algorithm.

        :return: None
        :rtype: None
        """
        # create directory for data and final grid
        if not path.exists(self.save_path):
            makedirs(self.save_path)

        # execute grid generation
        self._sampling.refine()

        # save the mesh info file
        pt.save(self._sampling.data_final_mesh, join(self.save_path, f"mesh_info_{self.save_name}.pt"))

        # to make things easier, assign the values we need from S^3 directly as properties, then we can delete the
        # SamplingTree
        self.levels = self._sampling.all_levels
        self.centers = self._sampling.all_centers
        self.vertices = self._sampling.all_nodes
        self.faces = self._sampling.face_ids
        self.size_initial_cell = self._sampling.data_final_mesh["size_initial_cell"]

        # reset SamplingTree
        self._sampling = None

        # save the s_cube instance, in case we want to interpolate some other fields later, we can just load it without
        # the necessity to re-run the grid generation
        pt.save(self, join(self.save_path, f"s_cube_{self.save_name}.pt"))

    def _check_input(self) -> None:
        """
        Check the user input for :math:`S^3` for invalid settings and adjust or correct
        them if possible.

        :return: None
        :rtype: None
        """
        # check if the metric is 1D
        assert len(self.metric.size()) == 1, (f"The size of the metric must be a 1D tensor of the length "
                                              f"{self.coordinates.size(0)}. The size of the metric given is "
                                              f"{self.metric.size()}.")

        # make sure the target metric is not larger than one (if it should be used as stopping criteria)
        if self._n_cells_max is None:
            if self._min_metric > 1:
                logger.warning("A value of min_metric > 1 is invalid. Changed min_metric to 1.")

                # set the new target value for min. metric to one
                self._min_metric = self._min_metric if self._min_metric < 1 else 1

        # check if at least one geometry object is present and represents the domain (keep_inside = True)
        assert self._geometries, ("No geometries are provided. Please provide at least one geometry for the "
                                  "numerical domain.")
        assert any([g.keep_inside for g in self._geometries]), ("No geometry for the domain provided. At least one "
                                                                "geometry object must have 'keep_inside = True' "
                                                                "representing the numerical domain.")

        # correct invalid level bounds
        if self._level_bounds <= 0:
            # we need lower level >= 1, because otherwise the stopping criteria is not working
            logger.warning(f"Lower level bound of {self._level_bounds} is invalid. Changed lower level bound to 1.")
            self._level_bounds = 1


def list_geometries() -> None:
    """
    List all available geometry objects along with a short description of each.

    :return: None
    :rtype: None
    """
    from . import geometry
    from .geometry.geometry_base import GeometryObject

    # find all classes in geometry that are subclasses of GeometryObject
    classes = [obj for name, obj in inspect.getmembers(geometry, inspect.isclass)
               if issubclass(obj, GeometryObject) and obj is not GeometryObject]

    msg = ["\n\tAvailable geometry objects:", "\t---------------------------"]
    max_len = max(len(cls.__name__) for cls in classes)
    for cls in sorted(classes, key=lambda c: c.__name__):
        short_desc = getattr(cls, "__short_description__", "")
        short_desc = textwrap.shorten(short_desc, width=100, placeholder="…")
        msg.append(f"\t\t- {cls.__name__.ljust(max_len)} : {short_desc}")

    msg.append("\n\tFor a more detailed description check out the documentation.")
    logger.info("\n".join(msg))


if __name__ == "__main__":
    pass

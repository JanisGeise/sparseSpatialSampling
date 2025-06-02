"""
    implements wrapper function for executing the S^3 algorithm
"""
import logging
import torch as pt

from os.path import join
from typing import Union
from os import path, makedirs

from .s_cube import SamplingTree

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SparseSpatialSampling:
    def __init__(self, coordinates: pt.Tensor, metric: pt.Tensor, geometry_objects: list, save_path: str,
                 save_name: str, grid_name: str = "grid_s_cube", uniform_levels: int = 5, n_cells_max: int = None,
                 min_metric: float = 0.75, max_delta_level: bool = False, write_times: Union[str, list] = None,
                 n_cells_iter_start: int = None, n_cells_iter_end: int = None, n_jobs: int = 1,
                 relTol: Union[int, float] = None, reach_at_least: float = 0.75):
        """
        Class for executing the S^3 algorithm.

        Note: the parameter "geometry_objects" needs to have at least one entry containing information about the domain.

        :param coordinates: Coordinates of the original grid
        :param metric: quantity which should be used as an indicator for refinement of a cell
        :param geometry_objects: list with dict containing information about the domain and geometries in it;
                                 each geometry is passed in as dict.
        :param save_path: path where the interpolated grid and data should be saved to
        :param save_name: name of the files (grid & data)
        :param grid_name: name of the grid (used in XDMF file)
        :param uniform_levels: number of uniform refinement cycles to perform
        :param n_cells_max: max. number of cells of the grid, if not set then early stopping based on captured variance
                            will be used
        :param min_metric: percentage of variance of the metric the generated grid should capture (wrt the original
                           grid), if 'None' the max. number of cells will be used as stopping criteria
        :param max_delta_level: flag for setting the constraint that two adjacent cells should have a max. level
                                difference of one
        :param write_times: numerical time steps of the simulation, needs to be provided as
                            Union[int | float | str, list[int | float | str]].
                            If 'None', the time steps need to be passed when calling the export method
        :param n_cells_iter_start: number of cells to refine per iteration at the beginning. If 'None' then the value is
                                   set to 1% of the number of vertices in the original grid
        :param n_cells_iter_end: number of cells to refine per iteration at the end. If 'None' then the value is set to
                                 5% of _n_cells_iter_start
        :param n_jobs: number of CPUs to use for the KNN prediction
        :param relTol: min. improvement between two consecutive iterations, defaults to:
                        1e-3 (metric as stopping criterion) or
                        10 cells (N_cells as stopping criterion)
        :param reach_at_least: reach at least x% of the target metric / number of cells before activating the
                               relTol stopping criterion
        :return: None
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
                                      relTol=self._relTol, reach_at_least=self._reach_at_least)

    def execute_grid_generation(self) -> None:
        """
        executed the S^3 algorithm

        :return: instance of the Datawriter class containing the generated grid along with additional information & data
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
        Check the user input for S^3 for invalid settings and adjust / correct them if possible

        :return: None
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


if __name__ == "__main__":
    pass

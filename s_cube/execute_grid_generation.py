"""
    implements wrapper function for executing the S^3 algorithm
"""
import logging
import torch as pt

from os.path import join
from os import path, makedirs

from .export_data import DataWriter
from .geometry import GeometryObject
from .s_cube import SamplingTree

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SparseSpatialSampling:
    def __init__(self, coordinates: pt.Tensor, metric: pt.Tensor, geometry_objects: list, save_path: str,
                 save_name: str, grid_name: str = "grid_s_cube", level_bounds: tuple = (3, 25), n_cells_max: int = None,
                 refine_geometry: bool = True, min_metric: float = 0.9, to_refine: list = None,
                 max_delta_level: bool = False, write_times: list = None, min_level_geometry: int = None,
                 n_cells_iter_start: int = None, n_cells_iter_end: int = None):
        """
        Class for executing the S^3 algorithm.

        Note: the parameter "geometry_objects" needs to have at least one entry containing information about the domain.

        :param coordinates: Coordinates of the original grid
        :param metric: quantity which should be used as an indicator for refinement of a cell
        :param geometry_objects: list with dict containing information about the domain and geometries in it;
                                 each geometry is passed in as dict.
                                 The dict must contain the entries "name", "bounds", "type" and "is_geometry".
                                 "name": name of the geometry object;
                                 "bounds": boundaries of the geometry sorted as: [[xmin, ymin, zmin], [xmax, ymax, zmax]];
                                 "type": type of the geometry, either "cube" or "sphere";
                                 "is_geometry": flag if the geometry is the domain (False) or a geometry inside
                                 the domain (True)

        :param save_path: path where the interpolated grid and data should be saved to
        :param save_name: name of the files (grid & data)
        :param grid_name: name of the grid (used in XDMF file)
        :param level_bounds: Tuple with (min., max.) refinement level
        :param n_cells_max: max. number of cells of the grid, if not set then early stopping based on captured variance
                            will be used
        :param refine_geometry: flag for final refinement of the mesh around geometries to ensure same cell level
        :param min_metric: percentage of variance of the metric the generated grid should capture (wrt the original
                           grid), if 'None' the max. number of cells will be used as stopping criteria
        :param to_refine: names of the geometries which should be refined, if None all except the domain will be refined
        :param max_delta_level: flag for setting the constraint that two adjacent cell should have a max. level
                                difference of one
        :param write_times: numerical time steps of the simulation, needs to be provided as list[str]. If 'None', the
                            write times need to be provided after refinement (before exporting the fields)
        :param min_level_geometry: flag if the geometries should be resolved with a min. refinement level.
                                   If 'None' and 'smooth_geometry = True', the geometries are resolved with the max.
                                   refinement level encountered at the geometry. If a min. level is specified, all
                                   defined geometries will be refined with this level.
        :param n_cells_iter_start: number of cells to refine per iteration at the beginning. If 'None' then the value is
                                   set to 1% of the number of vertices in the original grid
        :param n_cells_iter_end: number of cells to refine per iteration at the end. If 'None' then the value is set to
                                 5% of _n_cells_iter_start
        :return: None
        """
        self._coordinates = coordinates
        self._metric = metric
        self._geometries = geometry_objects
        self._save_path = save_path
        self._save_name = save_name
        self._grid_name = grid_name
        self._level_bounds = level_bounds
        self._n_cells_max = n_cells_max
        self._refine_geometry = refine_geometry
        self._min_metric = min_metric
        self._to_refine = to_refine
        self._max_delta_level = max_delta_level
        self._write_times = write_times
        self._min_level_geometry = min_level_geometry
        self._n_cells_iter_start = n_cells_iter_start
        self._n_cells_iter_end = n_cells_iter_end
        self._sampling = None
        self._export = None

        # check if the dicts for the geometry objects are correct
        if not self._check_geometry_objects():
            exit()

        # correct invalid level bounds
        if self._level_bounds[0] == 0:
            assert self._level_bounds[0] < self._level_bounds[1], (f"The given level_bounds of {self._level_bounds}"
                                                                   f"are invalid. The lower level must be smaller than "
                                                                   f"the upper level.")

            # we need lower level >= 1, because otherwise the stopping criteria is not working
            logger.warning(f"lower bound of {self._level_bounds[0]} is invalid. Changed lower bound to 1.")
            self._level_bounds = (1, self._level_bounds[1])

        # create SamplingTree
        self._sampling = SamplingTree(self._coordinates, self._metric, n_cells=self._n_cells_max,
                                      level_bounds=self._level_bounds, smooth_geometry=self._refine_geometry,
                                      min_metric=self._min_metric, which_geometries=self._to_refine,
                                      max_delta_level=self._max_delta_level, n_cells_iter_end=self._n_cells_iter_end,
                                      n_cells_iter_start=self._n_cells_iter_start,
                                      min_refinement_geometry=self._min_level_geometry)

        # create the geometry objects
        self._add_geometries()

    def execute_grid_generation(self) -> DataWriter:
        """
        executed the S^3 algorithm

        :return: instance of the Datawriter class containing the generated grid along with additional information & data
        """
        # create directory for data and final grid
        if not path.exists(self._save_path):
            makedirs(self._save_path)

        # execute S^3
        self._sampling.refine()

        # create a DataWriter instance
        self._export = DataWriter(self._sampling.face_ids, self._sampling.all_nodes, self._sampling.all_centers,
                                  self._sampling.all_levels, save_dir=self._save_path, save_name=self._save_name,
                                  grid_name=self._save_name, times=self._write_times,
                                  domain_boundaries=[g["bounds"] for g in self._geometries if not g["is_geometry"]][0])

        # add the metric and mesh info
        self._export._metric = self._metric
        self._export.mesh_info = self._sampling.data_final_mesh

        # remove everything, which is not required anymore
        self._coordinates = None
        self._metric = None
        self._sampling = None

        # save everything
        self._save()

        return self._export

    def _check_geometry_objects(self) -> bool:
        """
        check if the dict for each given geometry object is correct

        :return: bool
        """
        # check if the list is empty:
        assert len(self._geometries) >= 1, ("No geometry found. At least one geometry (numerical domain) must to be "
                                            "specified.")

        # loop over all entries and check if all keys are present and match the required format
        counter = 0
        keys = {"name", "is_geometry", "bounds", "type"}
        for i, g in enumerate(self._geometries):
            assert all([k in set(g.keys()) for k in keys]), (f"The specified keys of geometry no. {i} do not match the"
                                                             "required keys '{'name', 'is_geometry', 'bounds', "
                                                             "'type'}'.")

            # check if we have exactly one domain specified
            if not g["is_geometry"]:
                counter += 1

            # check if the format of the bounds is correct
            if g["type"].lower() != "stl":
                assert len(
                    g["bounds"]) == 2, f"too many bounds for geometry no. {i} specified. Expected exactly 2 lists."

                # for sphere, we have only the radius as 2nd element
                if g["type"] != "sphere":
                    assert len(g["bounds"][0]) == len(g["bounds"][1]), (f"the size of the lower bounds for geometry "
                                                                        f"no. {i} does not match the size of the upper "
                                                                        f"bounds.")
                    assert all(min_val < max_val for min_val, max_val in
                               zip(g["bounds"][0],
                                   g["bounds"][1])), f"lower boundaries of geometry no. {i} > upper bounds."

        assert counter == 1, "More than one domain specified."
        return True

    def _add_geometries(self) -> None:
        """
        add the geometry objects o the sampling tree

        :return: None
        """
        for g in self._geometries:
            if g["type"].lower() == "stl":
                self._sampling.geometry.append(GeometryObject(lower_bound=None, upper_bound=None, obj_type=g["type"],
                                                              geometry=g["is_geometry"], name=g["name"],
                                                              _coordinates=g["coordinates"],
                                                              _dimensions=self._coordinates.size(1)))
            else:
                self._sampling.geometry.append(GeometryObject(lower_bound=g["bounds"][0], upper_bound=g["bounds"][1],
                                                              obj_type=g["type"], geometry=g["is_geometry"],
                                                              name=g["name"]))

    def _save(self):
        pt.save(self._export.mesh_info, join(self._save_path, f"mesh_info_{self._save_name}.pt"))
        pt.save(self._export, join(self._save_path, f"DataWriter_{self._save_name}.pt"))


if __name__ == "__main__":
    pass

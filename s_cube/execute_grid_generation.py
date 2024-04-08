"""
    implements wrapper function for the S^3 algorithm
"""
import torch as pt

from time import time
from os import path, makedirs
from typing import List

from s_cube.export_data import DataWriter
from s_cube.geometry import GeometryObject
from s_cube.s_cube import SamplingTree


def check_geometry_objects(_geometries) -> bool:
    """
    check if the dict for each given geometry object is correct

    :param _geometries: list containing the dict for each geometry
    :return: bool
    """
    # check if the list is empty:
    assert len(_geometries) >= 1, "No geometry found. At least one geometry (numerical domain) must to be specified."

    # loop over all entries and check if all keys are present and match the required format
    counter = 0
    for i, g in enumerate(_geometries):
        assert set(g.keys()) == {"name", "is_geometry", "bounds", "type"}, f"The specified keys of geometry no. {i}" \
                                                                           f" 'g.keys()' do not match the required keys" \
                                                                           f"'{'name', 'is_geometry', 'bounds', 'type'}'."

        # check if we have exactly one domain specified
        if not g["is_geometry"]:
            counter += 1

        # check if the format of the bounds are correct
        assert len(g["bounds"]) == 2, f"too many bounds for geometry no. {i} specified. Expected exactly 2 lists."

        # for sphere, we have only the radius as 2nd element
        if g["type"] != "sphere":
            assert len(g["bounds"][0]) == len(g["bounds"][1]), f"the size of the lower bounds for geometry no. {i} " \
                                                               f"does not match the size of the upper bounds."
            assert all(min_val < max_val for min_val, max_val
                       in zip(g["bounds"][0], g["bounds"][1])), f"lower boundaries of geometry no. {i} > upper bounds."

    assert counter == 1, "More than one domain specified."

    return True


def execute_grid_generation(coordinates: pt.Tensor, metric: pt.Tensor, _n_cells_max: int, _geometry_objects: List[dict],
                            _load_path: str, _save_path: str, _save_name: str, _grid_name: str,
                            _level_bounds: tuple = (3, 25)) -> None:
    """
    wrapper function for executing the S^3 algorithm. Note: the parameter "_geometry_objects" needs to have at least
    one entry containing information about the domain.

    :param coordinates: coordinates of the original grid
    :param metric: quantity which should be used as indicator for refinement of a cell
    :param _n_cells_max: max. number of cells of the grid
    :param _geometry_objects: list with dict containing information about the domain and geometries in it; each
                              geometry is passed in as dict. The dict must contain the entries:
                              "name", "bounds", "type" and "is_geometry".
                              "name": name of the geometry object;
                              "bounds": boundaries of the geometry sorted as: [[xmin, ymin, zmin], [xmax, ymax, zmax]];
                              "type": type of the geometry, either "cube" or "sphere";
                              "is_geometry": flag if the geometry is the domain (False) or a geometry inside the domain
                              (True)
    :param _load_path: path to the CFD data
    :param _save_path: path where the interpolated grid and data should be saved to
    :param _save_name: name of the files (grid & data)
    :param _grid_name: name of the grid (used in XDMF file)
    :param _level_bounds: Tuple with (min., max.) Level
    :return: None
    """
    # check if the dicts for the geometry objects are correct
    if not check_geometry_objects(_geometry_objects):
        exit()

    # coarsen the cube mesh based on the std. deviation of the pressure
    sampling = SamplingTree(coordinates, metric, n_cells=_n_cells_max, level_bounds=(3, 50), cells_per_iter=25)

    # add the cube and the domain
    for g in _geometry_objects:
        sampling.geometry.append(GeometryObject(lower_bound=g["bounds"][0], upper_bound=g["bounds"][1],
                                                obj_type=g["type"], geometry=g["is_geometry"], name=g["name"]))

    # create the grid
    sampling.refine()

    # compute the cell centers and vertices of each leaf cell
    sampling.compute_nodes_final_mesh()

    # create directory for plots
    if not path.exists(_save_path):
        makedirs(_save_path)

    # fit the pressure field onto the new mesh and export the data
    t_start = time()
    export_data = DataWriter(sampling.leaf_cells(), load_dir=_load_path, save_dir=_save_path,
                             domain_boundaries=[g["bounds"] for g in _geometry_objects if not g["is_geometry"]][0],
                             save_name=_save_name, grid_name=_save_name)
    export_data.export()
    print(f"Export required {round((time() - t_start), 3)} s.\n")


if __name__ == "__main__":
    pass

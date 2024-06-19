"""
    implements wrapper function for executing the S^3 algorithm and a function for loading fields from OpenFoam
"""
import logging
import torch as pt

from os import path, makedirs
from typing import Union
from flowtorch.data import FOAMDataloader, mask_box

from .export_data import DataWriter
from .geometry import GeometryObject
from .s_cube import SamplingTree

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def check_geometry_objects(_geometries: list) -> bool:
    """
    check if the dict for each given geometry object is correct

    :param _geometries: list containing the dict for each geometry
    :return: bool
    """
    # check if the list is empty:
    assert len(_geometries) >= 1, "No geometry found. At least one geometry (numerical domain) must to be specified."

    # loop over all entries and check if all keys are present and match the required format
    counter = 0
    keys = {"name", "is_geometry", "bounds", "type"}
    for i, g in enumerate(_geometries):
        assert all([k in set(g.keys()) for k in keys]), f"The specified keys of geometry no. {i} do not match the " \
                                                        f"required keys '{'name', 'is_geometry', 'bounds', 'type'}'."

        # check if we have exactly one domain specified
        if not g["is_geometry"]:
            counter += 1

        # check if the format of the bounds is correct
        if g["type"].lower() != "stl":
            assert len(g["bounds"]) == 2, f"too many bounds for geometry no. {i} specified. Expected exactly 2 lists."

            # for sphere, we have only the radius as 2nd element
            if g["type"] != "sphere":
                assert len(g["bounds"][0]) == len(g["bounds"][1]), f"the size of the lower bounds for geometry no. " \
                                                                   f"{i} does not match the size of the upper bounds."
                assert all(min_val < max_val for min_val, max_val in
                           zip(g["bounds"][0], g["bounds"][1])), f"lower boundaries of geometry no. {i} > upper bounds."

    assert counter == 1, "More than one domain specified."

    return True


def execute_grid_generation(coordinates: pt.Tensor, metric: pt.Tensor, _geometry_objects: list, _save_path: str,
                            _save_name: str, _grid_name: str, _level_bounds: tuple = (3, 25), _n_cells_max: int = None,
                            _refine_geometry: bool = True, _min_metric: float = 0.9, _to_refine: list = None,
                            _max_delta_level: bool = False, _write_times: list = None, _min_level_geometry: int = None,
                            _n_cells_iter_start: int = None, _n_cells_iter_end: int = None) -> DataWriter:
    """
    Wrapper function for executing the S^3 algorithm.

    Note: the parameter "_geometry_objects" needs to have at least one entry containing information about the domain.

    :param coordinates: Coordinates of the original grid
    :param metric: quantity which should be used as an indicator for refinement of a cell
    :param _geometry_objects: list with dict containing information about the domain and geometries in it; each
                              geometry is passed in as dict. The dict must contain the entries:
                              "name", "bounds", "type" and "is_geometry".
                              "name": name of the geometry object;
                              "bounds": boundaries of the geometry sorted as: [[xmin, ymin, zmin], [xmax, ymax, zmax]];
                              "type": type of the geometry, either "cube" or "sphere";
                              "is_geometry": flag if the geometry is the domain (False) or a geometry inside the domain
                              (True)
    :param _save_path: path where the interpolated grid and data should be saved to
    :param _save_name: name of the files (grid & data)
    :param _grid_name: name of the grid (used in XDMF file)
    :param _level_bounds: Tuple with (min., max.) Level
    :param _n_cells_max: max. number of cells of the grid, if not set then early stopping based on captured variance
                         will be used
    :param _refine_geometry: flag for final refinement of the mesh around geometries to ensure same cell level
    :param _min_metric: percentage of variance of the metric the generated grid should capture (wrt the original
                          grid), if 'None' the max. number of cells will be used as stopping criteria
    :param _to_refine: names of the geometries which should be refined, if None all except the domain will be refined
    :param _max_delta_level: flag for setting the constraint that two adjacent cell should have a max. level
                             difference of one
    :param _write_times: numerical time steps of the simulation, needs to be provided as list[str]. If 'None', the
                         write times need to be provided after refinement (before exporting the fields)
    :param _min_level_geometry: flag if the geometries should be resolved with a min. refinement level.
                                If 'None' and 'smooth_geometry = True', the geometries are resolved with the
                                max. refinement level encountered at the geometry. If a min. level is specified,
                                all defined geometries will be refined with this level.
    :param _n_cells_iter_start: number of cells to refine per iteration at the beginning. If 'None' then the value is
                                set to 1% of the number of vertices in the original grid
    :param _n_cells_iter_end: number of cells to refine per iteration at the end. If 'None' then the value is
                                set to 5% of _n_cells_iter_start
    :return: instance of the Datawriter class containing the generated grid along with additional information & data.
    """
    # check if the dicts for the geometry objects are correct
    if not check_geometry_objects(_geometry_objects):
        exit()

    if _level_bounds[0] == 0:
        # we need lower level >= 1, because otherwise the stopping criteria is not working
        logger.warning(f"lower bound of {_level_bounds[0]} is invalid. Changed lower bound to 1.")
        _level_bounds = (1, _level_bounds[1])

    # coarsen the cube mesh based on the std. deviation of the pressure
    sampling = SamplingTree(coordinates, metric, n_cells=_n_cells_max, level_bounds=_level_bounds,
                            smooth_geometry=_refine_geometry, min_metric=_min_metric, which_geometries=_to_refine,
                            max_delta_level=_max_delta_level, n_cells_iter_end=_n_cells_iter_end,
                            n_cells_iter_start=_n_cells_iter_start, min_refinement_geometry=_min_level_geometry)

    # add the cube and the domain
    for g in _geometry_objects:
        if g["type"].lower() == "stl":
            sampling.geometry.append(GeometryObject(lower_bound=None, upper_bound=None, obj_type=g["type"],
                                                    geometry=g["is_geometry"], name=g["name"],
                                                    _coordinates=g["coordinates"], _dimensions=coordinates.size(1)))
        else:
            sampling.geometry.append(GeometryObject(lower_bound=g["bounds"][0], upper_bound=g["bounds"][1],
                                                    obj_type=g["type"], geometry=g["is_geometry"], name=g["name"]))

    # create the grid
    sampling.refine()

    # create directory for data and final grid
    if not path.exists(_save_path):
        makedirs(_save_path)

    # fit the pressure field onto the new mesh and export the data
    _export_data = DataWriter(sampling.face_ids, sampling.all_nodes, sampling.all_centers, sampling.all_levels,
                              save_dir=_save_path, save_name=_save_name, grid_name=_save_name, times=_write_times,
                              domain_boundaries=[g["bounds"] for g in _geometry_objects if not g["is_geometry"]][0])

    # add the final data of mesh and refinement
    _export_data.mesh_info = sampling.data_final_mesh

    # add the metric
    _export_data._metric = sampling.target

    return _export_data


def load_original_Foam_fields(_load_dir: str, _n_dimensions: int, _boundaries: list,
                              _field_names: Union[list, str] = None, _write_times: Union[list, str] = None,
                              _get_field_names_and_times: bool = False):
    """
    function for loading fields from OpenFoam either for a single field, multiple fields at once, single time steps
    or multiple time steps at once.

    :param _load_dir: path to the original CFD data
    :param _n_dimensions: number of physical dimensions
    :param _boundaries: boundaries of the numerical domain, need to be the same as used for the execution of the S^3
    :param _field_names: names of the fields which should be exported
    :param _write_times: numerical time steps which should be exported, can be either a str or a list of str
    :param _get_field_names_and_times: returns available field names at first available time steps and write times
    :return: if _get_field_names_and_times = True: available field names at first available time steps and write times
             if False: the specified field, if field can't be found, 'None' is returned
    """
    # create foam loader object
    loader = FOAMDataloader(_load_dir)

    # check which fields and write times are available
    if _get_field_names_and_times:
        _write_times = [t for t in loader.write_times[1:]]
        return _write_times, loader.field_names[_write_times[0]]

    else:
        # load vertices
        vertices = loader.vertices if _n_dimensions == 3 else loader.vertices[:, :2]
        mask = mask_box(vertices, lower=_boundaries[0], upper=_boundaries[1])

        # the coordinates are independent of the field
        # stack the coordinates to tuples
        coord = pt.stack([pt.masked_select(vertices[:, d], mask) for d in range(_n_dimensions)], dim=1)

        # get all available time steps, skip the zero folder
        if _write_times is None:
            _write_times = [t for t in loader.write_times[1:]]
        elif type(_write_times) is str:
            _write_times = [_write_times]

        # in case there are no fields specified, take all available fields
        if _field_names is None:
            _field_names = loader.field_names[_write_times[0]]
        elif type(_field_names) is str:
            _field_names = [_field_names]

        # assemble data matrix for each field and interpolate the values onto the coarser grid
        _fields_out = []
        for field in _field_names:
            # determine if we have a vector or a scalar
            try:
                _field_size = loader.load_snapshot(field, _write_times[0]).size()
            except ValueError:
                logger.warning(f"\tField '{field}' is not available. Skipping this field...")
                continue

            if len(_field_size) == 1:
                data = pt.zeros((mask.sum().item(), len(_write_times)), dtype=pt.float32)
            else:
                data = pt.zeros((mask.sum().item(), _field_size[1], len(_write_times)), dtype=pt.float32)
                mask = mask.unsqueeze(-1).expand(_field_size)

            try:
                for i, t in enumerate(_write_times):
                    # load the field
                    if len(_field_size) == 1:
                        data[:, i] = pt.masked_select(loader.load_snapshot(field, t), mask)
                    else:
                        # we always need to export all dimensions of a vector, because for 2D we don't know in which
                        # plane the flow problem is defined
                        data[:, :, i] = pt.masked_select(loader.load_snapshot(field, t), mask).reshape([coord.size(0),
                                                                                                        _field_size[1]])

            # if fields are written out only for specific parts of the domain, this leads to dimension mismatch between
            # the field and the mask. The mask takes all cells in the specified area, but the field is only written out
            # in a part of this mask.
            except RuntimeError:
                logger.warning(f"\tField '{field}' is does not match the size of the masked domain. Skipping this "
                               f"field...")
                continue

            # since the size of data matrix must be: [N_cells, N_dimensions, N_snapshots] (vector field) or
            # [N_cells, 1, N_snapshots] (scalar field); unsqueeze if we have a scalar field
            if len(_field_size) == 1:
                data = data.unsqueeze(1)

            _fields_out.append([coord, data])

        if len(_fields_out) > 1:
            return _fields_out
        elif not _fields_out:
            return None, None
        else:
            return _fields_out[0]


def export_openfoam_fields(datawriter: DataWriter, load_path: str, boundaries: list,
                           batch_size: int = None, fields: Union[list, str] = None) -> None:
    """
    Wrapper function for interpolating the original CFD data executed with OpenFoam onto the generated grid with the
    S^3 algorithm. If the data was not generated with OpenFoam, the 'export_data()' method of the DataWriter class needs
    to be called directly with the CFD data and coordinates of the original grid as, e.g., implemented in
    'examples/s3_for_OAT15_airfoil.py'

    Important Note: this wrapper function only exports fields that are available in all time steps based on the
                    available fields in the first time step. If fields that are only present at every Nth time step
                    should be exported, this function cannot be used. Instead, the time steps and field name have to
                    be given directly to the export_data method as it is done in 'examples/s3_for_OAT15_airfoil.py'

                    E.g.:
                            times = [0.1, 0.2, 0.3] \n
                            export = execute_grid_generation(...) \n
                            export.times = times    # times needs to be a list of str\n
                            export.export_data(...) \n

    :param datawriter: Datawriter object resulting from the refinement with S^3
    :param load_path: path to the original CFD data
    :param boundaries: boundaries used for generating the mesh
    :param batch_size: batch size, number of snapshots which should be interpolated and exported at once. If 'None',
                       then all available snapshots will be exported at once
    :param fields: fields to export, either str or list[str]. If 'None' then all available fields at the first time
                   step will be exported
    :return: None
    """
    # get the available time steps and field names based on the fields available in the first time step
    if fields is None:
        _, fields = load_original_Foam_fields(load_path, datawriter.n_dimensions, boundaries,
                                              _get_field_names_and_times=True)

    # save time steps of all snapshots if not already provided when starting the refinement, needs to be list[str]
    if datawriter.times is None:
        times, _ = load_original_Foam_fields(load_path, datawriter.n_dimensions, boundaries,
                                             _get_field_names_and_times=True)
        datawriter.times = times

    batch_size = batch_size if batch_size is not None else len(datawriter.times)

    if type(fields) is str:
        fields = [fields]

    # interpolate and export the specified fields
    for f in fields:
        counter = 1
        if not len(datawriter.times) % batch_size:
            n_batches = int(len(datawriter.times) / batch_size)
        else:
            n_batches = int(len(datawriter.times) / batch_size) + 1

        for t in pt.arange(0, len(datawriter.times), step=batch_size).tolist():
            logger.info(f"Exporting batch {counter} / {n_batches}")
            coordinates, data = load_original_Foam_fields(load_path, datawriter.n_dimensions, boundaries,
                                                          _field_names=f, _write_times=datawriter.times[t:t+batch_size])

            # in case the field is not available, the export()-method will return None
            if data is not None:
                datawriter.export_data(coordinates, data, f, _n_snapshots_total=len(datawriter.times))
            counter += 1


if __name__ == "__main__":
    pass

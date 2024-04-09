"""
    implements wrapper function for the S^3 algorithm
"""
import torch as pt

from time import time
from os import path, makedirs
from typing import List
from typing import Union
from flowtorch.data import FOAMDataloader, mask_box

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


def execute_grid_generation(coordinates: pt.Tensor, metric: pt.Tensor, _geometry_objects: List[dict],
                            _load_path: str, _save_path: str, _save_name: str, _grid_name: str,
                            _level_bounds: tuple = (3, 25), _n_cells_max: int = None) -> DataWriter:
    """
    wrapper function for executing the S^3 algorithm. Note: the parameter "_geometry_objects" needs to have at least
    one entry containing information about the domain.

    :param coordinates: coordinates of the original grid
    :param metric: quantity which should be used as indicator for refinement of a cell
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
    :param _n_cells_max: max. number of cells of the grid, if not set then early stopping will be used
    :return: None
    """
    # check if the dicts for the geometry objects are correct
    if not check_geometry_objects(_geometry_objects):
        exit()

    # coarsen the cube mesh based on the std. deviation of the pressure
    sampling = SamplingTree(coordinates, metric, n_cells=_n_cells_max, level_bounds=(2, 50))

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
    print(f"Computation of cell faces and nodes required {round((time() - t_start), 3)} s.\n")

    return export_data


def load_original_Foam_fields(_load_dir: str, _n_dimensions: int, _boundaries: list,
                              _field_names: Union[list, str] = None, _write_times: list = None,
                              _get_field_names_and_times: bool = False):
    """
        function for loading fields from OpenFoam either for a single field, multiple fields at once, single time steps
        or multiple time steps at once. documentation coming soon ...
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
        if _n_dimensions == 2:
            coord = pt.stack([pt.masked_select(vertices[:, 0], mask), pt.masked_select(vertices[:, 1], mask)], dim=1)
        else:
            coord = pt.stack([pt.masked_select(vertices[:, 0], mask), pt.masked_select(vertices[:, 1], mask),
                              pt.masked_select(vertices[:, 2], mask)], dim=1)

        # get all available time steps, skip the zero folder
        if _write_times is None:
            _write_times = [t for t in loader.write_times[1:]]
        elif type(_write_times) == str:
            _write_times = [_write_times]

        # in case there are no fields specified, take all available fields
        _field_names = loader.field_names[_write_times[0]] if _field_names is None else _field_names

        # assemble data matrix for each field and interpolate the values onto the coarser grid
        _fields_out = []
        for field in _field_names:
            # determine if we have a vector or a scalar
            try:
                _field_size = loader.load_snapshot(field, _write_times[0]).size()
            except ValueError:
                print(f"\tField '{field}' is not available. Skipping this field...")
                continue

            if len(_field_size) == 1:
                data = pt.zeros((mask.sum().item(), len(_write_times)), dtype=pt.float32)
            else:
                data = pt.zeros((mask.sum().item(), _field_size[1], len(_write_times)), dtype=pt.float32)
                mask_vec = mask.unsqueeze(-1).expand(list(_field_size))
                out = list(data.size()[:-1])

            try:
                for i, t in enumerate(_write_times):
                    # load the field
                    if len(_field_size) == 1:
                        data[:, i] = pt.masked_select(loader.load_snapshot(field, t), mask)
                    else:
                        data[:, :, i] = pt.masked_select(loader.load_snapshot(field, t), mask_vec).reshape(out)

            # if fields are written out only for specific parts of domain, this leads to dimension mismatch between
            # the field and the mask (mask takes all cells in the specified area, but field is only written out in a
            # part of this mask
            except RuntimeError:
                print(f"\tField '{field}' is does not match the size of the masked domain. Skipping this field...")
                continue

            # interpolate the KNN for each field over all time steps (for now, later an option to fit N snapshots at
            # once will be implemented), IMPORTANT: size of data matrix must be:
            #   - [N_cells, N_dimensions, N_snapshots] (vector field)
            #   - [N_cells, 1, N_snapshots] (scalar field)
            if len(_field_size) == 1:
                data = data.unsqueeze(1)

            _fields_out.append([coord, data])

        return _fields_out if len(_fields_out) > 1 else _fields_out[0]


if __name__ == "__main__":
    pass

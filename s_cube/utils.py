"""
    Helper functions for loading openfoam fields and functions encapsulating the complete loading, interpolation, and
    export of fields.
    Further, a function to compute an SVD on the data from s_cube for a given datamatrix and weights.
"""
import logging
import torch as pt

from typing import Union, Tuple

from flowtorch.analysis import SVD
from flowtorch.data import FOAMDataloader, mask_box

from s_cube.export import ExportData

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

        # in case the time steps are not provided as list[str], convert them into list[str]
        if type(_write_times[0]) is not str:
            _write_times = list(map(str, _write_times))

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
                logger.warning(f"Field '{field}' is not available. Skipping field {field}.")
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
                        # we always need to export all dimensions of a vector, because for 2D we don't know on which
                        # plane the flow problem is defined
                        data[:, :, i] = pt.masked_select(loader.load_snapshot(field, t), mask).reshape([coord.size(0),
                                                                                                        _field_size[1]])

            # If fields are written out only for specific parts of the domain, this leads to dimension mismatch between
            # the field and the mask. The mask takes all cells in the specified area, but the field is only written out
            # in a part of this mask.
            except RuntimeError:
                logger.warning(f"Field '{field}' is does not match the size of the masked domain. Skipping "
                               f"field {field}.")
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


def export_openfoam_fields(datawriter: ExportData, load_path: str, boundaries: list,
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
                            export.write_times = times                  # times needs to be a list of str\n
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
    if datawriter.write_times is None:
        times, _ = load_original_Foam_fields(load_path, datawriter.n_dimensions, boundaries,
                                             _get_field_names_and_times=True)
        datawriter.write_times = times

    batch_size = batch_size if batch_size is not None else len(datawriter.write_times)

    if type(fields) is str:
        fields = [fields]

    # interpolate and export the specified fields
    for f in fields:
        counter = 1
        if not len(datawriter.write_times) % batch_size:
            n_batches = int(len(datawriter.write_times) / batch_size)
        else:
            n_batches = int(len(datawriter.write_times) / batch_size) + 1

        for t in pt.arange(0, len(datawriter.write_times), step=batch_size).tolist():
            logger.info(f"Exporting batch {counter} / {n_batches}")
            coordinates, data = load_original_Foam_fields(load_path, datawriter.n_dimensions, boundaries,
                                                          _field_names=f,
                                                          _write_times=datawriter.write_times[t:t + batch_size])

            # in case the field is not available, the export()-method will return None
            if data is not None:
                datawriter.export(coordinates, data, f, _n_snapshots_total=len(datawriter.write_times))
            counter += 1


def load_cfd_data(load_dir: str, boundaries: list, field_name="p", n_dims: int = 2, t_start: Union[int, float] = 0.4,
                  scalar: bool = True) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, list]:
    """
    load the specified field, mask out the defined area.

    Note: vectors are always loaded with all three components (even if the case is 3D) because we don't know on which
          plane the flow problem is defined.

    :param load_dir: path to the simulation data
    :param boundaries: list with list containing the upper and lower boundaries of the mask
    :param field_name: name of the field
    :param n_dims: number of physical dimensions
    :param t_start: starting point in time, all snapshots for which t >= t_start will be loaded
    :param scalar: flag if the field that should be loaded is a scalar- or a vector field
    :return: the field at each write time, x- & y- and z- coordinates (depending on 2D or 3D) of the vertices and all
             write times as list[str]
    """
    # create foam loader object
    _loader = FOAMDataloader(load_dir)

    # load vertices and discard z-coordinate (if 2D)
    vertices = _loader.vertices[:, :n_dims]

    # create a mask
    mask = mask_box(vertices, lower=boundaries[0], upper=boundaries[1])

    # assemble data matrix
    write_time = [t for t in _loader.write_times[1:] if float(t) >= t_start]

    # stack the coordinates to tuples and free up some space
    xyz = pt.stack([pt.masked_select(vertices[:, d], mask) for d in range(n_dims)], dim=1)
    del vertices

    # allocate empty data matrix
    if scalar:
        data = pt.zeros((mask.sum().item(), len(write_time)), dtype=pt.float32)
    else:
        data = pt.zeros((mask.sum().item(), n_dims, len(write_time)), dtype=pt.float32)

        # we always load the vector in 3 dimensions first, so we always need to expand in 3 dimensions
        mask = mask.unsqueeze(-1).expand([xyz.size(0), 3])

    for i, t in enumerate(write_time):
        # load the specified field
        if scalar:
            data[:, i] = pt.masked_select(_loader.load_snapshot(field_name, t), mask)
        else:
            data[:, :, i] = pt.masked_select(_loader.load_snapshot(field_name, t), mask).reshape(mask.size())

    # get the cell area
    _cell_area = _loader.weights.sqrt().unsqueeze(-1)

    return data, xyz, _cell_area, write_time


def compute_svd(data_matrix: pt.Tensor, cell_area, rank: int = None) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Computes an SVD for a given field, the field is weighted with the cell area.
    For more information on the determination of the optimal rank, it is referred to the flowtorch documentation:

    https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/flowtorch.analysis.html#flowtorch.analysis.svd.SVD.opt_rank

    :param data_matrix: The data matrix with the snapshots, it is expected that the lasst dimension represents the
                        temporal evolution
    :param cell_area: Area (2D) / Volume (3D) for each cell
    :param rank: Number of modes which should be used to compute the SVD, if 'None' then the optimal rank will be used
    :return: Tuple containing the singular values, modes and mode coefficients as (s, U, V)
    """
    # subtract the temporal mean
    _field_size = data_matrix.size()
    data_matrix -= pt.mean(data_matrix, dim=-1).unsqueeze(-1)

    if len(_field_size) == 2:
        # multiply by the sqrt of the cell areas to weight their contribution
        data_matrix *= cell_area.sqrt().unsqueeze(-1)

        # save either everything until the optimal rank or up to a user specified rank
        svd = SVD(data_matrix, rank=rank if rank is not None else data_matrix.size(-1))

        return svd.s, svd.U / cell_area.sqrt().unsqueeze(-1), svd.V

    else:
        # multiply by the sqrt of the cell areas to weight their contribution
        data_matrix *= cell_area.sqrt().unsqueeze(-1).unsqueeze(-1)

        # stack the data of all components for the SVD
        orig_shape = _field_size
        data_matrix = data_matrix.reshape((orig_shape[1] * orig_shape[0], orig_shape[-1]))

        # save either everything until the optimal rank or up to a user specified rank
        svd = SVD(data_matrix, rank=rank if rank is not None else data_matrix.size(-1))

        # reshape the data back to ux, uy, uz
        new_shape = (orig_shape[0], orig_shape[1], svd.rank)

        return svd.s, svd.U.reshape(new_shape) / cell_area.sqrt().unsqueeze(-1).unsqueeze(-1), svd.V

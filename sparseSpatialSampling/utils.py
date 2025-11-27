"""
Helper functions for loading OpenFOAM fields and for encapsulating the complete workflow of
loading, interpolating, and exporting fields.

Additionally, provides a function to compute the SVD of data from :math:`S^3` for a given
data matrix and associated weights.
"""
import logging
import torch as pt

from typing import Union, Tuple

from flowtorch.analysis import SVD
from flowtorch.data import FOAMDataloader, mask_box

from .data import Dataloader, Datawriter
from .export import ExportData

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_original_Foam_fields(load_dir: str, n_dimensions: int, boundaries: list,
                              field_names: Union[list, str] = None, write_times: Union[list, str] = None,
                              get_field_names_and_times: bool = False):
    """
    Load one or multiple OpenFOAM fields for arbitrary write times, with flexible options for returning either:
        - available field names and times, or
        - actual field data (scalar or vector).

    Differences from `load_foam_data`:
        - Supports **multiple fields** at once, or a single field.
        - Supports **custom lists of write times** instead of only filtering by t_start.
        - Can return only **metadata** (field names + times) without loading data
          via `get_field_names_and_times=True`.
        - Returns a **list of (coord, data)** pairs when multiple fields are requested.
        - Handles variable dimensionality of fields (scalar or vector) dynamically.

    Use this function if:
        - You need more general-purpose access to OpenFOAM data.
        - You want to query available fields/times before loading.
        - You want to load multiple fields in one call.

    Note:
        This function is used in `export_openfoam_fields` to easily export all fields and time steps to :math:`S^3` data formats.
        For loading OpenFOAM data, e.g., to pass it to :math:`S^3` for grid generation it is recommended to use `load_foam_data`.

    :param load_dir: Path to the original CFD data
    :type load_dir: str
    :param n_dimensions: Number of physical dimensions
    :type n_dimensions: int
    :param boundaries: Boundaries of the numerical domain, must match those used for the execution of :math:`S^3`
    :type boundaries: dict
    :param field_names: Names of the fields to be exported
    :type field_names: str | list[str]
    :param write_times: Numerical time steps to export; can be a string or a list of strings
    :type write_times: str | list[str]
    :param get_field_names_and_times: If True, return available field names at the first available time steps and write times
    :type get_field_names_and_times: bool
    :return:
            - If ``get_field_names_and_times=True``: Tuple of ``(write_times, field_names_at_first_time_step)``
            - If False:
                - Single field: Tuple of ``(coord, data)``
                - Multiple fields: List of tuples ``[(coord1, data1), (coord2, data2), ...]``
                - No matching fields: ``(None, None)``
    :rtype: Union[
        Tuple[list, list],
        Tuple[pt.Tensor, pt.Tensor],
        List[Tuple[pt.Tensor, pt.Tensor]],
        Tuple[None, None]
    ]
    """
    # create foam loader object
    loader = FOAMDataloader(load_dir)

    # check which fields and write times are available
    if get_field_names_and_times:
        write_times = [t for t in loader.write_times[1:]]
        return write_times, loader.field_names[write_times[0]]

    else:
        # load vertices
        vertices = loader.vertices if n_dimensions == 3 else loader.vertices[:, :2]
        mask = mask_box(vertices, lower=boundaries[0], upper=boundaries[1])

        # the coordinates are independent of the field
        # stack the coordinates to tuples
        coord = pt.stack([pt.masked_select(vertices[:, d], mask) for d in range(n_dimensions)], dim=1)

        if write_times is None:
            write_times = [t for t in loader.write_times[1:]]
        elif type(write_times) is str:
            write_times = [write_times]

        # in case the time steps are not provided as list[str], convert them into list[str]
        if type(write_times[0]) is not str:
            write_times = list(map(str, write_times))

        # in case there are no fields specified, take all available fields
        if field_names is None:
            field_names = loader.field_names[write_times[0]]
        elif type(field_names) is str:
            field_names = [field_names]

        # assemble data matrix for each field and interpolate the values onto the coarser grid
        _fields_out = []
        for field in field_names:
            # determine if we have a vector or a scalar
            try:
                _field_size = loader.load_snapshot(field, write_times[0]).size()
            except ValueError:
                logger.warning(f"Field '{field}' is not available. Skipping field {field}.")
                continue

            if len(_field_size) == 1:
                data = pt.zeros((mask.sum().item(), len(write_times)), dtype=pt.float32)
            else:
                data = pt.zeros((mask.sum().item(), _field_size[1], len(write_times)), dtype=pt.float32)
                mask = mask.unsqueeze(-1).expand(_field_size)

            try:
                for i, t in enumerate(write_times):
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
    Wrapper function for interpolating the original CFD data (from OpenFOAM) onto the grid generated by
    :math:`S^3`.

    If the data was not generated with OpenFOAM, the ``export_data()`` method of the ``DataWriter`` class
    needs to be called directly with the CFD data and coordinates of the original grid, as illustrated in
    `Tutorial 2 <https://github.com/JanisGeise/sparseSpatialSampling/blob/main/docs/source/tutorials/tutorial2_oat15_buffet.ipynb>`_.

    Important Note:
        This wrapper function only exports fields available at all time steps, based on the fields present
        in the first time step. If fields exist only at every Nth time step, this function cannot be used.
        In that case, the time steps and field name must be provided directly to the ``export_data`` method,
        as shown in Tutorial 2.

        Example usage without this function::

            times = [0.1, 0.2, 0.3]
            export = execute_grid_generation(...)
            export.write_times = times  # times needs to be a list of str
            export.export_data(...)

    :param datawriter: DataWriter object resulting from the refinement with :math:`S^3`
    :type datawriter: DataWriter
    :param load_path: Path to the original CFD data
    :type load_path: str
    :param boundaries: Boundaries used for generating the mesh
    :type boundaries: dict
    :param batch_size: Number of snapshots to interpolate and export at once. If ``None``, all snapshots are exported
                at once
    :type batch_size: int | None
    :param fields: Fields to export, either a string or a list of strings. If ``None``, all available fields at the
                first time step are exported
    :type fields: str | list[str] | None
    :return: None
    :rtype: None
    """
    # get the available time steps and field names based on the fields available in the first time step
    if fields is None:
        _, fields = load_original_Foam_fields(load_path, datawriter.n_dimensions, boundaries,
                                              get_field_names_and_times=True)

    # save time steps of all snapshots if not already provided when starting the refinement, needs to be list[str]
    if datawriter.write_times is None:
        times, _ = load_original_Foam_fields(load_path, datawriter.n_dimensions, boundaries,
                                             get_field_names_and_times=True)
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
                                                          field_names=f,
                                                          write_times=datawriter.write_times[t:t + batch_size])

            # in case the field is not available, the export()-method will return None
            if data is not None:
                datawriter.export(coordinates, data, f, n_snapshots_total=len(datawriter.write_times))
            counter += 1

def load_foam_data(load_dir: str, boundaries: list, field_name="p", n_dims: int = 2, t_start: Union[int, float] = 0.4,
                   scalar: bool = True) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, list]:
    """
    Load a single OpenFOAM field (scalar or vector) from all write times greater than or equal to ``t_start``.

    Differences from `load_original_Foam_fields`:
        - Designed for a **single field only**.
        - Time filtering is based on a numerical threshold (``t_start``),
          instead of requiring explicit lists of times.
        - Input requires explicit **scalar/vector flag** rather than inferring
          from field data.
        - Always returns the simulation **weights** in addition to field data
          and coordinates.
        - Has a simpler interface but less flexibility.

    Use this function if:
        - You want a streamlined way to load one field from a time window.
        - You already know whether the field is scalar or vector.
        - You specifically need the `weights` array from the loader.

    :param load_dir: Path to the simulation data
    :type load_dir: str
    :param boundaries: List of [lower, upper] boundaries defining the mask
    :type boundaries: list[list[float]]
    :param field_name: Name of the field to load
    :type field_name: str
    :param n_dims: Number of physical dimensions (2 for 2D, 3 for 3D)
    :type n_dims: int
    :param t_start: Starting time; all snapshots with ``time >= t_start`` will be loaded
    :type t_start: int | float
    :param scalar: Flag indicating whether the field is scalar (True) or vector (False)
    :type scalar: bool
    :return:
        - Tensor containing the field values at each write time
        - Tensor with x-, y-, and z-coordinates of the vertices (depending on 2D or 3D)
        - Tensor with the masked coordinates of the vertices
        - List of write times as strings
    :rtype: Tuple[pt.Tensor, pt.Tensor, pt.Tensor, list[str]]
    """
    # create foam loader object
    _loader = FOAMDataloader(load_dir)

    # load vertices and discard z-coordinate (if 2D)
    vertices = _loader.vertices[:, :n_dims]

    # create a mask
    mask = mask_box(vertices, lower=boundaries[0], upper=boundaries[1])

    # assemble data matrix
    write_time = sorted([t for t in _loader.write_times[1:] if float(t) >= t_start], key=lambda x: float(x))

    # stack the coordinates to tuples and free up some space
    xyz = pt.stack([pt.masked_select(vertices[:, d], mask) for d in range(n_dims)], dim=1)
    del vertices

    # allocate empty data matrix
    if scalar:
        data = pt.zeros((mask.sum().item(), len(write_time)), dtype=pt.float32)
    else:
        data = pt.zeros((mask.sum().item(), n_dims, len(write_time)), dtype=pt.float32)

        # we always load the vector in 3 dimensions first, so we always need to expand in 3 dimensions
        mask = mask.unsqueeze(-1).expand([_loader.vertices.size(0), 3])

    for i, t in enumerate(write_time):
        # load the specified field
        if scalar:
            data[:, i] = pt.masked_select(_loader.load_snapshot(field_name, t), mask)
        else:
            data[:, :, i] = pt.masked_select(_loader.load_snapshot(field_name, t), mask).reshape(data.size(0), 3)[:, :n_dims]

    return data, xyz, _loader.weights, write_time


def compute_svd(data_matrix: pt.Tensor, cell_area: pt.Tensor, rank: int = None) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Compute a weighted SVD for a given field, where the field is weighted by the cell area (2D) or volume (3D).

    For more information on determining the optimal rank, see the FlowTorch documentation:
    `flowtorch.analysis.svd.SVD.opt_rank <https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/flowtorch.analysis.html#flowtorch.analysis.svd.SVD.opt_rank>`_.

    :param data_matrix: Data matrix with snapshots; the last dimension is expected to represent temporal evolution
    :type data_matrix: pt.Tensor
    :param cell_area: Area (2D) or volume (3D) for each cell
    :type cell_area: pt.Tensor
    :param rank: Number of modes to compute the SVD. If ``None``, the optimal rank will be used
    :type rank: int | None
    :return: Tuple containing the singular values, modes, and mode coefficients as ``(s, U, V)``
    :rtype: Tuple[pt.Tensor, pt.Tensor, pt.Tensor]
    """

    # subtract the temporal mean
    _field_size = data_matrix.size()
    data_matrix -= pt.mean(data_matrix, dim=-1).unsqueeze(-1)

    if len(_field_size) == 2:
        # multiply by the sqrt of the cell areas to weight their contribution
        data_matrix *= cell_area.sqrt().unsqueeze(-1)

        # save either everything until the optimal rank or up to a user specified rank
        svd = SVD(data_matrix, rank=rank)

        return svd.s, svd.U / cell_area.sqrt().unsqueeze(-1), svd.V

    else:
        # multiply by the sqrt of the cell areas to weight their contribution
        data_matrix *= cell_area.sqrt().unsqueeze(-1).unsqueeze(-1)

        # stack the data of all components for the SVD
        orig_shape = _field_size
        data_matrix = data_matrix.reshape((orig_shape[1] * orig_shape[0], orig_shape[-1]))

        # save either everything until the optimal rank or up to a user specified rank
        svd = SVD(data_matrix, rank=rank)

        # reshape the data back to ux, uy, uz
        new_shape = (orig_shape[0], orig_shape[1], svd.rank)

        return svd.s, svd.U.reshape(new_shape) / cell_area.sqrt().unsqueeze(-1).unsqueeze(-1), svd.V


def write_svd_s_cube_to_file(field_names: Union[list, str], load_dir: str, file_name: str,
                             new_file: bool, n_modes: int = None, rank=None, t_start: Union[int, float] = 0) -> None:
    """
    Compute an SVD for multiple fields and export the results to HDF5 and XDMF for visualization (e.g., in ParaView).

    :param field_names: Names of the fields for which the SVD should be computed
    :type field_names: str | list[str]
    :param load_dir: Directory from which the results of :math:`S^3` should be loaded. The SVD results will be written
                     into the same directory
    :type load_dir: str
    :param file_name: Name of the file from which the data should be loaded. For clarity, '_svd' will be appended
                      to the output file containing the SVD results
    :type file_name: str
    :param new_file: Flag indicating whether all exported fields are saved in a single HDF5 file or each field is
                     written into a separate file
    :type new_file: bool
    :param n_modes: Number of modes to write to the file. If larger than available modes, all available modes will be written
    :type n_modes: int
    :param rank: Number of modes to use for computing the SVD. If ``None``, the optimal rank will be used
    :type rank: int | None
    :param t_start: Starting time; all snapshots with time >= t_start will be loaded
    :type t_start: int | float
    :return: None
    :rtype: None
    """
    if type(field_names) is str:
        field_names = [field_names]

    for f in field_names:
        logger.info(f"Performing SVD for field {f}.")

        _name = f"{file_name}_{f}" if new_file else file_name
        dataloader = Dataloader(load_dir, f"{_name}.h5")
        _write_times = sorted([t for t in dataloader.write_times if float(t) >= t_start], key=lambda x: float(x))

        # assemble a datamatrix for computing an SVD and perform an SVD weighted with cell areas
        s, U, V = compute_svd(dataloader.load_snapshot(f, _write_times), dataloader.weights, rank)

        # write the data to HDF5 & XDMF
        datawriter = Datawriter(load_dir, file_name + f"_{f}_svd.h5")

        # write the grid
        datawriter.write_grid(dataloader)

        # set the max. number of modes to write, if the specified number of modes is larger than the available modes,
        # then only write all available modes
        n_modes = U.size(-1) if n_modes is None else n_modes
        if n_modes > U.size(-1):
            logger.warning(f"Number of modes to write is set to {n_modes}, but found only {U.size(-1)} modes to write.")
            n_modes = U.size(-1)

        # write the modes as vectors, where each mode is treated as an independent vector
        for i in range(n_modes):
            if len(U.size()) == 2:
                datawriter.write_data(f"mode_{i + 1}", group="constant", data=U[:, i].squeeze())
            else:
                datawriter.write_data(f"mode_{i + 1}", group="constant", data=U[:, :, i].squeeze())

        # write the rest as tensor (not referenced in XDMF file anyway)
        datawriter.write_data("V", group="constant", data=V)
        datawriter.write_data("s", group="constant", data=s)
        datawriter.write_data("cell_area", group="constant", data=dataloader.weights)

        # write XDMF file
        datawriter.write_xdmf_file()

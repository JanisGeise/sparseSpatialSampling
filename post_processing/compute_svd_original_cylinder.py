"""
    compute the SVD of the cylinder3D_Re3900 simulation (original simulation), perform an SVD an export the modes, mode
    coefficients and singular values to HDF5 & XDMF files
"""
import h5py
import logging
import torch as pt

from typing import Tuple
from os.path import join
from os import path, makedirs
from flowtorch.analysis import SVD
from flowtorch.data import FOAM2HDF5, FOAMDataloader

from sparseSpatialSampling.data import Datawriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_svd(_field: pt.Tensor, _sqrt_cell_area: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
        computes an SVD of the interpolated field.

        For more information on the determination of the optimal rank, it is referred to the flowtorch documentation:

        https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/flowtorch.analysis.html#flowtorch.analysis.svd.SVD.opt_rank

        :param _field: data matrix containing all snapshots of the field
        :param _sqrt_cell_area: sqrt of the cell area, used for weighting
        :return: None
    """
    # subtract the temporal mean
    _field -= pt.mean(_field, dim=-1).unsqueeze(-1)

    if len(_field.size()) == 2:
        # multiply by the sqrt of the cell areas to weight their contribution
        _field *= _sqrt_cell_area
        svd = SVD(_field, rank=_field.size(-1))
        _modes = svd.U[:, :svd.opt_rank] / _sqrt_cell_area
    else:
        _field *= _sqrt_cell_area.unsqueeze(-1)

        # stack the data of all components for the SVD
        orig_shape = _field.size()
        _field = _field.reshape((orig_shape[1] * orig_shape[0], orig_shape[-1]))
        svd = SVD(_field, rank=_field.size(-1))

        # reshape the data back to ux, uy, uz
        new_shape = (orig_shape[0], orig_shape[1], svd.rank)
        _modes = svd.U.reshape(new_shape)[:, :, :svd.opt_rank] / _sqrt_cell_area.unsqueeze(-1)

    return _modes, svd.V, svd.s


def write_hfd5_for_svd(_centers, _vertices, _face_id, _modes, _mode_coefficients, _singular_values, _cell_area,
                       _save_dir, _save_name, _field_name: str, _n_modes=150) -> None:
    """
        Write the HDF5 file storing the results from the SVD.

        :param _centers: cell centers of the grid
        :param _vertices: vertices of the grid
        :param _face_id: information about the connection between the grid nodes
        :param _modes: modes from the SVD
        :param _mode_coefficients: mode coefficients from the SVD
        :param _singular_values: singular values from the SVD
        :param _cell_area: cell area
        :param _save_dir: directory in which the file should be saved to
        :param _save_name: name of the hdf5 file
        :param _field_name: name of the field for which the SVD should be computed
        :param _n_modes: number of modes to save
        :return: None
    """
    datawriter = Datawriter(save_path, f"{save_name}_{field_name}_svd.h5", mixed=True)

    # write the grid
    datawriter.write_data("faces", group="grid", data=_face_id)
    datawriter.write_data("vertices", group="grid", data=_vertices)
    datawriter.write_data("centers", group="grid", data=_centers)

    # if all snapshots of the interpolated fields are available, perform an SVD for each component of the
    # field (will be extended to arbitrary batch sizes once this is working)
    _n_modes = min(_n_modes, _modes.size(-1))
    for i in range(_n_modes):
        if len(_modes.size()) == 2:
            datawriter.write_data(f"mode_{i + 1}", group="constant", data=_modes[:, i].squeeze())
        else:
            datawriter.write_data(f"mode_{i + 1}", group="constant", data=_modes[:, :, i].squeeze())

    # write the rest as tensor (not referenced in XDMF file anyway)
    datawriter.write_data("V", group="constant", data=_mode_coefficients)
    datawriter.write_data("s", group="constant", data=_singular_values)
    datawriter.write_data("cell_area", group="constant", data=_cell_area)

    # write XDMF file
    datawriter.write_xdmf_file()


if __name__ == "__main__":
    # path to original cylinder3D simulation
    load_path = join("/media", "janis", "Elements", "Janis", "cylinder_3D_Re3900_tests", "cylinder_3D_Re3900")
    save_path = join("..", "run", "final_benchmarks", "cylinder3D_Re3900_local_TKE",
                     "test_SVD_original_cylinder")
    save_name = "cylinder3D_Re3900_original"

    # for which field should the SVD be performed?
    field_name = "U"

    # assuming the h5 file with the extracted grid coordinates is in the same directory as the CFD data.
    # if not present, then it will be created
    grid_file_name = "coordinates"

    # instantiate FOAMDataLoader
    loader = FOAMDataloader(load_path)
    times = loader.write_times[1:-18]

    # convert grid to HDF5 if the file doesn't exist yet
    if not path.exists(join(load_path, f"{grid_file_name}.h5")):
        converter = FOAM2HDF5(load_path)

        # load a field as placeholder to write the file
        converter.convert(f"{grid_file_name}.h5", [field_name], [times[0]])

        # remove everything but the mesh
        with h5py.File(join(load_path, f"{grid_file_name}.h5"), "a") as f:
            del f["variable"]

        del converter

    # load the CFD data in the given boundaries
    logger.info(f"Loading data matrix for field {field_name}.")
    cell_area = loader.weights.unsqueeze(-1)
    coord = loader.vertices
    field = loader.load_snapshot(field_name, times)
    del loader

    # perform SVD of the specified field
    logger.info(f"Performing SVD for field {field_name}.")
    U, V, s = compute_svd(field, cell_area)
    del field

    # load the original grid from HDF5 file
    logger.info(f"Loading coordinates from {join(load_path, f'{grid_file_name}.h5')}.")
    converted_coord = h5py.File(join(load_path, f"{grid_file_name}.h5"), "r")

    # we only need the faces and vertices
    faces = converted_coord.get("constant/connectivity")[()]
    shape = (int(faces.shape[0] / pow(2, coord.size(1))), pow(2, coord.size(1)))
    vertices = converted_coord.get("constant/vertices")[()]
    del converted_coord

    if not path.exists(save_path):
        makedirs(save_path)

    # write HDF5 & XDMF file
    logger.info(f"Writing HDF5 to {join(save_path, save_name)}.")
    write_hfd5_for_svd(coord, vertices, faces, U, V, s, cell_area, save_path, save_name, field_name)

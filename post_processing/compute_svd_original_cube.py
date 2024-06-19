"""
    compute the SVD of the surfaceMountedCube (original simulation), perform an SVD an export the modes, mode
    coefficients and singular values to HDF5 & XDMF files
"""
import h5py
import torch as pt

from typing import Tuple
from os.path import join
from os import path, makedirs
from flowtorch.analysis import SVD
from flowtorch.data import FOAM2HDF5

from examples.s3_for_cylinder2D import load_cfd_data


def compute_svd(_field: pt.Tensor, _sqrt_cell_area: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
        Computes an SVd of the interpolated field.

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


def write_hfd5_for_svd(_centers, _vertices, _face_id, _modes, _mode_coefficients, _singular_values, _sqrt_cell_area,
                       _save_dir, _save_name, _field_name: str) -> None:
    """
        Write the HDF5 file storing the results from the SVD.

        :param _centers: cell centers of the grid
        :param _vertices: vertices of the grid
        :param _face_id: information about the connection between the grid nodes
        :param _modes: modes from the SVD
        :param _mode_coefficients: mode coefficients from the SVD
        :param _singular_values: singular values from the SVD
        :param _sqrt_cell_area: sqrt of the cell area
        :param _save_dir: directory in which the file should be saved to
        :param _save_name: name of the hdf5 file
        :param _field_name: name of the field for which the SVD should be computed
        :return: None
    """
    _writer = h5py.File(join(_save_dir, f"{_save_name}_svd_{_field_name}.h5"), "w")
    _grid_data = _writer.create_group("grid")
    _grid_data.create_dataset("faces", data=_face_id)
    _grid_data.create_dataset("vertices", data=_vertices)
    _grid_data.create_dataset("centers", data=_centers)

    # if all snapshots of the interpolated fields are available, perform an SVD for each component of the
    # field (will be extended to arbitrary batch sizes once this is working)
    if len(_modes.size()) == 2:
        _writer.create_dataset("mode", data=_modes)
    else:
        dims = ["x", "y", "z"]
        for i in range(_modes.size(1)):
            _writer.create_dataset(f"mode_{dims[i]}", data=_modes[:, i, :].squeeze())

    # write the mode coefficients and singular values only to the HDF5 file (not XDMF)
    _writer.create_dataset("mode_coefficients", data=_mode_coefficients)
    _writer.create_dataset("singular_values", data=_singular_values)
    _writer.create_dataset("sqrt_cell_area", data=_sqrt_cell_area)

    # close hdf file
    _writer.close()


def write_xdmf_for_svd(n_dimensions: int, _n_faces: int, _n_vertices: int, _mode_size: pt.Size,
                       _sqrt_cell_area_size: pt.Size, _save_dir: str, _save_name: str, _grid_name: str,
                       _field_name: str) -> None:
    """
    Write the XDMF file referencing the modes resulting from the SVD in the corresponding HDF5 file.

    :param n_dimensions: number of physical dimensions
    :param _n_faces: number of faces (connections between nodes)
    :param _n_vertices: number of nodes
    :param _mode_size: amount of modes
    :param _sqrt_cell_area_size: amount of cells
    :param _save_dir: directory in which the file should be saved to
    :param _save_name: name of the XDMF file
    :param _grid_name: name of the grid
    :param _field_name: name of the field for which the SVD should be computed
    :return: None
    """
    # the connectivity format which flowtorch returns only support Mixed topology types
    _grid_type = "Mixed"
    _file_name = f"{_save_name}_svd_{_field_name}"
    _dims = "XY" if n_dimensions == 2 else "XYZ"

    _global_header = f'<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n<Xdmf Version="2.0">\n' \
                     f'<Domain>\n<Grid Name="{_grid_name}" GridType="Uniform">\n' \
                     f'<Topology TopologyType="{_grid_type}" NumberOfElements="{_n_faces}">\n' \
                     f'<DataItem Format="HDF" DataType="Int" Dimensions="{_n_faces}">\n'

    # write the corresponding XDMF file
    with open(join(_save_dir, f"{_file_name}.xdmf"), "w") as f_out:
        # write global header
        f_out.write(_global_header)

        # include the grid data from the HDF5 file
        f_out.write(f"{_file_name}.h5:/grid/faces\n")

        # write geometry part
        f_out.write(f'</DataItem>\n</Topology>\n<Geometry GeometryType="{_dims}">\n'
                    f'<DataItem Rank="2" Dimensions="{_n_vertices} {n_dimensions}" '
                    f'NumberType="Float" Precision="8" Format="HDF">\n')

        # write coordinates of vertices
        f_out.write(f"{_file_name}.h5:/grid/vertices\n")

        # write end tags
        f_out.write("</DataItem>\n</Geometry>\n")

        # write POD modes to the last time step
        if len(_mode_size) == 2:
            f_out.write(f'<Attribute Name="mode" AttributeType="Vector" Center="Cell">\n<DataItem NumberType="Float" '
                        f'Precision="8" Format="HDF" Dimensions="{_mode_size[0]} {_mode_size[-1]}">\n')
            f_out.write(f"{_file_name}.h5:/mode\n</DataItem>\n</Attribute>\n")
        else:
            for d in ["x", "y", "z"]:
                f_out.write(f'<Attribute Name="mode_{d}" AttributeType="Vector" Center="Cell">\n<DataItem '
                            f'NumberType="Float" Precision="8" Format="HDF" '
                            f'Dimensions="{_mode_size[0]} {_mode_size[-1]}">\n')
                f_out.write(f"{_file_name}.h5:/mode_{d}\n</DataItem>\n</Attribute>\n")

        # write the sqrt cell area
        f_out.write(f'<Attribute Name="sqrt_cell_area" AttributeType="Vector" Center="Cell">\n<DataItem '
                    f'NumberType="Float" Precision="8" Format="HDF" Dimensions='
                    f'"{_sqrt_cell_area_size[0]} {_sqrt_cell_area_size[-1]}">\n')
        f_out.write(f"{_file_name}.h5:/sqrt_cell_area\n</DataItem>\n</Attribute>\n</Grid>\n</Domain>\n</Xdmf>")


if __name__ == "__main__":
    # path to original surfaceMountedCube simulation
    load_path = join("..", "data", "3D", "surfaceMountedCube_original_grid_size", "fullCase")
    save_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube",
                     "results_SVD_cube_2000_snapshots")
    save_name = "cube_original_grid"

    # for which field should the SVD be performed?
    field_name = "p"
    scalar_field = True

    # assuming the h5 file with the extracted grid coordinates is in the same directory as the CFD data. If not present,
    # then it will be created
    grid_file_name = "coordinates"

    # load the CFD data in the given boundaries
    bounds = [[0, 0, 0], [14.5, 9, 2]]  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    field, coord, cell_area, write_times = load_cfd_data(load_path, bounds, n_dims=3, field_name=field_name,
                                                         scalar=scalar_field)

    # perform SVD of the pressure field
    U, V, s = compute_svd(field, cell_area)
    del field

    # convert grid to HDF5 if file not exists yet
    if not path.exists(join(load_path, f"{grid_file_name}.h5")):
        converter = FOAM2HDF5(load_path)

        # load a dummy field
        converter.convert(f"{grid_file_name}.h5", [field_name], [write_times[0]])

        # remove everything but the mesh
        with h5py.File(join(load_path, f"{grid_file_name}.h5"), "a") as f:
            del f["variable"]

    # load the original grid from HDF5 file
    converted_coord = h5py.File(join(load_path, f"{grid_file_name}.h5"), "r")

    # we only need the faces and vertices
    faces = converted_coord.get("constant/connectivity")[()]
    shape = (int(faces.shape[0] / pow(2, coord.size(1))), pow(2, coord.size(1)))
    vertices = converted_coord.get("constant/vertices")[()]
    del converted_coord

    if not path.exists(save_path):
        makedirs(save_path)

    # write HDF5 file
    write_hfd5_for_svd(coord, vertices, faces, U, V, s, cell_area, save_path, save_name, field_name)

    # write XDMF file
    write_xdmf_for_svd(coord.size(-1), faces.shape[0], vertices.shape[0], U.size(), cell_area.size(), save_path,
                       save_name, "cube", field_name)

"""
    load the exported grids, data. Perform SVD and write the results to HDF5
    # TODO: del once everything else is refactored and working
"""
import h5py
import logging
import torch as pt

from os.path import join

from flowtorch.analysis import SVD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataLoader:
    def __init__(self):
        # properties loaded from HDF5 file
        self.cell_area = None
        self.data_matrix = None
        self.n_dimensions = None
        self._centers = None
        self._vertices = None
        self._face_id = None
        self._size_initial_cell = None
        self._n_vertices = None
        self._n_faces = None

        # properties for SVD
        self.U = None
        self.V = None
        self.s = None
        self._field_name = None

    def load_data(self, load_path: str, file_name: str, field_name: str, write_times: list = None) -> None:
        """
        Reconstruct the full data matrix of the interpolated field from the temporal grid structure inside the HDF5
        file. The data matrix is stored in the 'data_matrix' property.

        Note: the provided field name hass to be the same as used for interpolating the field onto the generated grid.

        :param load_path: path to the HDF5 file containing the interpolated field
        :param file_name: name of the HDF5 file from which the data should be loaded
        :param field_name: name of the field which should be loaded
        :param write_times: write times which should be loaded. If 'None', all available write times will be loaded
        :return: None
        """
        self._field_name = field_name
        try:
            hdf_file = h5py.File(join(load_path, f"{file_name}.h5"), "r")
        except FileNotFoundError:
            logger.error(f"HDF5 file with the interpolated field {self._field_name} not found. Make sure the specified "
                         f"field exists.")
            exit(0)

        # assemble the data matrix
        keys = list(hdf_file[f"{self._field_name}_center"].keys()) if write_times is None else write_times
        shape = hdf_file[f"{self._field_name}_center"][keys[0]].shape

        # assign the grid
        self._centers = pt.from_numpy(hdf_file.get("grid/centers")[()])
        self._vertices = pt.from_numpy(hdf_file.get("grid/vertices")[()])
        self._face_id = pt.from_numpy(hdf_file.get("grid/faces")[()])

        # assign the dimensions for all tensors
        self.n_dimensions = self._centers.size(-1)
        self._n_vertices = self._vertices.size()[0]
        self._n_faces = self._face_id.size()[0]

        # compute the cell area
        self._size_initial_cell = hdf_file.get("size_initial_cell")[()]
        _levels = pt.from_numpy(hdf_file.get("levels")[()])
        self.cell_area = (1 / pow(2, self.n_dimensions) * pow(self._size_initial_cell / pow(2, _levels),
                                                              self.n_dimensions))

        # assemble the data matrix
        if len(shape) == 1:
            self.data_matrix = pt.zeros((shape[0], len(keys)), dtype=pt.float32)
        else:
            self.data_matrix = pt.zeros((shape[0], shape[1], len(keys)), dtype=pt.float32)
        for i, k in enumerate(keys):
            if len(shape) == 1:
                self.data_matrix[:, i] = pt.from_numpy(hdf_file.get(f"{field_name}_center/{k}")[()])
            else:
                self.data_matrix[:, :, i] = pt.from_numpy(hdf_file.get(f"{field_name}_center/{k}")[()])

    def compute_svd(self, save_rank: int = None):
        """
        Computes a full SVD of the interpolated field.
        It is assumed that the field was already interpolated and exported completely (all snapshots) to HDF5 file.
        Although the SVD is performed on the complete dataset, only the modes until the optimal rank are saved.
        The singular values and left singular vectors are stored in full.
        For more information on the determination of the optimal rank, it is referred to the flowtorch documentation:

        https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/flowtorch.analysis.html#flowtorch.analysis.svd.SVD.opt_rank

        :param save_rank: number of modes which should be saved to HDF5, if 'None' then all modes up the optimal rank
                          are stored
        :return: None
        """
        logger.info(f"Computing SVD for field {self._field_name}.")

        # subtract the temporal mean
        _field = self.data_matrix - pt.mean(self.data_matrix, dim=-1).unsqueeze(-1)

        if len(_field.size()) == 2:
            # multiply by the sqrt of the cell areas to weight their contribution
            _field *= self.cell_area.sqrt()
            svd = SVD(_field, rank=_field.size(-1))

            # save either everything until the optimal rank or up to a user specified rank
            save_rank = save_rank if save_rank is not None else svd.opt_rank

            self.U = svd.U[:, :save_rank] / self.cell_area.sqrt()
        else:
            _field *= self.cell_area.sqrt().unsqueeze(-1)

            # stack the data of all components for the SVD
            orig_shape = _field.size()
            _field = _field.reshape((orig_shape[1] * orig_shape[0], orig_shape[-1]))
            svd = SVD(_field, rank=_field.size(-1))

            # save either everything until the optimal rank or up to a user specified rank
            save_rank = save_rank if save_rank is not None else svd.opt_rank

            # reshape the data back to ux, uy, uz
            new_shape = (orig_shape[0], orig_shape[1], svd.rank)
            self.U = svd.U.reshape(new_shape)[:, :, :save_rank] / self.cell_area.sqrt().unsqueeze(-1)
        self.V = svd.V
        self.s = svd.s

    def write_data(self, save_path: str, file_name: str, U: pt.Tensor = None, V: pt.Tensor = None,
                   s: pt.Tensor = None, n_modes: int = None) -> None:
        """
        Write the HDF5 and corresponding XDMF file storing the results of the SVD. If the results of the SVD are not
        passed as arguments, it is assumed that the SVD using the 'compute_svd()' method was already executed.

        :param save_path: path to the directory in which the file should be saved to
        :param file_name: name of the HDF5 & XDMF file in which the results should be saved in
        :param U: left singular vektors (optional)
        :param V: right singular vectors (optional)
        :param s: singular values (optional)
        :param n_modes: number of modes to be written into the HDF5 file, if 'None' all available modes will be written
        :return: None
        """
        # assign the results of the SVD
        if U is not None:
            self.U = U
            del U
        if V is not None:
            self.V = V
            del V
        if s is not None:
            self.s = s
            del s

        # create the grid
        _writer = h5py.File(join(save_path, f"{file_name}.h5"), "w")
        _grid_data = _writer.create_group("grid")
        _grid_data.create_dataset("faces", data=self._face_id)
        _grid_data.create_dataset("vertices", data=self._vertices)
        _grid_data.create_dataset("centers", data=self._centers)

        # write the modes as vectors, where each mode is treated as an independent vector.
        n_modes = self.U.size(-1) if n_modes is None else n_modes
        for i in range(n_modes):
            if len(self.U.size()) == 2:
                _writer.create_dataset(f"mode_{i}", data=self.U[:, i].squeeze())
            else:
                _writer.create_dataset(f"mode_{i}", data=self.U[:, :, i].squeeze())

        # write the mode coefficients and singular values only to the HDF5 file (not XDMF)
        _writer.create_dataset("mode_coefficients", data=self.V)
        _writer.create_dataset("singular_values", data=self.s)
        _writer.create_dataset("cell_area", data=self.cell_area)

        # close hdf file
        _writer.close()

        # write the corresponding XDMF file
        self._write_xdmf_file(save_path, file_name)

    def _write_xdmf_file(self, save_path: str, file_name: str, grid_name: str = "grid") -> None:
        """
        Write the XDMF file referencing the modes of the SVD in the corresponding HDF5 file.

        :param save_path: path to the directory in which the file should be saved to (same as for the HDF5 file)
        :return: None
        """
        _grid_type = "Quadrilateral" if self.n_dimensions == 2 else "Hexahedron"
        _dims = "XY" if self.n_dimensions == 2 else "XYZ"

        _global_header = f'<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n<Xdmf Version="2.0">\n' \
                         f'<Domain>\n<Grid Name="{grid_name}" GridType="Uniform">\n' \
                         f'<Topology TopologyType="{_grid_type}" NumberOfElements="{self._n_faces}">\n' \
                         f'<DataItem Format="HDF" DataType="Int" Dimensions="{self._n_faces} ' \
                         f'{pow(2, self.n_dimensions)}">\n'

        # write the corresponding XDMF file
        with open(join(save_path, f"{file_name}.xdmf"), "w") as f_out:
            # write global header
            f_out.write(_global_header)

            # include the grid data from the HDF5 file
            f_out.write(f"{file_name}.h5:/grid/faces\n")

            # write geometry part
            f_out.write(f'</DataItem>\n</Topology>\n<Geometry GeometryType="{_dims}">\n'
                        f'<DataItem Rank="2" Dimensions="{self._n_vertices} {self.n_dimensions}" '
                        f'NumberType="Float" Precision="8" Format="HDF">\n')

            # write coordinates of vertices
            f_out.write(f"{file_name}.h5:/grid/vertices\n")

            # write end tags
            f_out.write("</DataItem>\n</Geometry>\n")

            # write POD modes
            for m in range(self.U.size(-1)):
                f_out.write(f'<Attribute Name="mode_{m}" AttributeType="Vector" Center="Cell">\n<DataItem '
                            f'NumberType="Float" Precision="8" Format="HDF" Dimensions='
                            f'"{self.U.size(0)} {self.U.size(1)}">\n')
                f_out.write(f"{file_name}.h5:/mode_{m}\n</DataItem>\n</Attribute>\n")

            # write the sqrt cell area
            f_out.write(f'<Attribute Name="cell_area" AttributeType="Vector" Center="Cell">\n<DataItem '
                        f'NumberType="Float" Precision="8" Format="HDF" Dimensions='
                        f'"{self.cell_area.size(0)} {self.cell_area.size(-1)}">\n')
            f_out.write(f"{file_name}.h5:/cell_area\n</DataItem>\n</Attribute>\n</Grid>\n</Domain>\n</Xdmf>")


if __name__ == "__main__":
    pass

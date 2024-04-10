"""
    load the CFD data for the specified (or all available) fields and interpolate them onto the coarse grid, sampled
    with the s^3 algorithm. Export the interpolated data to HDF5 and XDMF in order to be able to load them into paraview
"""
import h5py
import numpy as np
import torch as pt

from os.path import join
from numba import njit, prange
from sklearn.neighbors import KNeighborsRegressor


class Fields:
    def __init__(self):
        pass


class DataWriter:
    def __init__(self, final_grid: list, load_dir: str, save_dir: str, domain_boundaries: list,
                 field_names: list = None, save_name: str = "data_final_grid", grid_name: str = "final grid"):
        self._save_dir = save_dir
        self._save_name = save_name
        self._grid_name = grid_name
        self._centers = pt.stack([cell.center for cell in final_grid], dim=0)
        self._vertices = pt.cat([cell.nodes for cell in final_grid]).unique(dim=0)
        self._faces = pt.cat([cell.nodes.unsqueeze(0) for cell in final_grid], dim=0)
        self._n_vertices = self._vertices.size()[0]
        self._n_faces = self._faces.size()[0]
        self._n_dimensions = self._vertices.size()[1]
        self.times = None

        # compute and resort the cell faces for loading in paraview
        print("Computing cell faces and nodes of final grid...")
        self._faces = resort_grid(self._faces.numpy(), self._vertices.numpy(), self._n_dimensions)
        print("Done.")

        # empty dict which stores the interpolated fields, created when calling the '_load_and_fit_data' method, if no
        # fields are specified all available fields will be interpolated onto the coarser mesh
        self._load_dir = load_dir
        self._field_names = field_names
        self._boundaries = domain_boundaries
        self._knn = KNeighborsRegressor(n_neighbors=8 if self._n_dimensions == 2 else 26, weights="distance")
        self._interpolated_fields = {}
        self._snapshot_counter = 0

    def _write_data(self):
        _global_header = f'<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n<Xdmf Version="2.0">\n' \
                         f'<Domain>\n<Grid Name="{self._grid_name}" GridType="Collection" CollectionType="temporal">\n'
        _grid_type = "Quadrilateral" if self._n_dimensions == 2 else "Hexahedron"
        _dims = "XY" if self._n_dimensions == 2 else "XYZ"

        # create a writer and datasets for the grid
        _writer = h5py.File(join(self._save_dir, f"{self._save_name}.h5"), "w")
        _grid_data = _writer.create_group("grid")
        _grid_data.create_dataset("faces", data=self._faces)
        _grid_data.create_dataset("vertices", data=self._vertices)

        # create group for each specified field
        for key in self._interpolated_fields.keys():
            _vertices = _writer.create_group(f"{key}_vertices")
            _center = _writer.create_group(f"{key}_center")

            # write the datasets for each time step
            for i, t in enumerate(self.times):
                # in case we have a scalar
                if len(self._interpolated_fields[key].centers.size()) == 2:
                    _center.create_dataset(str(t), data=self._interpolated_fields[key].centers[:, i])
                    _vertices.create_dataset(str(t), data=self._interpolated_fields[key].vertices[:, i])
                # in case we have a vector
                else:
                    _center.create_dataset(str(t), data=self._interpolated_fields[key].centers[:, :, i])
                    _vertices.create_dataset(str(t), data=self._interpolated_fields[key].vertices[:, :, i])

        # write global header of XDMF file
        with open(join(self._save_dir, f"{self._save_name}.xdmf"), "w") as f_out:
            f_out.write(_global_header)

        # loop over all available time steps and write all specified data to XDMF & HDF5 files
        for i, t in enumerate(self.times):
            with open(join(self._save_dir, f"{self._save_name}.xdmf"), "a") as f_out:
                # write grid specific header
                tmp = f'<Grid Name="{self._grid_name} {t}" GridType="Uniform">\n<Time Value="{t}"/>\n' \
                      f'<Topology TopologyType="{_grid_type}" NumberOfElements="{self._n_faces}">\n' \
                      f'<DataItem Format="HDF" DataType="Int" Dimensions="{self._n_faces} ' \
                      f'{pow(2, self._n_dimensions)}">\n'
                f_out.write(tmp)

                # include the grid data from the HDF5 file
                f_out.write(f"{self._save_name}.h5:/grid/faces\n")

            # write geometry part
            with open(join(self._save_dir, f"{self._save_name}.xdmf"), "a") as f_out:
                # write header for geometry
                f_out.write(f'</DataItem>\n</Topology>\n<Geometry GeometryType="{_dims}">\n'
                            f'<DataItem Rank="2" Dimensions="{self._n_vertices} {self._n_dimensions}" '
                            f'NumberType="Float" Format="HDF">\n')

                # write coordinates of vertices
                f_out.write(f"{self._save_name}.h5:/grid/vertices\n")

                # write end tags
                f_out.write("</DataItem>\n</Geometry>\n")

            # write the interpolated fields
            with open(join(self._save_dir, f"{self._save_name}.xdmf"), "a") as f_out:
                for key in self._interpolated_fields.keys():
                    # determine 2nd dimension (scalar vs. vector)
                    if len(self._interpolated_fields[key].centers.size()) == 2:
                        _second_dim = 1
                    else:
                        _second_dim = self._interpolated_fields[key].centers.size()[1]

                    # write header
                    f_out.write(f'<Attribute Name="{key}" Center="Cell">\n<DataItem Format="HDF" Dimensions='
                                f'"{self._interpolated_fields[key].centers.size()[0]} {_second_dim}">\n')

                    # write interpolated field at cell center
                    f_out.write(f"{self._save_name}.h5:/{key}_center/{t}\n")
                    f_out.write("</DataItem>\n</Attribute>\n")

                    # then do the same for field at the vertices
                    f_out.write(f'<Attribute Name="{key}" Center="Node">\n<DataItem Format="HDF" Dimensions='
                                f'"{self._interpolated_fields[key].vertices.size()[0]} {_second_dim}">\n')
                    f_out.write(f"{self._save_name}.h5:/{key}_vertices/{t}\n")
                    f_out.write("</DataItem>\n</Attribute>\n")

                # write end tag of the current grid
                f_out.write('</Grid>\n')

        # write rest of file
        with open(join(self._save_dir, f"{self._save_name}.xdmf"), "a") as f_out:
            f_out.write('</Grid>\n</Domain>\n</Xdmf>')

        # close hdf file
        _writer.close()

    def fit_data(self, _coord, _data, _field_name, _n_snapshots_total: int = None):
        """
        TODO: documentation

        Note: the variables centers & vertices are denoting the values of the field at center and nodes of each cell
        (not the coordinates of the generated mesh)

        :param _coord:
        :param _data:
        :param _field_name:
        :param _n_snapshots_total:
        :return: None
        """
        # determine the required size of the data matrix
        _n_snapshots_total = _n_snapshots_total if _n_snapshots_total is not None else _data.size()[-1]

        # currently loaded number of snapshots
        _nc = _data.size()[-1]

        # add the data to the current field or create a new field if not yet existing
        if _field_name not in self._interpolated_fields:
            # reset the snapshot counter, because apparently we create a new field
            self._snapshot_counter = 0

            # instantiate field object
            self._interpolated_fields[_field_name] = Fields()

            # create empty tensors for the field values at centers & vertices with dimensions:
            # [N_cells, N_dimensions, N_snapshots_total]
            self._interpolated_fields[_field_name].centers = pt.zeros((self._centers.size()[0], _data.size()[1],
                                                                       _n_snapshots_total))
            self._interpolated_fields[_field_name].vertices = pt.zeros((self._vertices.size()[0], _data.size()[1],
                                                                        _n_snapshots_total))

        # create empty tensors for the values of the field at the centers & vertices with dimensions:
        # [N_cells, N_dimensions, N_snapshots_currently]
        centers = pt.zeros((self._centers.size()[0], _data.size()[1], _nc))
        vertices = pt.zeros((self._vertices.size()[0], _data.size()[1], _nc))

        # fit the KNN and interpolate the data, we need to predict each dimension separately (otherwise dim. mismatch)
        for dimension in range(_data.size()[1]):
            self._knn.fit(_coord, _data[:, dimension, :])
            centers[:, dimension, :_nc] = pt.from_numpy(self._knn.predict(self._centers))
            vertices[:, dimension, :_nc] = pt.from_numpy(self._knn.predict(self._vertices))

        # update the fields, for which snapshots we already executed the interpolation
        self._interpolated_fields[_field_name].centers[:, :, self._snapshot_counter:self._snapshot_counter + _nc] = centers
        self._interpolated_fields[_field_name].vertices[:, :, self._snapshot_counter:self._snapshot_counter + _nc] = vertices
        self._snapshot_counter += _nc

    def write_data_to_file(self):
        print("Writing the data ...")
        self._write_data()
        print("Finished export.")


@njit(fastmath=True, parallel=True)
def resort_grid(_faces, _vertices, _n_dimensions):
    """
    resort the grid points, replace coordinates which make up a cell face with the idx of the cells. Goal is to remove
    duplicate nodes, since each cell has 4 (8) nodes in 2D (3D), but neighboring cells share the same node. However,
    in the _faces array, each cell has 4 (8) nodes, ignoring the sharing of nodes.

    1.
        - _faces contains many duplicate nodes, which need to be removed
    2.
        - we need to replace the coordinates of the nodes with an index, since we need to describe the cell by idx for
          export to HDF5 & XDMF. For example:

            o---o           1---2
            |   |       -> |  A |   meaning that node no. 1, 2, 3, 4 make up cell face A
            o---o          4---3

        for the neighbouring cells, we already have e.g. idx 2, 3 stored, so we need to assign these nodes to the right
        neighbor by replacing the upper & lower node of this neighbor cell with these indices

    :param _faces: array containing coord. of all nodes, size _faces = [N_cells, N_nodes_per_cell, N_dimensions]
    :param _vertices: array containing coord. of all unique nodes in the grid, size _vertices = [N_nodes_total, N_dimensions]
    :param _n_dimensions: number of physical dimensions (3D / 3D)
    :return: idx of resorted _faces array, as: [N_cells, N_nodes_per_cell]
    """
    # map the faces to the corresponding idx of the vertices, therefore loop over all faces
    _indices = np.zeros((_faces.shape[0], _faces.shape[1]), dtype=np.int64)

    # check to which positions the vertices of the current face correspond within the vertices list
    for f in prange(_faces.shape[0]):
        # if parallel = True, we need to assign empty array inside this loop, otherwise the assignment is incorrect
        n_nodes_per_cell = np.zeros(_faces.shape[1], dtype=np.int64)
        counter = 0

        for idx in prange(_vertices.shape[0]):
            # check if the current node matches any of the nodes of the current cell (np.isin() not supported by numba)
            if (np.sum(_faces[f, :, :] == _vertices[idx, :], axis=1) == _n_dimensions).any():
                # if that is the case, add the current node to the list of nodes, which make up the cell
                n_nodes_per_cell[counter] = idx
                counter += 1

        _indices[f, :] = n_nodes_per_cell

    # replace the vertices with the computed cell faces, swap the last columns to avoid issues in paraview, note:
    # idx slicing not supported in numba, e.g. _indices[:, (0, 1, 3, 2)] or _indices[:, (0, 1, 3, 2, 4, 5, 7, 6)]
    if _n_dimensions == 2:
        return np.column_stack((_indices[:, :2], _indices[:, 3], _indices[:, 2]))
    else:
        return np.column_stack((_indices[:, :2], _indices[:, 3], _indices[:, 2], _indices[:, 4:6], _indices[:, 7],
                                _indices[:, 6]))


if __name__ == "__main__":
    pass

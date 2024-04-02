"""
    load the CFD data for the specified (or all available) fields and interpolate them onto the coarse grid, sampled
    with the s^3 algorithm. Export the interpolated data to HDF5 and XDMF in order to be able to load them into paraview
"""
import h5py
import torch as pt

from tqdm import tqdm       # progressbar
from os.path import join
from sklearn.neighbors import KNeighborsRegressor
from flowtorch.data import mask_box, FOAMDataloader


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
        self._times = None

        # compute and resort the cell faces for loading in paraview
        print("Computing cell faces and nodes of final grid...")
        self._resort_grid()
        print("Done.")

        # empty dict which stores the interpolated fields, created when calling the '_load_and_fit_data' method, if no
        # fields are specified all available fields will be interpolated onto the coarser mesh
        self._load_dir = load_dir
        self._field_names = field_names
        # boundaries need to be the same as used for creating the coarse grid, otherwise mismatch(?)
        self._boundaries = domain_boundaries
        self._n_neighbors = 8 if self._n_dimensions == 2 else 26
        self._knn = KNeighborsRegressor(n_neighbors=self._n_neighbors, weights="distance")
        self._interpolated_fields = {}

    def _resort_grid(self):
        # map the faces to the corresponding idx of the vertices, therefore loop over all faces
        _indices = pt.zeros((self._faces.size()[0], self._faces.size()[1]))
        for f in tqdm(range(self._faces.size()[0])):
            # check to which positions the vertices of the current face correspond within the vertices list
            tmp, counter = pt.zeros(self._faces.size()[1]), 0

            for idx in range(self._vertices.size()[0]):
                # check if the current node matches any of the nodes of the current cell
                if (pt.isin(self._faces[f, :, :], self._vertices[idx, :]).int().sum(dim=1) == self._n_dimensions).any():
                    # if that is the case, add the current node to the list of nodes, which make up the cell
                    tmp[counter] = idx
                    counter += 1
            _indices[f, :] = tmp

        # replace the vertices with the computed cell faces
        if self._n_dimensions == 2:
            self._faces = pt.index_select(_indices.int(), 1, pt.tensor([0, 1, 3, 2]))
        else:
            self._faces = pt.index_select(_indices.int(), 1, pt.tensor([0, 1, 3, 2, 4, 5, 7, 6]))

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
            for i, t in enumerate(self._times):
                # in case we have a scalar
                if len(self._interpolated_fields[key][0].size()) == 2:
                    _center.create_dataset(str(t), data=self._interpolated_fields[key][0][:, i])
                    _vertices.create_dataset(str(t), data=self._interpolated_fields[key][1][:, i])
                # in case we have a vector
                else:
                    _center.create_dataset(str(t), data=self._interpolated_fields[key][0][:, :, i])
                    _vertices.create_dataset(str(t), data=self._interpolated_fields[key][1][:, :, i])

        # write global header of XDMF file
        with open(join(self._save_dir, f"{self._save_name}.xdmf"), "w") as f_out:
            f_out.write(_global_header)

        # loop over all available time steps and write all specified data to XDMF & HDF5 files
        for i, t in enumerate(self._times):
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
                    if len(self._interpolated_fields[key][0].size()) == 2:
                        _second_dim = 1
                    else:
                        _second_dim = self._interpolated_fields[key][0].size()[1]

                    # write header
                    f_out.write(f'<Attribute Name="{key}" Center="Cell">\n<DataItem Format="HDF" Dimensions='
                                f'"{self._interpolated_fields[key][0].size()[0]} {_second_dim}">\n')

                    # write interpolated field at cell center
                    f_out.write(f"{self._save_name}.h5:/{key}_center/{t}\n")
                    f_out.write("</DataItem>\n</Attribute>\n")

                    # then do the same for field at the vertices
                    f_out.write(f'<Attribute Name="{key}" Center="Node">\n<DataItem Format="HDF" Dimensions='
                                f'"{self._interpolated_fields[key][1].size()[0]} {_second_dim}">\n')
                    f_out.write(f"{self._save_name}.h5:/{key}_vertices/{t}\n")
                    f_out.write("</DataItem>\n</Attribute>\n")

                # write end tag of the current grid
                f_out.write('</Grid>\n')

        # write rest of file
        with open(join(self._save_dir, f"{self._save_name}.xdmf"), "a") as f_out:
            f_out.write('</Grid>\n</Domain>\n</Xdmf>')

        # close hdf file
        _writer.close()

    def _load_and_fit_data(self):
        # create foam loader object
        loader = FOAMDataloader(self._load_dir)

        # load vertices
        vertices = loader.vertices if self._n_dimensions == 3 else loader.vertices[:, :2]
        mask = mask_box(vertices, lower=self._boundaries[0], upper=self._boundaries[1])

        # the coordinates are independent of the field
        # stack the coordinates to tuples
        if self._n_dimensions == 2:
            coord = pt.stack([pt.masked_select(vertices[:, 0], mask), pt.masked_select(vertices[:, 1], mask)], dim=1)
        else:
            coord = pt.stack([pt.masked_select(vertices[:, 0], mask), pt.masked_select(vertices[:, 1], mask),
                             pt.masked_select(vertices[:, 2], mask)], dim=1)

        # get all available time steps, skip the zero folder
        _write_time = [t for t in loader.write_times[1:]]
        self._times = list(map(float, _write_time))

        # in case there are no fields specified, take all available fields
        self._field_names = loader.field_names[_write_time[0]] if self._field_names is None else self._field_names

        # assemble data matrix for each field and interpolate the values onto the coarser grid
        for field in self._field_names:
            # determine if we have a vector or a scalar
            _field_size = loader.load_snapshot(field, _write_time[0]).size()
            if len(_field_size) == 1:
                data = pt.zeros((mask.sum().item(), len(_write_time)), dtype=pt.float32)
            else:
                data = pt.zeros((mask.sum().item(), _field_size[1], len(_write_time)), dtype=pt.float32)
                mask_vec = mask.unsqueeze(-1).expand(list(_field_size))
                out = list(data.size()[:-1])

            for i, t in enumerate(_write_time):
                # load the field
                if len(_field_size) == 1:
                    data[:, i] = pt.masked_select(loader.load_snapshot(field, t), mask)
                else:
                    data[:, :, i] = pt.masked_select(loader.load_snapshot(field, t), mask_vec).reshape(out)

            # fit the KNN
            if len(_field_size) == 1:
                self._knn.fit(coord, data)

                # interpolate the values onto the coarser grid at the cell centers and at the vertices
                self._interpolated_fields[field] = [pt.from_numpy(self._knn.predict(self._centers)),
                                                    pt.from_numpy(self._knn.predict(self._vertices))]
            else:
                # in case we have a vector, then we need to predict each dimension separately
                _centers_tmp = pt.zeros(size=(self._centers.size()[0], _field_size[1], data.size()[-1]))
                _vertices_tmp = pt.zeros(size=(self._vertices.size()[0], _field_size[1], data.size()[-1]))
                for i in range(_field_size[1]):
                    self._knn.fit(coord, data[:, i, :])
                    _centers_tmp[:, i, :] = pt.from_numpy(self._knn.predict(self._centers))
                    _vertices_tmp[:, i, :] = pt.from_numpy(self._knn.predict(self._vertices))
                self._interpolated_fields[field] = [_centers_tmp, _vertices_tmp]

    def export(self):
        print("Exporting the data...")
        self._load_and_fit_data()
        self._write_data()
        print("Finished export.")


if __name__ == "__main__":
    pass

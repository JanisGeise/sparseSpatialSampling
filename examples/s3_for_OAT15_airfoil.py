"""
    Test the S^3 algorithm on larger CFD dataset, namely the OAT 15 airfoil (2D). The test dataset was already reduced
    (only ever 10th numerical dt) and pre-processed, some key info:

        - each field contains approx. 3.4 GB of data
        - all fields combined contain approx. 17 GB of data
        - the grid size of masked area is 152 257 cells (small) & 245 568 cells (large)
        - 559 time steps (stripped down version)
        - although the simulation is 2D, everything is stored and treated as 3D
        - the geometry is stored as a stl file and will be approximated in a first step with circles
"""
import numpy as np
import torch as pt

from stl import mesh
from os.path import join

from sparseSpatialSampling.export import ExportData
from examples.s3_for_cylinder2D_Re100 import write_svd_s_cube_to_file
from sparseSpatialSampling.sparse_spatial_sampling import SparseSpatialSampling
from sparseSpatialSampling.geometry import CubeGeometry, GeometryCoordinates2D


def load_airfoil_from_stl_file(_load_path: str, _name: str = "oat15.stl", sf: float = 1.0, dimensions: str = "xy",
                               x_offset: float = 0.0, y_offset: float = 0.0, z_offset: float = 0.0):
    """
    Example function for loading airfoil geometries stored as STL file and extract an enclosed 2D-area from it.
    Important Note:

        the structure of the coordinates within the stl files depends on the order the blocks are exported from
        Paraview; the goal is to form an enclosed area, through which we can draw a polygon. Therefore, the way of
        loading and sorting the data depends on the stl file. For an airfoil, the data can be sorted as:

            TE -> via suction side -> LE -> via pressure side -> TE

        It is helpful to export the airfoil geometry without a training edge and close it manually by connecting the
        last points from pressure to suction side

    :param _load_path: path to the STL file
    :param _name: name of the STL file
    :param sf: scaling factor, in case the airfoil needs to be scaled
    :param dimensions: which plane (orientation) to extract from the STL file
    :param x_offset: offset for x-direction, in case the airfoil should be shifted in x-direction
    :param y_offset: offset for y-direction, in case the airfoil should be shifted in y-direction
    :param z_offset: offset for z-direction, in case the airfoil should be shifted in z-direction
    :return: coordinates representing a 2D-airfoil as enclosed area
    """
    # mapping for the coordinate directions
    dim_mapping = {"x": 0, "y": 1, "z": 2}
    dimensions = [dim_mapping[d] for d in dimensions.lower()]

    # load stl file
    stl_file = mesh.Mesh.from_file(_load_path)

    # scale the airfoil to the original size used in CFD and shift if specified
    stl_file.x = stl_file.x * sf + x_offset
    stl_file.y = stl_file.y * sf + y_offset
    stl_file.z = stl_file.z * sf + z_offset

    # stack the coordinates (zeros column, because values are the same in all columns)
    coord_af = np.stack([stl_file.x[:, 0], stl_file.y[:, 0], stl_file.z[:, 0]], -1)

    # remove duplicates without altering the order -> required, otherwise the number of points is very large
    coord_af = coord_af[:, dimensions]
    _, idx = np.unique(coord_af, axis=0, return_index=True)
    coord_af = coord_af[np.sort(idx)]

    return coord_af


if __name__ == "__main__":
    # path to the CFD data and settings
    load_path = join("..", "data", "2D", "OAT15")
    field_name = "Ma"
    area = "large"
    save_path_results = join("..", "run", "final_benchmarks", f"OAT15_{area}",
                             "results_with_geometry_refinement_with_dl_constraint")

    # execute S^3 for range of variances (for parameter study)
    min_variance = pt.arange(0.25, 1.05, 0.05)

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join(load_path, "vertices_and_masks.pt"))

    # load the pressure field of the original CFD data
    if area == "small":
        field = pt.load(join(load_path, f"p_{area}_every10.pt"))
    else:
        field = pt.load(join("..", "data", "2D", "OAT15", f"ma_{area}_every10.pt"))

    # compute the metric
    metric = pt.std(field, dim=1)

    # load the airfoil geometry of the leading airfoil from an STL file
    oat15 = load_airfoil_from_stl_file(join(load_path, "oat15_airfoil_no_TE.stl"), dimensions="xz")

    # define the boundaries for the domain and assemble the geometry objects
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)
    bounds = [[pt.min(xz[:, 0]).item(), pt.min(xz[:, 1]).item()], [pt.max(xz[:, 0]).item(), pt.max(xz[:, 1]).item()]]

    # create a geometry object for the domain and the OAT airfoil (loaded from coordinates)
    geometry = [CubeGeometry("domain", True, bounds[0], bounds[1]),
                GeometryCoordinates2D("OAT15", False, oat15, refine=True)]

    # if we use the large domain, load the rear airfoil of the tandem configuration as well (NACA airfoil)
    if area == "large":
        naca = load_airfoil_from_stl_file(join(load_path, "naca_airfoil_no_TE.stl"), dimensions="xz")
        geometry.append(GeometryCoordinates2D("NACA", False, naca, refine=True))

    # load the corresponding write times and stack the coordinates, the field is available for all time steps sow can
    # pass it to execute_grid_generation directly. If the field to export is only available at certain time steps, we
    # would have to set them after calling execute_grid_generation, e.g., as export.times = times
    times = pt.load(join(load_path, "oat15_tandem_times.pt"))[::10]

    # execute the S^3 algorithm and export the pressure field for the generated grid
    for v in min_variance:
        save_name = "OAT15_" + str(area) + "_area_variance_{:.2f}".format(v)

        # instantiate an S^3 object
        s_cube = SparseSpatialSampling(xz, metric, geometry, save_path_results, save_name, "OAT15", min_metric=v,
                                       write_times=times.tolist(), n_jobs=6)

        # execute S^3
        s_cube.execute_grid_generation()

        # create export instance, export all fields into the same HFD5 file and create single XDMF from it
        export = ExportData(s_cube, write_new_file_for_each_field=False)

        # we need to add one dimension if we have a scalar field
        if len(field.size()) == 2:
            export.export(xz, field.unsqueeze(1), field_name, _n_snapshots_total=None)
        else:
            export.export(xz, field, field_name, _n_snapshots_total=None)

        # compute SVD on grid generated by S^3 and export the results to HDF5 & XDMF
        write_svd_s_cube_to_file(field_name, save_path_results, save_name, export.new_file, 50, rank=int(1e5))

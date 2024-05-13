"""
    test the S^3 algorithm on larger CFD dataset, namely the OAT 15 airfoil (2D). The test dataset was already reduced
    (only ever 10th numerical dt) and pre-processed, some key infos:

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

from s_cube.execute_grid_generation import execute_grid_generation


def load_airfoil_as_stl_file(_load_path: str, _name: str = "oat15.stl", sf: float = 1.0, dimensions: str = "xy",
                             x_offset: float = 0.0, y_offset: float = 0.0, z_offset: float = 0.0):
    """
    example function for loading airfoil geometries stored as STL file and extract an enclosed 2D-area from it.
    Important Note:

        the structure of the coordinates within the stl files depend on the order the blocks are exported from paraview
        the goal is to form an enclosed area, through wich we can draw a polygon. Therefore, the way of loading and
        sorting the data depends on the stl file. For an airfoil, the data can be sorted as:

            TE -> via suction side -> LE -> via pressure side -> TE

        it is helpful to export the airfoil geometry without a training edge and close it manually by connecting the
        last points from pressure to suction side

    :param _load_path: path to the STL file
    :param _name: name of the STL file
    :param sf: scaling factor, in case the airfoil need to be scaled
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

    # close TE for OAT15 airfoil, because the blocks are sorted as: suction side -> TE -> pressure side,
    # but we need an enclosed area
    coord_af = np.append(coord_af, np.expand_dims(coord_af[0, :], axis=0), axis=0)

    return coord_af


def prepare_fields(_load_path: str, _field_name: str) -> pt.Tensor:
    # load all fields saved on external hard drive & only extract the small, masked fields.
    # Return the field for interpolation & export
    pass


if __name__ == "__main__":
    # path to the CFD data and path to directory the results should be saved to
    load_path = join("..", "data", "2D", "OAT15")
    area = "large"
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                             f"results_metric_based_on_ma_stl_{area}")

    # execute S^3 for range of variances (for parameter study)
    min_variance = pt.arange(0.25, 1.05, 0.05)

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join(load_path, "vertices_and_masks.pt"))

    # load the pressure field of the original CFD data, small area around the leading airfoil
    # p = pt.load(join(load_path, "p_small_every10.pt"))
    # metric = pt.std(p, dim=1)

    # test on the large field with 245 568 cells
    load_path_ma_large = join("/media", "janis", "Elements", "FOR_data", "oat15_aoa5_tandem_Johannes")
    ma = pt.load(join(load_path_ma_large, f"ma_{area}_every10.pt"))
    metric = pt.std(ma, dim=1)

    # load the airfoil geometry of the leading airfoil from STL file
    oat15 = load_airfoil_as_stl_file(join(load_path, "oat15_airfoil_no_TE.stl"), dimensions="xz")

    # define the boundaries for the domain and assemble the geometry objects
    bounds = [[pt.min(xz[:, 0]).item(), pt.min(xz[:, 1]).item()], [pt.max(xz[:, 0]).item(), pt.max(xz[:, 1]).item()]]
    geometry = [{"name": "domain", "bounds": bounds, "type": "cube", "is_geometry": False},
                {"name": "OAT15", "bounds": None, "type": "stl", "is_geometry": True, "coordinates": oat15}]

    # if we use the large domain, load the rear airfoil of the tandem configuration as well
    if area == "large":
        naca = load_airfoil_as_stl_file(join(load_path, "naca_airfoil_no_TE.stl"), dimensions="xz")
        geometry.append({"name": "NACA", "bounds": None, "type": "stl", "is_geometry": True, "coordinates": naca})

    # load the corresponding write times and stack the coordinates
    times = pt.load(join(load_path, "oat15_tandem_times.pt"))[::10]
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)

    # execute the S^3 algorithm and export the pressure field for the generated grid
    for v in min_variance:
        export = execute_grid_generation(xz, metric, geometry, save_path_results, "OAT15_" + str(area) +
                                         "_area_variance_{:.2f}".format(v), "OAT15", _min_metric=v,
                                         _write_times=times)
        pt.save(export.mesh_info, join(save_path_results, "mesh_info_OAT15_" + str(area) +
                                       "_area_variance_{:.2f}.pt".format(v)))

        # we need to add one dimension, if we have a scalar field
        # export.fit_data(xz, p.unsqueeze(1), "p", _n_snapshots_total=None)
        export.fit_data(xz, ma.unsqueeze(1), "Ma", _n_snapshots_total=None)
        export.write_data_to_file()

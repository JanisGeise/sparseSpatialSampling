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
from shapely.geometry import Polygon

from s_cube.execute_grid_generation import execute_grid_generation


def load_OAT15_as_stl_file(_load_path: str, _name: str = "oat15.stl", sf: float = 0.15, dimensions: str = "xy",
                           x_offset: float = 0.0, y_offset: float = 0.0, z_offset: float = 0.0):
    # TODO: - generalize for arbitrary orientations and refactor to geometry.py, deal with the case that the 3rd
    #         dimension is not set to z, but to some other dimension
    #       - documentation of function
    # mapping for the coordinate directions
    dim_mapping = {"x": 0, "y": 1, "z": 2}
    dimensions = [dim_mapping[d] for d in dimensions.lower()]

    # load stl file
    stl_file = mesh.Mesh.from_file(_load_path)

    # scale the airfoil to the original size used in CFD and shift if specified
    stl_file.x = stl_file.x * sf + x_offset
    stl_file.y = stl_file.y * sf + y_offset
    stl_file.z = stl_file.z * sf + z_offset

    # stack the coordinates (zeros column, because values are the same in all columns), z-coordinate alternates
    coord_af = np.stack([stl_file.x[:, 0], stl_file.y[:, 0], stl_file.z[:, 1]], -1)

    # only take the required direction, e.g. for 2D x-y, if 2D then we only need every 2nd point, because each other
    # point corresponds to the theoretical negative z-direction
    coord_af = coord_af[::2, dimensions] if len(dimensions) == 2 else coord_af[:, dimensions]

    return Polygon(coord_af)


def prepare_fields(_load_path: str, _field_name: str) -> pt.Tensor:
    # load all fields saved on external hard drive & only extract the small, masked fields.
    # Return the field for interpolation & export
    pass


if __name__ == "__main__":
    # path to the CFD data and path to directory the results should be saved to
    # TODO: test for 3D grid & 3D stl file, once this works update documentation header 'geometry.py'
    load_path = join("..", "data", "2D", "OAT15")
    area = "large"
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                             f"results_metric_based_on_ma_stl_{area}")
    # execute S^3 for range of variances (for parameter study)
    min_variance = pt.arange(0.25, 1.05, 0.05)

    # load the pressure field of the original CFD data, small area around the leading airfoil
    # p = pt.load(join(load_path, "p_small_every10.pt"))
    xz = pt.load(join(load_path, "vertices_and_masks.pt"))
    # xz = pt.stack([xz["x_small"], xz["z_small"]], dim=-1)

    # test on the large field with 245 568 cells
    load_path_ma_large = join("/media", "janis", "Elements", "FOR_data", "oat15_aoa5_tandem_Johannes")
    xz = pt.stack([xz["x_large"], xz["z_large"]], dim=-1)
    ma = pt.load(join(load_path_ma_large, f"ma_{area}_every10.pt"))

    # load the airfoil geometry of the leading airfoil from STL file
    airfoil_geometry = load_OAT15_as_stl_file(join(load_path, "oat15.stl"), dimensions="xy")

    # define the boundaries for the domain and assemble the geometry objects
    bounds = [[pt.min(xz[:, 0]).item(), pt.min(xz[:, 1]).item()], [pt.max(xz[:, 0]).item(), pt.max(xz[:, 1]).item()]]
    domain = {"name": "domain", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "OAT15", "bounds": None, "type": "stl", "is_geometry": True, "coordinates": airfoil_geometry}

    # load the corresponding write times
    times = pt.load(join(load_path, "oat15_tandem_times.pt"))[::10]

    # execute the S^3 algorithm and export the pressure field for the generated grid
    # metric = pt.std(p, dim=1)
    metric = pt.std(ma, dim=1)

    for v in min_variance:
        export = execute_grid_generation(xz, metric, [domain, geometry], save_path_results,
                                         "OAT15_" + str(area) + "_area_variance_{:.2f}".format(v), "OAT15",
                                         _min_variance=v)
        pt.save(export.mesh_info, join(save_path_results, "mesh_info_OAT15_" + str(area) +
                                       "_area_variance_{:.2f}.pt".format(v)))
        export.times = times

        # we need to add one dimension, if we have a scalar field
        if len(ma.size()) == 2:
            # export.fit_data(xz, p.unsqueeze(1), "p", _n_snapshots_total=None)
            export.fit_data(xz, ma.unsqueeze(1), "Ma", _n_snapshots_total=None)
        else:
            export.fit_data(xz, ma, "Ma", _n_snapshots_total=None)
        export.write_data_to_file()

"""
    test the S^3 algorithm on larger CFD dataset, namely the OAT 15 airfoil (2D). The test dataset was already reduced
    (only ever 10th numerical dt) and pre-processed, some key infos:

        - each field contains approx. 3.4 GB of data
        - all fields combined contain approx. 17 GB of data
        - the grid size of masked area is 152 257 cells
        - 559 time steps (stripped down version)
        - although the simulation is 2D, everything is stored and treated as 3D
        - the geometry is stored as a stl file and will be approximated in a first step with circles
"""
import numpy as np
import torch as pt

from os.path import join

from shapely.geometry import Polygon
from stl import mesh

from post_processing.load_airfoil_data import compute_geometry_objects_OAT15
from s_cube.execute_grid_generation import execute_grid_generation


def load_OAT15_as_stl_file(_load_path: str, _name: str = "oat15.stl", _sf: float = 0.15, dimensions: str = "xy"):
    # TODO: generalize for arbitrary orientations and refactor to geometry.py
    # mapping for the coordinate directions
    dim_mapping = {"x": 0, "y": 1, "z": 2}
    dimensions = [dim_mapping[d] for d in dimensions.lower()]

    # load stl file
    stl_file = mesh.Mesh.from_file(_load_path)

    # scale the airfoil to the original size used in CFD
    stl_file.x *= _sf
    stl_file.y *= _sf
    stl_file.z *= _sf

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
    # TODO: test for 3D grid & 3D stl file
    load_path_pressure = join("..", "data", "2D", "OAT15")
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15", "results_stl_test")

    # execute S^3 for range of variances (for parameter study)
    # min_variance = pt.arange(0.25, 1.05, 0.05)
    min_variance = [0.25]

    # load the pressure field of the original CFD data, small area around the leading airfoil
    p = pt.load(join(load_path_pressure, "p_small_every10.pt"))
    xz = pt.load(join(load_path_pressure, "vertices_and_masks.pt"))
    xz = pt.stack([xz["x_small"], xz["z_small"]], dim=-1)

    # load the airfoil geometry frm STL file
    airfoil_geometry = load_OAT15_as_stl_file(join(load_path_pressure, "oat15.stl"), dimensions="xy")

    # define the boundaries for the domain and assemble the geometry objects
    bounds = [[pt.min(xz[:, 0]).item(), pt.min(xz[:, 1]).item()], [pt.max(xz[:, 0]).item(), pt.max(xz[:, 1]).item()]]
    domain = {"name": "domain", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "OAT15", "bounds": None, "type": "stl", "is_geometry": True, "coordinates": airfoil_geometry}

    # TODO: - export all available fields & compare to original data -> time steps stored somewhere?
    # for now just name the write times t_i = 1, ... , N
    times = list(range(p.size(1)))

    # execute the S^3 algorithm and export the pressure field for the generated grid
    for v in min_variance:
        export = execute_grid_generation(xz, pt.std(p, dim=1), [domain, geometry], save_path_results,
                                         "OAT15_small_area_variance_{:.2f}".format(v), "OAT15", _min_variance=v)
        pt.save(export.mesh_info, join(save_path_results, "mesh_info_OAT15_small_area_variance__{:.2f}.pt".format(v)))
        export.times = times

        # we need to add one dimension, because the pressure is a scalar field
        export.fit_data(xz, p.unsqueeze(1), "p", _n_snapshots_total=None)
        export.write_data_to_file()

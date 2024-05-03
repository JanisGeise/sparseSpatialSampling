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
import torch as pt

from os.path import join

from post_processing.load_airfoil_data import compute_geometry_objects_OAT15
from s_cube.execute_grid_generation import execute_grid_generation


def prepare_fields(_load_path: str, _field_name: str) -> pt.Tensor:
    # load all fields saved on external hard drive & only extract the small, masked fields.
    # Return the field for interpolation & export
    pass


if __name__ == "__main__":
    min_variance = 0.90
    load_path_pressure = join("..", "data", "2D", "OAT15")
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15", "results")

    # load the pressure field of the original CFD data, small area around the leading airfoil
    p = pt.load(join(load_path_pressure, "p_small_every10.pt"))
    xz = pt.load(join(load_path_pressure, "vertices_and_masks.pt"))
    xz = pt.stack([xz["x_small"], xz["z_small"]], dim=-1)

    # load the approximated airfoil (airfoil is approximated with circles). The geometry data of the airfoil itself is
    # restricted and consequently can't be shown here. Later, (hopefully) a function for directly loading STL-files as
    # geometry objects will be provided, making the geometry approximation obsolete
    airfoil_geometry = compute_geometry_objects_OAT15(load_path_pressure)

    # define the boundaries for the domain and assemble the geometry objects
    bounds = [[pt.min(xz[:, 0]).item(), pt.min(xz[:, 1]).item()], [pt.max(xz[:, 0]).item(), pt.max(xz[:, 1]).item()]]
    domain = [{"name": "domain", "bounds": bounds, "type": "cube", "is_geometry": False}]
    geometry = [{"name": "OAT15", "bounds": b, "type": "sphere", "is_geometry": True} for b in airfoil_geometry]

    # execute the S^3 algorithm
    export = execute_grid_generation(xz, pt.std(p, dim=1), domain + geometry, save_path_results,
                                     "OAT15_small_area_variance_{:.2f}_test".format(min_variance), "OAT15",
                                     _min_variance=min_variance)
    pt.save(export.mesh_info, join(save_path_results,
                                   "mesh_info_OAT15_small_area_variance__{:.2f}.pt".format(min_variance)))

    # TODO: - export all available fields & compare to original data -> time steps stored somewhere?
    # for now just name the write times t_i = 1, ... , N
    export.times = list(range(p.size(1)))

    # we need to add one dimension, because the pressure is a scalar field
    export.fit_data(xz, p.unsqueeze(1), "p", _n_snapshots_total=None)
    export.write_data_to_file()

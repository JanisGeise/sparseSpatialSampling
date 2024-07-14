"""
    execute parameter study for different target metrics of the generated grid wrt the original grid from CFD for
    various test cases
"""
import torch as pt
from os.path import join

from s_cube.geometry import CubeGeometry
from s_cube.sparse_spatial_sampling import SparseSpatialSampling
from s_cube.utils import export_openfoam_fields, load_cfd_data


def execute_parameter_study(coordinates: pt.Tensor, metric: pt.Tensor, geometries: list, boundaries: list,
                            save_path: str, load_path: str, grid_name: str,
                            variances_to_run: pt.Tensor = pt.arange(0.25, 1.05, 0.05), fields: list = None) -> None:
    """
    wrapper function for executing the parameter study captured variance of original grid vs. number of cells final
    grid and required execution times of S^3

    :param coordinates: coordinates of the original grid from CFD
    :param metric: the metric based on which the grid is generated
    :param geometries: list containing the domain and geometries
    :param boundaries: boundaries of the domain
    :param save_path: path in which directory the results should be saved to
    :param load_path: path to the original CFD data
    :param grid_name: name of the grid, only used internally in XDMF and HDF5 files
    :param variances_to_run: for which target variances the S^3 should be executed
    :param fields: fields to export, either str or list[str]. If 'None' then all available fields at the first time
                   step will be exported
    :return: None
    """
    for v in variances_to_run:
        s_cube = SparseSpatialSampling(coordinates, metric, geometries, save_path,
                                       "interpolated_mesh_variance_{:.2f}".format(v.item()), grid_name,
                                       min_metric=v.item())
        # execute S^3
        export = s_cube.execute_grid_generation()

        # export the fields
        export_openfoam_fields(export, load_path, boundaries, fields=fields)


if __name__ == "__main__":
    # -----------------------------------------   execute for cylinder   -----------------------------------------
    load_path_cylinder = join("..", "data", "2D", "cylinder2D_re1000")
    save_path_cylinder = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D", "results")

    # boundaries of the masked domain for the cylinder
    bounds = [[0, 0], [2.2, 0.41]]  # full domain
    cylinder = [[0.2, 0.2], [0.05]]  # [[x, y], [r]]

    # load the CFD data
    pressure, coord, _, _ = load_cfd_data(load_path_cylinder, bounds)

    # generate the grid, export the data
    domain = {"name": "domain cylinder", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "cylinder", "bounds": cylinder, "type": "sphere", "is_geometry": True}

    # execute the parameter study for the cylinder
    execute_parameter_study(coord, pt.std(pressure, 1), [domain, geometry], bounds, save_path_cylinder,
                            load_path_cylinder, "cylinder2D")

    # -----------------------------------------   execute for cube   -----------------------------------------
    load_path_cube = join("..", "data", "3D", "surfaceMountedCube_original_grid_size", "fullCase")
    save_path_cube = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube",
                          "results")

    # boundaries of the masked domain for the cube, [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    bounds = [[0, 0, 0], [9, 14.5, 2]]

    # load the CFD data
    pressure, coord, _, _ = load_cfd_data(load_path_cube, bounds, n_dims=3)

    # define the geometry object for the domain and cube
    domain = CubeGeometry("domain", True, bounds[0], bounds[1])
    geometry = CubeGeometry("cube", False, [3.5, 4, -1], [4.5, 5, 1])

    # execute the parameter study for the cylinder
    execute_parameter_study(coord, pt.std(pressure, 1), [domain, geometry], bounds, save_path_cube, load_path_cube,
                            "cube", variances_to_run=pt.arange(0.4, 1.05, 0.05), fields=["p", "U"])

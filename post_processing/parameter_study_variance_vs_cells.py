"""
    execute parameter study for different target variances of the generated grid wrt the original grid from CFD
"""
import torch as pt
from os.path import join

from main import load_cylinder_data, load_cube_data
from s_cube.execute_grid_generation import execute_grid_generation, export_data


def execute_parameter_study(coordinates: pt.Tensor, metric: pt.Tensor, geometries: list, boundaries: list,
                            save_path: str, load_path: str, grid_name: str,
                            variances_to_run: pt.Tensor = pt.arange(0.25, 1.05, 0.05)) -> None:
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
    :return: None
    """
    for v in variances_to_run:
        export = execute_grid_generation(coordinates, metric, geometries, save_path,
                                         "interpolated_mesh_variance_{:.2f}".format(v.item()), grid_name,
                                         _min_metric=v.item())
        pt.save(export.mesh_info, join(save_path, "mesh_info_variance_{:.2f}.pt".format(v.item())))
        export_data(export, load_path, boundaries)


if __name__ == "__main__":
    # -----------------------------------------   execute for cylinder   -----------------------------------------
    load_path_cylinder = join("..", "data", "2D", "cylinder2D_re1000")
    save_path_cylinder = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D",
                              "with_geometry_refinement", "results")

    # boundaries of the masked domain for the cylinder
    bounds = [[0, 0], [2.2, 0.41]]  # full domain
    cylinder = [[0.2, 0.2], [0.05]]  # [[x, y], [r]]

    # load the CFD data
    pressure, coord, _ = load_cylinder_data(load_path_cylinder, bounds)

    # generate the grid, export the data
    domain = {"name": "domain cylinder", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "cylinder", "bounds": cylinder, "type": "sphere", "is_geometry": True}

    # execute the parameter study for the cylinder
    execute_parameter_study(coord, pt.std(pressure, 1), [domain, geometry], bounds, save_path_cylinder,
                            load_path_cylinder, "cylinder2D")

    # -----------------------------------------   execute for cube   -----------------------------------------
    load_path_cube = join("..", "data", "3D", "surfaceMountedCube_original_grid_size", "fullCase")
    save_path_cube = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube",
                          "with_geometry_refinement", "results")

    # boundaries of the masked domain for the cube
    bounds = [[0, 0, 0], [9, 14.5, 2]]              # full domain
    cube = [[3.5, 4, -1], [4.5, 5, 1]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    # load the CFD data
    pressure, coord, _ = load_cube_data(load_path_cube, bounds)

    # generate the grid, export the data
    domain = {"name": "domain cube", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "cube", "bounds": cube, "type": "cube", "is_geometry": True}

    # execute the parameter study for the cylinder
    execute_parameter_study(coord, pt.std(pressure, 1), [domain, geometry], bounds, save_path_cube, load_path_cube,
                            "cube")

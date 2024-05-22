"""
    Execute the sparse spatial sampling algorithm on 2D CFD data, export the resulting mesh as XDMF and HDF5 files.
    The test case here is an OpenFoam tutorial with some adjustments to the number of cells and Reynolds number,
    currently:

        - cylinder2D (at Re = 1000), located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/

    IMPORTANT: the size of the data matrix (of the original CFD data) provided for interpolation onto the generated
               coarser grid must be

                - [N_cells, N_dimensions, N_snapshots] (vector field)
                - [N_cells, 1, N_snapshots] (scalar field)

                to correctly execute the 'fit_data()' method of the DataWriter class

    In this example, the cylinder is represented by a center and radius (no STL file).
"""
import torch as pt

from typing import Tuple
from os.path import join
from flowtorch.data import FOAMDataloader, mask_box

from s_cube.execute_grid_generation import execute_grid_generation, export_data


def load_cylinder_data(load_dir: str, boundaries: list) -> Tuple[pt.Tensor, pt.Tensor, list]:
    """
    load the pressure field of the cylinder2D case, mask out an area

    :param load_dir: path to the simulation data
    :param boundaries: list with list containing the upper and lower boundaries of the mask
    :return: pressure fields at each write time, x- & y-coordinates of the cells as tuples and all write times
    """
    # create foam loader object
    loader = FOAMDataloader(load_dir)

    # load vertices and discard z-coordinate
    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, lower=boundaries[0], upper=boundaries[1])

    # assemble data matrix
    write_time = [t for t in loader.write_times[1:] if float(t) >= 0.4]
    data = pt.zeros((mask.sum().item(), len(write_time)), dtype=pt.float32)
    for i, t in enumerate(write_time):
        # load the pressure field
        data[:, i] = pt.masked_select(loader.load_snapshot("p", t), mask)

    # stack the coordinates to tuples
    xy = pt.stack([pt.masked_select(vertices[:, 0], mask), pt.masked_select(vertices[:, 1], mask)], dim=1)

    return data, xy, list(map(float, write_time))


if __name__ == "__main__":
    # -----------------------------------------   execute for cylinder   -----------------------------------------
    # load paths to the CFD data
    load_path_cylinder = join("..", "data", "2D", "cylinder2D_re1000")
    save_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D", "test")

    # how much of the metric within the original grid should be captured at least
    min_metric = 0.75
    save_name = f"metric_{round(min_metric, 2)}_cylinder_full_domain"

    # boundaries of the masked domain for the cylinder
    bounds = [[0, 0], [2.2, 0.41]]          # [[xmin, ymin], [xmax, ymax]]
    cylinder = [[0.2, 0.2], [0.05]]         # [[x, y], [r]]

    # load the CFD data
    pressure, coord, write_times = load_cylinder_data(load_path_cylinder, bounds)

    # create a setup for geometry objects for the domain and the cylinder, we don't use any STL files
    domain = {"name": "domain cylinder", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "cylinder", "bounds": cylinder, "type": "sphere", "is_geometry": True}

    # generate the grid
    export = execute_grid_generation(coord, pt.std(pressure, 1), [domain, geometry], save_path, save_name,
                                     "cylinder2D", _min_metric=min_metric)

    # save information about the refinement and grid
    pt.save(export.mesh_info, join(save_path, "mesh_info_cube_variance_{:.2f}.pt".format(min_metric)))

    # export the data
    export_data(export, load_path_cylinder, bounds)

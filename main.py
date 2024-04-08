"""
    execute the sparse spatial sampling algorithm on 2D & 3D CFD data for testing purposes, export the resulting mesh
    as XDMF and HDF5 files. The test cases are OpenFoam tutorials with some adjustments wrt number of cells and Reynolds
    number, currently:

        - cylinder2D (at Re = 1000), located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/
        - surfaceMountedCube (coarser grid), located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/LES/
"""
import torch as pt

from typing import Tuple
from os.path import join
from flowtorch.data import FOAMDataloader, mask_box

from s_cube.execute_grid_generation import execute_grid_generation


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


def load_cube_data(load_dir: str, boundaries: list) -> Tuple[pt.Tensor, pt.Tensor, list]:
    """
    load the pressure field of the surfaceMountedCube case, mask out an area

    :param load_dir: path to the simulation data
    :param boundaries: list with list containing the upper and lower boundaries of the mask
    :return: pressure fields at each write time, x-, y- & z-coordinates of the cells as tuples and all write times
    """
    # create foam loader object
    loader = FOAMDataloader(load_dir)

    # load vertices
    vertices = loader.vertices
    mask = mask_box(vertices, lower=boundaries[0], upper=boundaries[1])

    # assemble data matrix
    write_time = [t for t in loader.write_times[1:] if float(t) >= 0.0]
    data = pt.zeros((mask.sum().item(), len(write_time)), dtype=pt.float32)
    for i, t in enumerate(write_time):
        # load the pressure field
        data[:, i] = pt.masked_select(loader.load_snapshot("p", t), mask)

    # stack the coordinates to tuples
    xyz = pt.stack([pt.masked_select(vertices[:, 0], mask), pt.masked_select(vertices[:, 1], mask),
                    pt.masked_select(vertices[:, 2], mask)], dim=1)

    return data, xyz, list(map(float, write_time))


if __name__ == "__main__":
    # -----------------------------------------   execute for cube   -----------------------------------------
    load_path_cube = join("", "data", "3D", "surfaceMountedCube", "fullCase")
    save_path_cube = join("", "data", "3D", "exported_grids")
    save_name = f"final_mesh_cube_test_early_stopping"

    # boundaries of the masked domain for the cube
    bounds = [[1.4, 3, 0], [9, 6, 1.5]]          # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    cube = [[3.5, 4, -1], [4.5, 5, 1]]       # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    # load the CFD data
    pressure, coord, _ = load_cube_data(load_path_cube, bounds)

    # generate the grid, export the data
    domain = {"name": "domain cube", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "cube", "bounds": cube, "type": "cube", "is_geometry": True}

    execute_grid_generation(coord, pt.std(pressure, 1), [domain, geometry], load_path_cube, save_path_cube, save_name,
                            "cube")

    # -----------------------------------------   execute for cylinder   -----------------------------------------
    # load paths to the CFD data
    load_path_cylinder = join("", "data", "2D", "cylinder2D_re1000")
    save_path_cylinder = join("", "data", "2D", "exported_grids")
    save_name = f"final_mesh_cylinder_test_early_stopping_dynamic_cells"

    # boundaries of the masked domain for the cylinder
    bounds = [[0.1, 0], [1.0, 0.41]]      # [[xmin, ymin], [xmax, ymax]]
    cylinder = [[0.2, 0.2], [0.05]]       # [[x, y], [r]]

    # load the CFD data
    pressure, coord, _ = load_cylinder_data(load_path_cylinder, bounds)

    # generate the grid, export the data
    domain = {"name": "domain cylinder", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "cylinder", "bounds": cylinder, "type": "sphere", "is_geometry": True}

    execute_grid_generation(coord, pt.std(pressure, 1), [domain, geometry], load_path_cylinder, save_path_cylinder,
                            save_name, "cylinder2D")


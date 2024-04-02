"""
    execute the sparse spatial sampling algorithm on 2D CFD data, export the resulting mesh as XDMF and plot the results
"""
from os import path, makedirs

import torch as pt

from typing import Tuple
from os.path import join
from flowtorch.data import FOAMDataloader, mask_box

from s3_implementation import SamplingTree
from export_data import DataWriter


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
    # load path to the CFD data
    load_path_cylinder = join("..", "data", "2D", "cylinder2D_re1000")
    save_path_cylinder = join("..", "data", "2D", "exported_grids")

    load_path_cube = join("..", "data", "3D", "surfaceMountedCube", "fullCase")
    save_path_cube = join("..", "data", "3D", "exported_grids")

    # target number of cells in the coarsened grid
    n_cells_cube = 500
    n_cells_cylinder = 500     # (orig: 14150 cells for chosen bounds)

    # boundaries of the masked domain for the cube
    bounds = [[1.4, 3, 0], [9, 6, 1.5]]          # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    geometry = [[3.5, 4, -1], [4.5, 5, 1]]       # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    # create directory for plots
    if not path.exists(save_path_cylinder) or not path.exists(save_path_cube):
        makedirs(save_path_cylinder)
        makedirs(save_path_cube)

    # load the CFD data
    pressure, coord, times = load_cube_data(load_path_cube, bounds)

    # coarsen the cube mesh based on the std. deviation of the pressure
    sampling = SamplingTree(coord, pt.std(pressure, 1), n_cells=n_cells_cube, level_bounds=(0, 25), cells_per_iter=1,
                            n_neighbors=26, boundaries=bounds, geometry=geometry, write_times=times)
    sampling.refine()

    # compute the cell centers and vertices of each leaf cell
    sampling.compute_nodes_final_mesh()

    # fit the pressure field onto the new mesh and export the data
    export_data = DataWriter(sampling.leaf_cells(), load_dir=load_path_cube, field_names=["p", "U"],
                             save_dir=save_path_cube, domain_boundaries=bounds,
                             save_name=f"final_mesh_{n_cells_cube}_cells_cube", grid_name="cube")
    export_data.export()

    # -----------------------------------------   execute for cylinder   -----------------------------------------
    # boundaries of the masked domain for the cylinder
    bounds = [[0.1, 0], [1.0, 0.41]]      # [[xmin, ymin], [xmax, ymax]]
    geometry = [[0.2, 0.2], 0.05]       # [[x, y], r]

    # load the CFD data
    pressure, coord, times = load_cylinder_data(load_path_cylinder, bounds)

    # coarsen the cylinder2D mesh based on the std. deviation of the pressure
    sampling = SamplingTree(coord, pt.std(pressure, 1), n_cells=n_cells_cylinder, level_bounds=(0, 25),
                            cells_per_iter=1, n_neighbors=8, boundaries=bounds, geometry=geometry)
    sampling.refine()

    # compute the cell centers and vertices of each leaf cell
    sampling.compute_nodes_final_mesh()

    # fit the pressure field onto the new mesh and export the data
    export_data = DataWriter(sampling.leaf_cells(), load_dir=load_path_cylinder, field_names=["p", "U"],
                             save_dir=save_path_cylinder, domain_boundaries=bounds,
                             save_name=f"final_mesh_{n_cells_cylinder}_cells_cylinder", grid_name="cylinder")
    export_data.export()

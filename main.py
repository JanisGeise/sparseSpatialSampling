"""
    execute the sparse spatial sampling algorithm on 2D & 3D CFD data for testing purposes, export the resulting mesh
    as XDMF and HDF5 files. The test cases are OpenFoam tutorials with some adjustments wrt number of cells and Reynolds
    number, currently:

        - cylinder2D (at Re = 1000), located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/
        - surfaceMountedCube (coarser grid), located under: $FOAM_TUTORIALS/incmpressible/pimpleFoam/LES/
"""
import torch as pt

from time import time
from typing import Tuple
from os.path import join
from os import path, makedirs
from flowtorch.data import FOAMDataloader, mask_box

from s_cube.export_data import DataWriter
from s_cube.geometry import GeometryObject
from s_cube.s_cube import SamplingTree


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

    # target number of cells in the coarsened grid (orig. 27974 cells for chosen bounds)
    n_cells_cube = 10000

    # boundaries of the masked domain for the cube
    bounds = [[1.4, 3, 0], [9, 6, 1.5]]          # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    geometry = [[3.5, 4, -1], [4.5, 5, 1]]       # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    # load the CFD data
    pressure, coord, times = load_cube_data(load_path_cube, bounds)

    # coarsen the cube mesh based on the std. deviation of the pressure
    sampling = SamplingTree(coord, pt.std(pressure, 1), n_cells=n_cells_cube, level_bounds=(1, 25), cells_per_iter=10,
                            n_neighbors=26, write_times=times)

    # add the cube and the domain
    sampling.geometry.append(GeometryObject(lower_bound=bounds[0], upper_bound=bounds[1], obj_type="cube",
                                            geometry=False, name="domain"))
    sampling.geometry.append(GeometryObject(lower_bound=geometry[0], upper_bound=geometry[1], obj_type="cube"))

    sampling.refine()

    # compute the cell centers and vertices of each leaf cell
    sampling.compute_nodes_final_mesh()

    # create directory for plots
    if not path.exists(save_path_cube):
        makedirs(save_path_cube)

    # fit the pressure field onto the new mesh and export the data
    t_start = time()
    export_data = DataWriter(sampling.leaf_cells(), load_dir=load_path_cube, field_names=["p", "U"],
                             save_dir=save_path_cube, domain_boundaries=bounds,
                             save_name=f"final_mesh_{n_cells_cube}_cells_cube", grid_name="cube")
    export_data.export()
    print(f"Export required {round((time() - t_start), 3)} s.\n")

    # -----------------------------------------   execute for cylinder   -----------------------------------------
    # load paths to the CFD data
    load_path_cylinder = join("", "data", "2D", "cylinder2D_re1000")
    save_path_cylinder = join("", "data", "2D", "exported_grids")

    # target number of cells in the coarsened grid (orig: 14150 cells for chosen bounds)
    n_cells_cylinder = 10000

    # boundaries of the masked domain for the cylinder
    bounds = [[0.1, 0], [1.0, 0.41]]      # [[xmin, ymin], [xmax, ymax]]
    geometry = [[0.2, 0.2], 0.05]       # [[x, y], r]

    # load the CFD data
    pressure, coord, times = load_cylinder_data(load_path_cylinder, bounds)

    # coarsen the cylinder2D mesh based on the std. deviation of the pressure
    sampling = SamplingTree(coord, pt.std(pressure, 1), n_cells=n_cells_cylinder, level_bounds=(1, 25),
                            cells_per_iter=10, n_neighbors=8)

    # add the cube and the domain
    sampling.geometry.append(GeometryObject(lower_bound=bounds[0], upper_bound=bounds[1], obj_type="cube",
                                            geometry=False, name="domain"))
    sampling.geometry.append(GeometryObject(lower_bound=geometry[0], upper_bound=geometry[1], obj_type="sphere",
                                            name="cylinder"))

    sampling.refine()

    # compute the cell centers and vertices of each leaf cell
    sampling.compute_nodes_final_mesh()

    # create directory for plots
    if not path.exists(save_path_cylinder):
        makedirs(save_path_cylinder)

    # fit the pressure field onto the new mesh and export the data
    t_start = time()
    export_data = DataWriter(sampling.leaf_cells(), load_dir=load_path_cylinder, field_names=["p", "U"],
                             save_dir=save_path_cylinder, domain_boundaries=bounds,
                             save_name=f"final_mesh_{n_cells_cylinder}_cells_cylinder", grid_name="cylinder")

    export_data.export()
    print(f"Export required {round((time() - t_start), 3)} s.")

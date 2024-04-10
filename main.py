"""
    execute the sparse spatial sampling algorithm on 2D & 3D CFD data for testing purposes, export the resulting mesh
    as XDMF and HDF5 files. The test cases are OpenFoam tutorials with some adjustments wrt number of cells and Reynolds
    number, currently:

        - cylinder2D (at Re = 1000), located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/
        - surfaceMountedCube (coarser grid), located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/LES/

    IMPORTANT: size of data matrix (of the original CFD data) provided for interpolation onto the generated coarser grid
               must be:
                - [N_cells, N_dimensions, N_snapshots] (vector field)
                - [N_cells, 1, N_snapshots] (scalar field)

                in order to correctly execute the 'fit_data()' method of the DataWriter class
"""
import torch as pt

from typing import Tuple
from os.path import join
from flowtorch.data import FOAMDataloader, mask_box

from s_cube.execute_grid_generation import execute_grid_generation, load_original_Foam_fields


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
    # path to original surfaceMountedCube simulation (size ~ 8.4 GB, reconstructed)
    load_path_cube = join("", "data", "3D", "surfaceMountedCube_original_grid_size", "fullCase")
    save_name = f"test_cube_new_data_from_original_cube"

    # surfaceMountedCube simulation with coarser grid for testing purposes
    # load_path_cube = join("", "data", "3D", "surfaceMountedCube", "fullCase")
    # save_name = "test_cube_new"
    save_path_cube = join("", "data", "3D", "exported_grids")

    # boundaries of the masked domain for the cube
    bounds = [[1.4, 3, 0], [9, 6, 1.5]]          # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    cube = [[3.5, 4, -1], [4.5, 5, 1]]       # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    # load the CFD data
    pressure, coord, _ = load_cube_data(load_path_cube, bounds)

    # generate the grid, export the data
    domain = {"name": "domain cube", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "cube", "bounds": cube, "type": "cube", "is_geometry": True}

    export = execute_grid_generation(coord, pt.std(pressure, 1), [domain, geometry], load_path_cube, save_path_cube,
                                     save_name, "cube")

    # export the data
    times, fields = load_original_Foam_fields(load_path_cube, 2, bounds, _get_field_names_and_times=True)

    # save time steps of all snapshots, which will be exported to HDF5 & XDMF
    export.times = list(map(float, times))

    # interpolate and export the specified fields
    for f in fields:
        coord, data = load_original_Foam_fields(load_path_cube, 3, bounds, _field_names=f)

        # in case the field is not available the function will return None
        if data is not None:
            export.fit_data(coord, data, f, _n_snapshots_total=len(times))
    export.write_data_to_file()

    """
    # alternatively subset of them or each snapshot can be passed separately (if data size too large)
    for f in fields:
        for t in times:
            coord, data = load_original_Foam_fields(load_path_cube, 3, bounds, _field_names=f, _write_times=t)
            
            # in case the field is not available the function will return None
            if data is not None:
                export.fit_data(coord, data, f, _n_snapshots_total=len(times))
    export.write_data_to_file()
    """
    # -----------------------------------------   execute for cylinder   -----------------------------------------
    # load paths to the CFD data
    load_path_cylinder = join("", "data", "2D", "cylinder2D_re1000")
    save_path_cylinder = join("", "data", "2D", "exported_grids")
    save_name = f"test_cylinder_new"

    # boundaries of the masked domain for the cylinder
    bounds = [[0.1, 0], [1.0, 0.41]]      # [[xmin, ymin], [xmax, ymax]]
    cylinder = [[0.2, 0.2], [0.05]]       # [[x, y], [r]]

    # load the CFD data
    pressure, coord, write_times = load_cylinder_data(load_path_cylinder, bounds)

    # generate the grid, export the data
    domain = {"name": "domain cylinder", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "cylinder", "bounds": cylinder, "type": "sphere", "is_geometry": True}

    export = execute_grid_generation(coord, pt.std(pressure, 1), [domain, geometry], load_path_cylinder,
                                     save_path_cylinder, save_name, "cylinder2D")

    # export the data
    times, fields = load_original_Foam_fields(load_path_cylinder, 2, bounds, _get_field_names_and_times=True)

    # save time step of all snapshots, which will be exported to HDF5 & XDMF
    export.times = list(map(float, times))

    # interpolate and export the specified fields
    for f in fields:
        coord, data = load_original_Foam_fields(load_path_cylinder, 2, bounds, _field_names=f)

        # in case the field is not available the function will return None
        if data is not None:
            export.fit_data(coord, data, f, _n_snapshots_total=len(times))
    export.write_data_to_file()

    """
    # alternatively subset of them or each snapshot can be passed separately (if data size too large)
    for f in fields:
        for t in times:
            coord, data = load_original_Foam_fields(load_path_cylinder, 2, bounds, _field_names=f, _write_times=t)

            # in case the field is not available the function will return None
            if data is not None:
                export.fit_data(coord, data, f, _n_snapshots_total=len(times))
    export.write_data_to_file()
    """

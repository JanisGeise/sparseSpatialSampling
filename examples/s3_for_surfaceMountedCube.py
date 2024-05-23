"""
    Execute the sparse spatial sampling algorithm on 2D CFD data, export the resulting mesh as XDMF and HDF5 files.
    The test case here is the OpenFoam tutorial

        - surfaceMountedCube, located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/LES/

    IMPORTANT: the size of the data matrix (of the original CFD data) provided for interpolation onto the generated
               coarser grid must be

                - [N_cells, N_dimensions, N_snapshots] (vector field)
                - [N_cells, 1, N_snapshots] (scalar field)

                to correctly execute the 'fit_data()' method of the DataWriter class

    In this example, the cube can be represented by an STL file or by given position and dimensions.
"""
import torch as pt
import pyvista as pv

from typing import Tuple
from os.path import join
from flowtorch.data import FOAMDataloader, mask_box

from s_cube.execute_grid_generation import execute_grid_generation, export_data


def load_cube_data(load_dir: str, boundaries: list) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
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

    return data, xyz, pt.tensor(list(map(float, write_time)))


if __name__ == "__main__":
    # -----------------------------------------   execute for cube   -----------------------------------------
    # path to original surfaceMountedCube simulation (size ~ 8.4 GB, reconstructed)
    load_path_cube = join("..", "data", "3D", "surfaceMountedCube_original_grid_size", "fullCase")
    save_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube", "results")

    # how much of the metric within the original grid should be captured at least
    min_metric = 0.75
    save_name = f"metric_{round(min_metric, 2)}_cube_full_domain"

    # load the CFD data in the given boundaries
    bounds = [[0, 0, 0], [9, 14.5, 2]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    pressure, coord, _ = load_cube_data(load_path_cube, bounds)

    # create a setup for geometry objects for the domain
    domain = {"name": "domain cube", "bounds": bounds, "type": "cube", "is_geometry": False}

    # either define the cube with its dimensions...
    cube = [[3.5, 4, -1], [4.5, 5, 1]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    geometry = {"name": "cube", "bounds": cube, "type": "cube", "is_geometry": True}

    # ... or use the provided STL file
    # cube = pv.PolyData(join("..", "tests", "cube.stl"))
    # geometry = {"name": "cube", "bounds": None, "type": "stl", "is_geometry": True, "coordinates": cube}

    # execute the S^3 algorithm
    export = execute_grid_generation(coord, pt.std(pressure, 1), [domain, geometry], save_path, save_name, "cube",
                                     _min_metric=min_metric)

    # save information about the refinement and grid
    pt.save(export.mesh_info, join(save_path, "mesh_info_cube_variance_{:.2f}.pt".format(min_metric)))

    # export the data
    export_data(export, load_path_cube, bounds)

"""
    Execute the sparse spatial sampling algorithm on 2D CFD data, export the resulting mesh as XDMF and HDF5 files.
    The test case here is the OpenFoam tutorial

        - surfaceMountedCube, located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/LES/

    IMPORTANT: the size of the data matrix (from the original CFD data) provided for interpolation onto the generated
               coarser grid must be

                - [N_cells, N_dimensions, N_snapshots] (vector field)
                - [N_cells, 1, N_snapshots] (scalar field)

                to correctly execute the 'export_data()' method of the DataWriter class

    In this example, the cube can be represented by an STL file or by given position and dimensions.
"""
import torch as pt
import pyvista as pv

from os.path import join

from s3_for_cylinder2D import load_cfd_data
from s_cube.execute_grid_generation import execute_grid_generation, export_openfoam_fields


if __name__ == "__main__":
    # -----------------------------------------   execute for cube   -----------------------------------------
    # path to original surfaceMountedCube simulation (size ~ 8.4 GB, reconstructed)
    load_path = join("..", "data", "3D", "surfaceMountedCube_original_grid_size", "fullCase")
    save_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube", "results")

    # how much of the metric within the original grid should be captured at least
    min_metric = 0.75
    save_name = "metric_{:.2f}".format(min_metric) + "_cube_full_domain"

    # load the CFD data in the given boundaries
    bounds = [[0, 0, 0], [9, 14.5, 2]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    pressure, coord, _ = load_cfd_data(load_path, bounds, n_dims=3)

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
    export_openfoam_fields(export, load_path, bounds)

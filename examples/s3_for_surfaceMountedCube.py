"""
    Execute the sparse spatial sampling algorithm on 3D CFD data, export the resulting mesh and fields as XDMF and HDF5
    files.

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
from s_cube.export_data import export_openfoam_fields
from s_cube.execute_grid_generation import SparseSpatialSampling
from s_cube.load_data import DataLoader

if __name__ == "__main__":
    # path to original surfaceMountedCube simulation (size ~ 8.4 GB, reconstructed)
    load_path = join("..", "data", "3D", "surfaceMountedCube_original_grid_size", "fullCase")
    save_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube",
                     "results")

    # how much of the metric within the original grid should be captured at least
    min_metric = 0.75
    save_name = "metric_{:.2f}".format(min_metric)

    # load the CFD data in the given boundaries
    bounds = [[0, 0, 0], [14.5, 9, 2]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    field, coord, _, write_times = load_cfd_data(load_path, bounds, n_dims=3)

    # create a setup for geometry objects for the domain
    domain = {"name": "domain cube", "bounds": bounds, "type": "cube", "is_geometry": False}

    # either define the cube by its dimensions...
    cube = [[3.5, 4, -1], [4.5, 5, 1]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    geometry = {"name": "cube", "bounds": cube, "type": "cube", "is_geometry": True}

    # ... or use the provided STL file
    # cube = pv.PolyData(join("..", "tests", "cube.stl"))
    # geometry = {"name": "cube", "bounds": None, "type": "stl", "is_geometry": True, "coordinates": cube}

    # execute the S^3 algorithm
    s_cube = SparseSpatialSampling(coord, pt.std(field, 1), [domain, geometry], save_path, save_name, "cube",
                                   min_metric=min_metric, write_times=write_times)

    # execute S^3
    export = s_cube.execute_grid_generation()

    # export the fields available in all time steps
    export_openfoam_fields(export, load_path, bounds, fields=["U", "p"])

    # alternatively, we can export data available at only certain time steps as
    # export.times = [str(i.item()) for i in pt.arange(0.1, 0.5, 0.1)]         # replace with actual time steps
    # export.export_data(coord, field.unsqueeze(1), "p", _n_snapshots_total=None)

    # perform an SVD for the pressure and velocity field
    loader = DataLoader()

    for f in ["p", "U"]:
        # assemble the data matrix
        loader.load_data(save_path, save_name + f"_{f}", f)

        # perform the svd
        loader.compute_svd()

        # write the data to HDF5 & XDMF
        loader.write_data(save_path, save_name + f"_svd_{f}")

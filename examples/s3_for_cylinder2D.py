"""
    Execute the sparse spatial sampling algorithm on 2D CFD data, export the resulting mesh and fields as XDMF and HDF5
    files.

    The test case here is an OpenFoam tutorial with some adjustments to the number of cells and Reynolds number,
    currently:

        - cylinder2D (at Re = 1000), located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/

    IMPORTANT: the size of the data matrix (from the original CFD data) provided for interpolation onto the generated
               coarser grid must be

                - [N_cells, N_dimensions, N_snapshots] (vector field)
                - [N_cells, 1, N_snapshots] (scalar field)

                to correctly execute the 'export_data()' method of the DataWriter class

    In this example, the cylinder is represented by a center and radius (no STL file).
"""
import torch as pt

from os.path import join

from s_cube.load_data import DataLoader
from s_cube.utils import load_cfd_data, export_openfoam_fields
from s_cube.geometry import CubeGeometry, SphereGeometry
from s_cube.sparse_spatial_sampling import SparseSpatialSampling

if __name__ == "__main__":
    # load paths to the CFD data
    load_path = join("..", "data", "2D", "cylinder2D_re1000")
    save_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D", "TEST")

    # how much of the metric within the original grid should be captured at least
    min_metric = 0.95
    save_name = "metric_{:.2f}".format(min_metric) + "_cylinder_full_domain"

    # boundaries of the masked domain for the cylinder
    bounds = [[0, 0], [2.2, 0.41]]          # [[xmin, ymin], [xmax, ymax]]
    cylinder = [[0.2, 0.2], 0.05]         # [[x, y], r]

    # load the CFD data
    field, coord, _, write_times = load_cfd_data(load_path, bounds)

    # create geometry objects for the domain and the cylinder
    domain = CubeGeometry("domain", True, bounds[0], bounds[1])
    geometry = SphereGeometry("cylinder", False, cylinder[0], cylinder[1], refine=True)

    # create a S^3 instance
    s_cube = SparseSpatialSampling(coord, pt.std(field, 1), [domain, geometry], save_path, save_name,
                                   "cylinder2D", min_metric=min_metric, write_times=write_times, max_delta_level=True)
    # execute S^3
    export = s_cube.execute_grid_generation()

    # we used the time steps t = 0.4 ... t_end for computing the metric, but we want to export all time steps, so reset
    # the 'times' property
    export.times = None

    # export the fields available in all time steps
    export_openfoam_fields(export, load_path, bounds)

    # alternatively, we can export data available at only certain time steps as
    # export.times = [str(i.item()) for i in pt.arange(0.1, 0.5, 0.1)]          # replace with actual time steps
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

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
from typing import Tuple, Union
from flowtorch.data import FOAMDataloader, mask_box

from s_cube.load_data import DataLoader
from s_cube.export_data import export_openfoam_fields
from s_cube.execute_grid_generation import SparseSpatialSampling


def load_cfd_data(load_dir: str, boundaries: list, field_name="p", n_dims: int = 2, t_start: Union[int, float] = 0.4,
                  scalar: bool = True) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, list]:
    """
    load the specified field, mask out the defined area.

    Note: vectors are always loaded with all three components (even if the case is 3D) because we don't know on which
          plane the flow problem is defined.

    :param load_dir: path to the simulation data
    :param boundaries: list with list containing the upper and lower boundaries of the mask
    :param field_name: name of the field
    :param n_dims: number of physical dimensions
    :param t_start: starting point in time, all snapshots for which t >= t_start will be loaded
    :param scalar: flag if the field that should be loaded is a scalar- or a vector field
    :return: the field at each write time, x- & y- and z- coordinates (depending on 2D or 3D) of the vertices and all
             write times as list[str]
    """
    # create foam loader object
    _loader = FOAMDataloader(load_dir)

    # load vertices and discard z-coordinate (if 2D)
    vertices = _loader.vertices[:, :n_dims]

    # create a mask
    mask = mask_box(vertices, lower=boundaries[0], upper=boundaries[1])

    # assemble data matrix
    write_time = [t for t in _loader.write_times[1:] if float(t) >= t_start]

    # stack the coordinates to tuples and free up some space
    xyz = pt.stack([pt.masked_select(vertices[:, d], mask) for d in range(n_dims)], dim=1)
    del vertices

    # allocate empty data matrix
    if scalar:
        data = pt.zeros((mask.sum().item(), len(write_time)), dtype=pt.float32)
    else:
        data = pt.zeros((mask.sum().item(), n_dims, len(write_time)), dtype=pt.float32)

        # we always load the vector in 3 dimensions first, so we always need to expand in 3 dimensions
        mask = mask.unsqueeze(-1).expand([xyz.size(0), 3])

    for i, t in enumerate(write_time):
        # load the specified field
        if scalar:
            data[:, i] = pt.masked_select(_loader.load_snapshot(field_name, t), mask)
        else:
            data[:, :, i] = pt.masked_select(_loader.load_snapshot(field_name, t), mask).reshape(mask.size())

    # get the cell area
    _cell_area = _loader.weights.sqrt().unsqueeze(-1)

    return data, xyz, _cell_area, write_time


if __name__ == "__main__":
    # load paths to the CFD data
    load_path = join("..", "data", "2D", "cylinder2D_re1000")
    save_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D", "TEST")

    # how much of the metric within the original grid should be captured at least
    min_metric = 0.95
    save_name = "metric_{:.2f}".format(min_metric) + "_cylinder_full_domain"

    # boundaries of the masked domain for the cylinder
    bounds = [[0, 0], [2.2, 0.41]]          # [[xmin, ymin], [xmax, ymax]]
    cylinder = [[0.2, 0.2], [0.05]]         # [[x, y], [r]]

    # load the CFD data
    field, coord, _, write_times = load_cfd_data(load_path, bounds)

    # create a setup for geometry objects for the domain and the cylinder, we don't use any STL files
    domain = {"name": "domain cylinder", "bounds": bounds, "type": "cube", "is_geometry": False}
    geometry = {"name": "cylinder", "bounds": cylinder, "type": "sphere", "is_geometry": True}

    # create a S^3 instance
    s_cube = SparseSpatialSampling(coord, pt.std(field, 1), [domain, geometry], save_path, save_name,
                                   "cylinder2D", min_metric=min_metric, write_times=write_times)

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

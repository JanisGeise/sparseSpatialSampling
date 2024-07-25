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

                to correctly execute the 'export()' method of the Datawriter class. If the data was created with,
                OpenFoam s_cube.utils.export_openfoam_fields can be used alternatively to automatically export a given
                number of fields.

    In this example, the cylinder is represented by a center and radius (no STL file).
"""
import logging
import torch as pt

from os.path import join
from typing import Union

from s_cube.export import ExportData
from s_cube.data import Datawriter, Dataloader
from s_cube.geometry import CubeGeometry, SphereGeometry
from s_cube.sparse_spatial_sampling import SparseSpatialSampling
from s_cube.utils import load_cfd_data, export_openfoam_fields, compute_svd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_svd_s_cube_to_file(field_names: Union[list, str], load_dir: str, file_name: str,
                             new_file: bool, n_modes: int = None, rank=None) -> None:
    """
    computes an SVD for a given number of fields and exports the results to HDF5 & XDMF for visualizing,
     e.g., in ParaView

    :param field_names: names of the fields for which the SVD should be computed
    :param load_dir: directory from which the results of S^3 should be loaded from, the results of the SVD will be
                     written into the same directory
    :param file_name: the name of the file from which the data should be loaded, for clarity '_svd' will be appended to
                      the file name which contains the results of the SVD
    :param new_file: flag if all exported fields are located in a single HDF5 file or if each field is written into a
                     separate file
    :param n_modes: number of modes to write to the file, if larger than available modes, all available modes will be
                    written
    :param rank: number of modes which should be used to compute the SVD, if 'None' then the optimal rank will be used
    :return: None
    """
    if type(field_names) is str:
        field_names = [field_names]

    for f in field_names:
        _name = f"{file_name}_{f}" if new_file else file_name
        dataloader = Dataloader(load_dir, f"{_name}.h5")

        # assemble a datamatrix for computing an SVD
        dm_u = dataloader.load_snapshot(f)

        # perform an SVD weighted with cell areas
        s, U, V = compute_svd(dm_u, dataloader.weights, rank)

        # write the data to HDF5 & XDMF
        datawriter = Datawriter(load_dir, file_name + f"_{f}_svd.h5")

        # write the grid
        datawriter.write_grid(dataloader)

        # set the max. number of modes to write, if the specified number of modes is larger than the available modes,
        # then only write all available modes
        n_modes = U.size(-1) if n_modes is None else n_modes
        if n_modes > U.size(-1):
            logger.warning(f"Number of modes to write is set to {n_modes}, but found only {U.size(-1)} modes to write.")
            n_modes = U.size(-1)

        # write the modes as vectors, where each mode is treated as an independent vector
        for i in range(n_modes):
            if len(U.size()) == 2:
                datawriter.write_data(f"mode_{i + 1}", group="constant", data=U[:, i].squeeze())
            else:
                datawriter.write_data(f"mode_{i + 1}", group="constant", data=U[:, :, i].squeeze())

        # write the rest as tensor (not referenced in XDMF file anyway)
        datawriter.write_data("V", group="constant", data=V)
        datawriter.write_data("s", group="constant", data=s)

        # write XDMF file
        datawriter.write_xdmf_file()


if __name__ == "__main__":
    # load paths to the CFD data
    load_path = join("..", "data", "2D", "cylinder2D_re1000")
    save_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D", "results")

    # how much of the metric within the original grid should be captured at least
    min_metric = 0.2
    save_name = "metric_{:.2f}".format(min_metric) + "_cylinder_full_domain"

    # boundaries of the masked domain for the cylinder
    bounds = [[0, 0], [2.2, 0.41]]  # [[xmin, ymin], [xmax, ymax]]
    cylinder = [[0.2, 0.2], 0.05]  # [[x, y], r]

    # load the CFD data
    field, coord, _, write_times = load_cfd_data(load_path, bounds)

    # create geometry objects for the domain and the cylinder
    domain = CubeGeometry("domain", True, bounds[0], bounds[1])
    geometry = SphereGeometry("cylinder", False, cylinder[0], cylinder[1], refine=True)

    # create a S^3 instance
    s_cube = SparseSpatialSampling(coord, pt.std(field, 1), [domain, geometry], save_path, save_name,
                                   "cylinder2D", min_metric=min_metric, write_times=write_times, max_delta_level=True)

    # execute S^3
    s_cube.execute_grid_generation()

    # create export instance, export all fields into the same HFD5 file and create single XDMF from it
    export = ExportData(s_cube, write_new_file_for_each_field=False)

    # we used the time steps t = 0.4 ... t_end for computing the metric, but we want to export all time steps, so reset
    # the 'times' property
    export.write_times = None

    # export the fields available in all time steps
    export_openfoam_fields(export, load_path, bounds)

    # alternatively, we can export data available at only certain time steps as
    # export.write_times = [str(i.item()) for i in pt.arange(0.1, 0.5, 0.1)]          # replace with actual time steps
    # export.export(coord, field.unsqueeze(1), "p", _n_snapshots_total=None)

    # compute SVD on grid generated by S^3 and export the results to HDF5 & XDMF
    write_svd_s_cube_to_file(["p", "U"], save_path, save_name, export.new_file, 50)

"""
    Execute the sparse spatial sampling algorithm on 3D CFD data, export the resulting mesh as XDMF and HDF5 files.
    The test case here is a 3D flow past a cylinder at Re = 3900, which can be found in the flow_data repository:

    https://github.com/AndreWeiner/flow_data/

    In this example, the cylinder is executed for a long time span, generating a large amount of data.
    Since these data are not fitting into the RAM all at once, the computation of the metric as well as the
    interpolation and export is performed snapshot-by-snapshot

    TODO: - once flowtorch contains the implementation for loading tensors
                -> replace avg. TKE computation with loading UPrime2Mean field of specified time step
                -> much lower computational costs
          - once parallel SVD is implemented -> use to compute SVD for large datasets
"""

import logging
import torch as pt

from os.path import join
from typing import Union
from os import path, makedirs

from flowtorch.data import FOAMDataloader, mask_box

from sparseSpatialSampling.export import ExportData
from sparseSpatialSampling.geometry import CubeGeometry, CylinderGeometry3D
from sparseSpatialSampling.utils import load_original_Foam_fields
from examples.s3_for_cylinder2D_Re100 import write_svd_s_cube_to_file
from sparseSpatialSampling.sparse_spatial_sampling import SparseSpatialSampling

logger = logging.getLogger(__name__)


def load_cube_coordinates_and_times(load_dir: str, boundaries: list, scalar: bool = True, _compute_metric: bool = True,
                                    _field_name: str = "p", time_boundaries: list = None):
    """
    Load the vertices within the given boundaries and write times of the original simulation. If specified, compute the
    metric as standard deviation wrt time of the loaded field as:

        std = sqrt(1/N sum((x_i - mu)^2))

    If the loaded field is a vector field, the mean turbulent kinetic energy will be computed (it is assumed that the
    defined vector field (arg 'field_name') is the velocity field)

    :param load_dir: path to the simulation data
    :param boundaries: list with list containing the upper and lower boundaries of the mask
    :param scalar: flag if the field is a scalar field or a vector field
    :param _compute_metric: flag if the metric should be computed
    :param _field_name: name of the field, which should be loaded
    :param time_boundaries: start and end time of the sequence to load, if 'None' all available fields except zero are
                            loaded
    :return: x-, y- & z-coordinates of the cells, all write times and the metric if specified
    """
    # create foam loader object
    loader = FOAMDataloader(load_dir)

    # get the defined boundaries for start and end time to use if provided
    if time_boundaries is not None:
        idx = sorted([i for i, t in enumerate(loader.write_times) if t in time_boundaries])
        write_times = loader.write_times[idx[0]:idx[1]+1]

    # else use all times steps but zero
    else:
        write_times = loader.write_times[1:]

    # load vertices
    vertices = loader.vertices
    mask = mask_box(vertices, lower=boundaries[0], upper=boundaries[1])

    # stack the coordinates to tuples
    xyz = pt.stack([pt.masked_select(vertices[:, 0], mask), pt.masked_select(vertices[:, 1], mask),
                    pt.masked_select(vertices[:, 2], mask)], dim=1)

    # free up some memory
    del vertices

    if _compute_metric:
        if not scalar:
            # we always load the vector in 3 dimensions first, so we always need to expand in 3 dimensions
            mask = mask.unsqueeze(-1).expand([xyz.size(0), 3])

            # allocate empty tensors for avg. field
            avg_field = pt.zeros((xyz.size(0), 3))
        else:
            avg_field = pt.zeros((xyz.size(0),))

        # compute the avg. wrt time
        for t_i in write_times:
            logger.info(f"Loading time step {t_i} / {write_times[-1]} for computing the avg. field")

            # load the specified field
            if scalar:
                avg_field += pt.masked_select(loader.load_snapshot(_field_name, t_i), mask)
            else:
                avg_field += pt.masked_select(loader.load_snapshot(_field_name, t_i), mask).reshape(mask.size())
        avg_field /= len(write_times)

        # compute the TKE (vector field) or the std. dev. of the field (scalar field)
        if not scalar:
            avg_tke = pt.zeros((xyz.size(0),))
            for t_i in write_times:
                logger.info(f"Loading time step {t_i} / {write_times[-1]} for computing the TKE")
                # TKE = 0.5 * (u'^2 + v'^2 + w'^2), with U' = 1 / N sum_i_N((U_i - U_mean)^2)
                avg_tke += 0.5 * (pt.masked_select(loader.load_snapshot(_field_name, t_i),
                                                   mask).reshape(mask.size()) - avg_field).pow(2).sum(1)
            avg_tke /= len(write_times)

            return xyz, avg_tke, write_times

        else:
            # the std. tensor is always 1D
            std_field = pt.zeros((xyz.size(0),))

            # compute the standard deviation wrt time
            for t_i in write_times:
                logger.info(f"Loading time step {t_i} for computing the std. dev.")
                std_field += (pt.masked_select(loader.load_snapshot(_field_name, t_i), mask) - avg_field).pow(2)
            std_field /= len(write_times)

            return xyz, std_field.sqrt(), write_times

    else:
        return xyz, write_times


def export_fields_snapshot_wise(load_dir: str, datawriter: ExportData, field_names: list, boundaries: list,
                                write_times: Union[str, list], batch_size: int = 25) -> None:
    """
    For each field specified, interpolate all snapshots onto the generated grid and export it to HDF5 & XDMF. The
    interpolation and export of the data is performed snapshot-by-snapshot (batch_size = 1) or in batches to avoid out
    of memory issues for large datasets.

    :param load_dir: path to the simulation data
    :param datawriter: DataWriter class after executing the S^3 algorithm
    :param field_names: names of the fields to export
    :param boundaries: boundaries of the masked area of the domain (needs to be the same as used for loading the
                       vertices and computing the metric)
    :param write_times: the write times of the simulation
    :param batch_size: batch size, number of snapshots which should be interpolated and exported at once
    :return: None
    """
    write_times = [write_times] if type(write_times) is str else write_times
    for f in field_names:
        if f.endswith("Mean"):
            # int, because in OpenFoam only significant decimal points are written, e.g., t = 30 is not written as 30.0
            # tolist, because otherwise we would have to call _write_times=t.item()
            # TODO: adjust times for mean fields
            datawriter.write_times = [str(i.item()) for i in pt.arange(30, 140, 10).int()]
        else:
            datawriter.write_times = write_times

        counter = 1
        if not len(datawriter.write_times) % batch_size:
            n_batches = int(len(datawriter.write_times) / batch_size)
        else:
            n_batches = int(len(datawriter.write_times) / batch_size) + 1
        for i in pt.arange(0, len(datawriter.write_times), step=batch_size).tolist():
            logger.info(f"Exporting batch {counter} / {n_batches}")
            coordinates, data = load_original_Foam_fields(load_dir, datawriter.n_dimensions, boundaries, _field_names=f,
                                                          _write_times=datawriter.write_times[i:i+batch_size])

            # in case the field is not available, the export()-method will return None
            if data is not None:
                datawriter.export(coordinates, data, f, _n_snapshots_total=len(datawriter.write_times))
            counter += 1


if __name__ == "__main__":
    # -----------------------------------------   execute for cylinder3D   -----------------------------------------
    # path to original cylinder3D simulation
    load_path = join("/media", "janis", "Elements", "Janis", "cylinder_3D_Re3900_tests", "cylinder_3D_Re3900")
    save_path = join("..", "run", "final_benchmarks", "cylinder3D_Re3900_local_TKE",
                     "results_no_geometry_refinement_no_dl_constraint")
    save_name = "cylinder3D_Re3900"

    # load an exiting s_cube object or start new
    load_existing = False

    # for which field should we compute the metric?
    field_name = "U"
    scalar_field = False

    # fields which should be exported, the write times for the fields can be adjusted in the function
    # export_fields_snapshot_wise
    # fields = ["p", "U", "pMean", "UMean", "pPrime2Mean"]
    fields = ["U"]

    # how much of the metric within the original grid should be captured at least
    min_metric = pt.arange(0.25, 1.05, 0.05)

    # compute the metric or load an existing one (we only have the velocity and pressure fields for this simulation)
    compute_metric = False
    save_name_metric = "metric_avg_TKE" if field_name == "U" else "metric_std_pressure"

    # load the CFD data in the given boundaries (full domain) and compute the metric snapshot-by-snapshot
    d = 0.1
    bounds = [[0, 0, 0], [2.4, 2.0, pt.pi * d]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    # boundaries for the time steps to load, if None all available time steps except the zero time step are used;
    # the boundaries are included and have to be strings
    time_bounds = ["0.19225", "0.49225"]

    # load the vertices and write times, compute / load the metric
    if compute_metric:
        logger.info("Loading coordinates and computing metric.")
        coord, metric, times = load_cube_coordinates_and_times(load_path, bounds, _compute_metric=True,
                                                               _field_name=field_name, scalar=scalar_field,
                                                               time_boundaries=time_bounds)

        # save the metric, so we don't need to compute it again
        if not path.exists(save_path):
            makedirs(save_path)

        pt.save(metric, join(save_path, f"{save_name_metric}.pt"))
    else:
        logger.info("Loading coordinates and metric.")
        coord, times = load_cube_coordinates_and_times(load_path, bounds, _compute_metric=False)
        metric = pt.load(join(save_path, f"{save_name_metric}.pt"))

    # define the geometries for the domain and the cylinder, increase the height of the cylinder to make sure it is
    # masked out completely
    geometry = [CubeGeometry("domain", True, bounds[0], bounds[1]),
                CylinderGeometry3D("cylinder", False, [(0.8, 1.0, -1), (0.8, 1.0, 1)], d / 2,
                                   refine=True)
                ]

    # execute the S^3 algorithm and export the specified fields
    for m in min_metric:
        # overwrite save name
        new_save_name = f"{save_name}_{save_name_metric}" + "_{:.2f}".format(m)

        if load_existing:
            logger.info(f"Loading s_cube object for metric {m}.")
            s_cube = pt.load(join(save_path, f"s_cube_{new_save_name}.pt"))

            # set the (new) save path within the s_cube object
            s_cube.train_path = save_path
        else:
            s_cube = SparseSpatialSampling(coord, metric, geometry, save_path, new_save_name, "cylinder", min_metric=m,
                                           n_jobs=6)

            # execute S^3
            s_cube.execute_grid_generation()

        # create export instance, export all fields into the same HFD5 file and create single XDMF from it
        export = ExportData(s_cube, write_new_file_for_each_field=False)

        # export the fields snapshot-by-snapshot (batch_size = 1) or in batches
        # TODO: change times back once full time series should be exported
        export_fields_snapshot_wise(load_path, export, fields, bounds, times[0], batch_size=10)

        # compute SVD on grid generated by S^3 and export the results to HDF5 & XDMF
        # write_svd_s_cube_to_file([f for f in fields if "Mean" not in f], save_path, save_name, export.new_file, 50,
        #                          rank=int(1e5))

"""
    Execute the sparse spatial sampling algorithm on 3D CFD data, export the resulting mesh as XDMF and HDF5 files.
    The test case here is the OpenFoam tutorial

        - surfaceMountedCube, located under: $FOAM_TUTORIALS/incompressible/pimpleFoam/LES/

    In this example, the cube is executed for a long time span, generating a large amount of data. Since these data are
    not fitting into the RAM all at once, the computation of the metric as well as the interpolation and export is
    performed snapshot-by-snapshot
"""
import logging
import torch as pt

from os.path import join
from os import path, makedirs

from flowtorch.data import FOAMDataloader, mask_box

from s_cube.export import ExportData
from s_cube.geometry import CubeGeometry
from s_cube.utils import load_original_Foam_fields
from examples.s3_for_cylinder2D import write_svd_s_cube_to_file
from s_cube.sparse_spatial_sampling import SparseSpatialSampling

logger = logging.getLogger(__name__)


def load_cube_coordinates_and_times(load_dir: str, boundaries: list, scalar: bool = True, _compute_metric: bool = True,
                                    _field_name: str = "p"):
    """
    Load the vertices within the given boundaries and write times of the original simulation. If specified, compute the
    metric as standard deviation wrt time of the loaded field as:

        std = sqrt(1/N sum((x_i - mu)^2))

    If the loaded field is a vector field, the turbulent kinetic energy will be computed (it is assumed that the only
    available vector field is the velocity field)

    :param load_dir: path to the simulation data
    :param boundaries: list with list containing the upper and lower boundaries of the mask
    :param scalar: flag if the field is a scalar field or a vector field
    :param _compute_metric: flag if the metric should be computed
    :param _field_name: name of the field, which should be loaded
    :return: x-, y- & z-coordinates of the cells, all write times and the metric if specified
    """
    # create foam loader object
    loader = FOAMDataloader(load_dir)

    # all times steps but zero
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
                # for vector fields, we use the magnitude
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
                                write_times: list, batch_size: int = 25) -> None:
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
    for f in field_names:
        if f.endswith("Mean"):
            # int, because in OpenFoam only significant decimal points are written, e.g., t = 30 is not written as 30.0
            # tolist, because otherwise we would have to call _write_times=t.item()
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
    # -----------------------------------------   execute for cube   -----------------------------------------
    # path to original surfaceMountedCube simulation
    load_path = join("/media", "janis", "Elements", "FOR_data", "surfaceMountedCube_Janis", "fullCase")
    # load_path = join("..", "data", "3D", "surfaceMountedCube", "fullCase")
    # save_path = join("/media", "janis", "Elements", "FOR_data", "surfaceMountedCube_s_cube_Janis")
    save_path = join("..", "run", "final_benchmarks", "surfaceMountedCube_local_TKE",
                     "results_no_geometry_refinement_no_dl_constraint")

    # load an exiting s_cube object or start new
    load_existing = True

    # for which field should we compute the metric?
    field_name = "U"
    scalar_field = False

    # fields which should be exported, the write times for the fields can be adjusted in the function
    # export_fields_snapshot_wise
    fields = ["p", "U", "pMean", "UMean", "pPrime2Mean"]

    # how much of the metric within the original grid should be captured at least
    min_metric = pt.arange(0.25, 1.05, 0.05)

    # compute the metric or load an existing one (we only have the velocity and pressure fields for this simulation)
    compute_metric = True
    save_name_metric = "metric_TKE" if field_name == "U" else "metric_std_pressure"

    # load the CFD data in the given boundaries (full domain) and compute the metric snapshot-by-snapshot
    bounds = [[0, 0, 0], [14.5, 9, 2]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    # load the vertices and write times, compute / load the metric
    if compute_metric:
        logger.info("Loading coordinates and computing metric.")
        coord, metric, times = load_cube_coordinates_and_times(load_path, bounds, _compute_metric=True,
                                                               _field_name=field_name, scalar=scalar_field)

        # save the metric, so we don't need to compute it again
        if not path.exists(save_path):
            makedirs(save_path)

        pt.save(metric, join(save_path, f"{save_name_metric}.pt"))
    else:
        logger.info("Loading coordinates and metric.")
        coord, times = load_cube_coordinates_and_times(load_path, bounds, _compute_metric=False)
        metric = pt.load(join(save_path, f"{save_name_metric}.pt"))

    # define the geometries for the domain and the cube
    geometry = [CubeGeometry("domain", True, bounds[0], bounds[1]),
                CubeGeometry("cube", False, [3.5, 4, -1], [4.5, 5, 1])]

    # execute the S^3 algorithm and export the specified fields
    for m in min_metric:
        # overwrite save name
        save_name = f"surfaceMountedCube_{save_name_metric}" + "_{:.2f}".format(m)

        if load_existing:
            logger.info(f"Loading s_cube object for metric {m}.")
            s_cube = pt.load(join(save_path, f"s_cube_{save_name}.pt"))
        else:
            s_cube = SparseSpatialSampling(coord, metric, geometry, save_path, save_name, "cube", min_metric=m)

            # execute S^3
            s_cube.execute_grid_generation()

        # create export instance, export all fields into the same HFD5 file and create single XDMF from it
        export = ExportData(s_cube, write_new_file_for_each_field=False)

        # export the fields snapshot-by-snapshot (batch_size = 1) or in batches
        export_fields_snapshot_wise(load_path, export, fields, bounds, times)

        # compute SVD on grid generated by S^3 and export the results to HDF5 & XDMF
        write_svd_s_cube_to_file([f for f in fields if "Mean" not in f], save_path, save_name, export.new_file, 50)

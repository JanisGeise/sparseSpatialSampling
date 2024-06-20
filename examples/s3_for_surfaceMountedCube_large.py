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

from s_cube import DataWriter
from s_cube.execute_grid_generation import execute_grid_generation, load_original_Foam_fields

logger = logging.getLogger(__name__)


def load_cube_coordinates_and_times(load_dir: str, boundaries: list, scalar: bool = True, _compute_metric: bool = True,
                                    field_name: str = "p"):
    """
    Load the vertices within the given boundaries and write times of the original simulation. If specified, compute the
    metric as standard deviation wrt time of the loaded field as:

        std = sqrt(1/N sum(x_i - mu)^2)

    If the loaded field is a vector field, the standard deviation of its magnitude will be computed.

    :param load_dir: path to the simulation data
    :param boundaries: list with list containing the upper and lower boundaries of the mask
    :param scalar: flag if the field is a scalar field or a vector field
    :param _compute_metric: flag if the metric should be computed
    :param field_name: name of the field, which should be loaded
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
        # allocate empty tensors for avg. and std.
        avg_field = pt.zeros((xyz.size(0),))
        std_field = pt.zeros((xyz.size(0),))

        if not scalar:
            # we always load the vector in 3 dimensions first, so we always need to expand in 3 dimensions
            mask = mask.unsqueeze(-1).expand([xyz.size(0), 3])

        # compute the avg. wrt time
        for t_i in write_times:
            logger.info(f"Loading time step {t_i} for computing the avg.")

            # load the specified field
            if scalar:
                avg_field += pt.masked_select(loader.load_snapshot(field_name, t_i), mask)
            else:
                # for vector fields, we use the magnitude
                avg_field += pt.masked_select(loader.load_snapshot(field_name, t_i),
                                              mask).reshape(mask.size()).pow(2).sum(1).sqrt()
        avg_field /= len(write_times)

        # compute the standard deviation wrt time
        for t_i in write_times:
            logger.info(f"Loading time step {t_i} for computing the std.")

            # load the specified field
            if scalar:
                std_field += (pt.masked_select(loader.load_snapshot(field_name, t_i), mask) - avg_field).pow(2)
            else:
                _mag_field = pt.masked_select(loader.load_snapshot(field_name, t_i),
                                              mask).reshape(mask.size()).pow(2).sum(1).sqrt()
                std_field += (_mag_field - avg_field).pow(2)
        std_field /= len(write_times)

        return xyz, std_field.sqrt(), write_times

    else:
        return xyz, write_times


def export_fields_snapshot_wise(load_dir: str, datawriter: DataWriter, field_names: list, boundaries: list,
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
        # in our setup, the write times differ for the p & U field and the prime2Mean fields
        if f.endswith("Prime2Mean"):
            datawriter.times = [str(i.item()) for i in pt.arange(40, 140, 10).int()]
        elif f.endswith("Mean"):
            # int, because in OpenFoam only significant decimal points are written, e.g., t = 30 is not written as 30.0
            # tolist, because otherwise we would have to call _write_times=t.item()
            datawriter.times = [str(i.item()) for i in pt.arange(30, 140, 10).int()]
        else:
            datawriter.times = write_times

        counter = 1
        if not len(datawriter.times) % batch_size:
            n_batches = int(len(datawriter.times) / batch_size)
        else:
            n_batches = int(len(datawriter.times) / batch_size) + 1
        for i in pt.arange(0, len(datawriter.times), step=batch_size).tolist():
            logger.info(f"Exporting batch {counter} / {n_batches}")
            coordinates, data = load_original_Foam_fields(load_dir, datawriter.n_dimensions, boundaries, _field_names=f,
                                                          _write_times=datawriter.times[i:i+batch_size])

            # in case the field is not available, the export()-method will return None
            if data is not None:
                datawriter.export_data(coordinates, data, f, _n_snapshots_total=len(datawriter.times))
            counter += 1


if __name__ == "__main__":
    # -----------------------------------------   execute for cube   -----------------------------------------
    # path to original surfaceMountedCube simulation
    load_path = join("/media", "janis", "Elements", "FOR_data", "surfaceMountedCube_Janis", "fullCase")
    save_path = join("/media", "janis", "Elements", "FOR_data", "surfaceMountedCube_s_cube_Janis")

    # for which field should we compute the metric?
    field_name = "U"
    scalar_field = False

    # fields which should be exported, the write times for the fields can be adjusted in the function
    # export_fields_snapshot_wise
    fields = ["p", "U", "pMean", "UMean", "pPrime2Mean"]

    # how much of the metric within the original grid should be captured at least
    min_metric = pt.arange(0.25, 1.05, 0.05)

    # compute the metric or load an existing one (we only have the velocity and pressure fields for this simulation)
    compute_metric = False
    save_name_metric = "metric_std_mag_velocity" if field_name == "U" else "metric_std_pressure"

    # load the CFD data in the given boundaries (full domain) and compute the metric (std(p)) snapshot-by-snapshot
    bounds = [[0, 0, 0], [14.5, 9, 2]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    # load the vertices and write times, compute / load the metric
    if compute_metric:
        logger.info("Loading coordinates and computing metric.")
        coord, metric, times = load_cube_coordinates_and_times(load_path, bounds, _compute_metric=True,
                                                               field_name=field_name, scalar=scalar_field)

        # save the metric, so we don't need to compute it again
        if not path.exists(save_path):
            makedirs(save_path)

        pt.save(metric, join(save_path, f"{save_name_metric}.pt"))
    else:
        logger.info("Loading coordinates and metric.")
        coord, times = load_cube_coordinates_and_times(load_path, bounds, _compute_metric=False)
        metric = pt.load(join(save_path, f"{save_name_metric}.pt"))

    # define the geometries for the domain and the cube
    geometry = [{"name": "domain cube", "bounds": bounds, "type": "cube", "is_geometry": False},
                {"name": "cube", "bounds": [[3.5, 4, -1], [4.5, 5, 1]], "type": "cube", "is_geometry": True}]

    # execute the S^3 algorithm
    for m in min_metric:
        # overwrite save name
        save_name = f"surfaceMountedCube_{save_name_metric}" + "_{:.2f}".format(m)
        export = execute_grid_generation(coord, metric, geometry, save_path, save_name, "cube",
                                         _min_metric=m, _refine_geometry=False)

        # save information about the refinement and grid
        pt.save(export.mesh_info, join(save_path, f"mesh_info_{save_name}.pt"))

        # export the fields snapshot-by-snapshot (batch_size = 1) or in batches
        export_fields_snapshot_wise(load_path, export, fields, bounds, times)

        # perform an SVD for the pressure and velocity field
        for f in ["p", "U"]:
            export.compute_svd(f)

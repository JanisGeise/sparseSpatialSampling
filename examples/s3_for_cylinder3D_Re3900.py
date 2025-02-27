"""
    Execute the sparse spatial sampling algorithm on 3D CFD data, export the resulting mesh as XDMF and HDF5 files.
    The test case here is a 3D flow past a cylinder at Re = 3900, which can be found in the flow_data repository:

    https://github.com/AndreWeiner/flow_data/

    In this example, the cylinder is executed for a long time span, generating a large amount of data.
    Since these data are not fitting into the RAM all at once, the computation of the metric as well as the
    interpolation and export is performed snapshot-by-snapshot
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
    save_path = join("..", "run", "final_benchmarks", "cylinder3D_Re3900_local_TKE", "test_new_metric")
    save_name = "cylinder3D_Re3900"

    # load an exiting s_cube object or start new
    load_existing = False

    vec1 = pt.tensor([1, 1, 1]).type(pt.float64)
    vec2 = pt.tensor([1, 2, 3]).type(pt.float64)
    res = pt.cross(vec1, vec2)

    # for which field should we compute the metric?
    field_name = "UPrime2Mean"

    # fields which should be exported, the write times for the fields can be adjusted in the function
    # export_fields_snapshot_wise
    fields = ["pMean", "p", "U", "UMean", "pPrime2Mean", "UPrime2Mean"]

    # how much of the metric within the original grid should be captured at least
    # min_metric = pt.arange(0.25, 1.05, 0.05)
    min_metric = [0.01]

    # load the CFD data in the given boundaries (full domain) and compute the metric snapshot-by-snapshot
    d = 0.1
    bounds = [[0, 0, 0], [2.4, 2.0, pt.pi * d]]              # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    # load Prim2Mean field of last dt and compute TKE from it
    logger.info("Loading coordinates and computing the metric.")
    loader = FOAMDataloader(load_path)
    prime2Mean = loader.load_snapshot(field_name, loader.write_times[-1])

    # compute the TKE; the order of 'volSymmTensorField' is: [XX, XY, XZ, YY, YZ, ZZ]
    metric = 0.5 * prime2Mean[:, [0, 3, 5]].sum(-1)

    # define the geometries for the domain and the cylinder, increase the height of the cylinder to make sure it is
    # masked out completely
    geometry = [CubeGeometry("domain", True, bounds[0], bounds[1]),
                CylinderGeometry3D("cylinder", False, [(0.8, 1.0, -1), (0.8, 1.0, 1)], d / 2,
                                   refine=True)
                ]

    # execute the S^3 algorithm and export the specified fields
    for m in min_metric:
        if load_existing:
            logger.info(f"Loading s_cube object for metric {m}.")
            s_cube = pt.load(join(save_path, f"s_cube_{save_name}.pt"))

            # set the (new) save path within the s_cube object
            s_cube.train_path = save_path
        else:
            s_cube = SparseSpatialSampling(loader.vertices, metric, geometry, save_path, save_name, "cylinder",
                                           min_metric=m, n_jobs=6)

            # execute S^3
            s_cube.execute_grid_generation()

        # create export instance, export all fields into the same HFD5 file and create single XDMF from it
        export = ExportData(s_cube, write_new_file_for_each_field=False)

        # export the fields snapshot-by-snapshot (batch_size = 1) or in batches
        export_fields_snapshot_wise(load_path, export, fields, bounds, loader.write_times, batch_size=10)

        # compute SVD on grid generated by S^3 and export the results to HDF5 & XDMF
        # write_svd_s_cube_to_file([f for f in fields if "Mean" not in f], save_path, save_name, export.new_file, 50,
        #                          rank=int(1e5))

"""
execute s^3 for the generic metrics.
"""
import torch as pt
from os.path import join

from s3_for_OAT15_airfoil import load_airfoil_from_stl_file
from sparseSpatialSampling.export import ExportData
from sparseSpatialSampling.geometry import CubeGeometry, GeometryCoordinates2D
from sparseSpatialSampling.sparse_spatial_sampling import SparseSpatialSampling


def load_original_field(load_dir: str, field_name: str) -> pt.Tensor:
    print(f"Loading snapshots for field {field_name}.")

    if field_name == "vel":
        _field = []
        for cmp in ["x", "y", "z"]:
            _field.append(pt.load(join(load_dir, f"{field_name}_{cmp}_large_every10.pt"), weights_only=False).unsqueeze(-1))
        _field = pt.stack(_field, dim=-1).squeeze()
    else:
        _field = pt.load(join(load_dir, f"{field_name}_large_every10.pt"), weights_only=False)
    return _field


def export_fields(export_object, field_names: list, load_dir: str) -> None:
    for f in field_names:
        _field = load_original_field(load_dir, f)

        # we need to add one dimension if we have a scalar field
        if len(_field.size()) == 2:
            export_object.export(xz, _field.unsqueeze(1), f)
        else:
            export_object.export(xz, _field, f)
        del _field

if __name__ == "__main__":
    load_path = join("..", "data", "2D", "OAT15")
    load_path_metric = join("..", "run", "final_benchmarks", "OAT15_large_new", "results_generic_metrics")

    # fields to load / export
    metrics = ["metric_mu_only", "metric_sigma_only", "metric_mu_and_sigma"]
    fields = ["p", "rho", "u", "ma"]
    min_variance = 0.75

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join(load_path, "vertices_and_masks.pt"))
    xz = pt.stack([xz[f"x_large"], xz[f"z_large"]], dim=-1)
    bounds = [[pt.min(xz[:, 0]).item(), pt.min(xz[:, 1]).item()], [pt.max(xz[:, 0]).item(), pt.max(xz[:, 1]).item()]]

    # load the metrics
    metric_fields = [pt.load(join(load_path_metric, f"{m}.pt"), weights_only=False) for m in metrics]

    # load the corresponding write times
    times = pt.load(join(load_path, "oat15_tandem_times.pt"))[::10]

    # load the airfoil geometries of the leading airfoil from an STL file
    oat15 = load_airfoil_from_stl_file(join(load_path, "oat15_airfoil_no_TE.stl"), dimensions="xz")
    naca = load_airfoil_from_stl_file(join(load_path, "naca_airfoil_no_TE.stl"), dimensions="xz")

    # create a geometry object for the domain and the OAT airfoil (loaded from coordinates)
    geometry = [CubeGeometry("domain", True, bounds[0], bounds[1]),
                GeometryCoordinates2D("OAT15", False, oat15, refine=True),
                GeometryCoordinates2D("NACA", False, naca, refine=True)]

    # loop over the metrics and run Scube
    for i, m in enumerate(metric_fields):
        # update the save name
        save_name = metrics[i] + "_variance_{:.2f}".format(min_variance)

        # instantiate an S^3 object
        s_cube = SparseSpatialSampling(xz, m, geometry, load_path_metric, save_name, "OAT15",
                                       min_metric=min_variance, n_jobs=8, max_delta_level=False,
                                       reach_at_least=0.95)

        # execute S^3
        s_cube.execute_grid_generation()

        # create export instance, export all fields into the same HFD5 file and create single XDMF from it
        export = ExportData(s_cube, write_times=times.tolist())

        # loop over the specified fields and export them onto the S^3 grid
        export_fields(export, fields, load_path)

"""
    compute the error between the generated grid and the original grid wrt space, time and space & time
"""
import regex as re
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from stl import mesh
from glob import glob
from os.path import join
from os import path, makedirs
from matplotlib.patches import Polygon
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.colors as colors

from sparseSpatialSampling.data import Dataloader


def load_airfoil_as_stl_file(_load_path: str, _name: str = "oat15.stl", dimensions: str = "xy"):
    """
    same as in "examples/s3_for_OAT15_airfoil", but relative import not working, so for now just copy the function into
    this directory
    """
    # mapping for the coordinate directions
    dim_mapping = {"x": 0, "y": 1, "z": 2}
    dimensions = [dim_mapping[d] for d in dimensions.lower()]

    # load stl file
    stl_file = mesh.Mesh.from_file(_load_path)
    coord_af = np.stack([stl_file.x[:, 0], stl_file.y[:, 0], stl_file.z[:, 0]], -1)
    coord_af = coord_af[:, dimensions]
    _, i = np.unique(coord_af, axis=0, return_index=True)
    coord_af = coord_af[np.sort(i)]
    coord_af = np.append(coord_af, np.expand_dims(coord_af[0, :], axis=0), axis=0)

    return coord_af


def plot_metric_original_grid(coord_x_orig: pt.Tensor, coord_y_orig: pt.Tensor, metric_orig: pt.Tensor,
                              save_dir: str, geometry_: list = None, field_: str = "p") -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    vmin, vmax = metric_orig.min(), metric_orig.max()
    tcf = ax.tricontourf(coord_x_orig, coord_y_orig, metric_orig, vmin=vmin, vmax=vmax,
                         levels=pt.linspace(vmin, vmax, 100))
    fig.colorbar(tcf, shrink=0.75, label=r"$\sigma(" + str(field_) + ") / " + str(field_) + r"_{\infty}$",
                 format="{x:.2f}")
    if geometry_ is not None:
        for g in geometry_:
            ax.add_patch(Polygon(g, facecolor="white"))

    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_dir, f"metric_{field_}.png"), dpi=340)
    plt.close("all")


def plot_grid_and_metric(_faces, _vertices, coord_x_orig: pt.Tensor, coord_y_orig: pt.Tensor, metric_orig: pt.Tensor,
                         save_name: str, save_dir: str, geometry_: list = None) -> None:
    # we need to reconstruct the cells, because otherwise we can't plot it nicely (since we have an unsorted node
    # tensor, and we want to plot a non-uniform grid)
    fig, ax = plt.subplots(figsize=(6, 3))
    for i in range(_faces.shape[0]):
        ax.plot([_vertices[j, 0] for j in _faces[i, :]] + [_vertices[_faces[i, :][0], 0]],
                [_vertices[j, 1] for j in _faces[i, :]] + [_vertices[_faces[i, :][0], 1]],
                color="red", lw=0.25)

    ax.tricontourf(coord_x_orig, coord_y_orig, metric_orig, levels=100, alpha=0.9)
    if geometry_ is not None:
        for g in geometry_:
            ax.add_patch(Polygon(g, facecolor="white"))

    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_error_in_space(coord_x: pt.Tensor, coord_y: pt.Tensor, error_field: list, save_name: str, save_dir: str,
                        geometry_: list = None, field: str = "p", chord: float = 0.15) -> None:
    label = [r"$\mathrm{mean}(\Delta \mathbf{" + field + "}) / " + field + r"_{\infty}$",
             r"$\mathrm{std}(\Delta \mathbf{" + field + "}) / " + field + r"_{\infty}$"]

    # set global vmin/vmax for consistency
    vmin_global = 1e-8
    vmax_global = 1e-4

    # define levels, logarithmically spaced
    levels = np.logspace(np.log10(vmin_global), np.log10(vmax_global), 5)

    fig, ax = plt.subplots(ncols=2, sharey="row", figsize=(6, 2))
    for i in range(2):
        tcf = ax[i].tricontourf(coord_x / chord, coord_y / chord, error_field[i],
                                norm=colors.LogNorm(vmin=error_field[i].min(), vmax=error_field[i].max()),
                                levels=levels, extend="both")
        cbar = fig.colorbar(tcf, ax=ax[i], shrink=0.9, location="top", pad=0.1, ticks=levels)
        cbar.set_label(label[i], labelpad=10)

        if geometry_ is not None:
            [ax[i].add_patch(Polygon(g / chord, facecolor="white")) for g in geometry_]
        ax[i].set_aspect("equal")
    ax[0].set_ylabel("$z / c$")
    fig.supxlabel("$x / c$", y=0.025)
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_error_in_time(time_steps: list, errors: list, metrics: list, save_name: str, save_dir: str,
                       field_: str = "p") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for e, m in zip(errors, metrics):
        ax.plot(time_steps, e, label=rf"$\sigma(" + str(field_) + f") = {m}$")
    ax.set_xlabel(r"$snapshot$ $no. \#$")
    ax.set_ylabel(r"$\Delta L_2 / L_{2, orig}$")
    ax.set_xlim(min(time_steps), max(time_steps))
    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, ncol=4)
    fig.subplots_adjust(top=0.75)
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_total_error(errors: list, metrics: list, save_name: str, save_dir: str, field_: str = "p") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(metrics, errors)
    ax.set_xlabel(r"$\sigma(" + str(field_) + r") / \sigma(" + str(field_) + "_{orig})$")
    ax.set_ylabel(r"$\Delta L_2 / L_{2, orig}$")
    ax.set_xlim(min(metrics), max(metrics))
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.close("all")


if __name__ == "__main__":
    # path to the CFD data and path to directory the results should be saved to TODO: clean up script
    field_name = "Ma"
    area = "large"
    load_path = join("..", "run", "final_benchmarks", f"OAT15_{area}_new",
                     "results_with_geometry_refinement_no_dl_constraint")
    save_path_results = join("..", "run", "final_benchmarks", f"OAT15_{area}_new",
                             "plots_with_geometry_refinement_no_dl_constraint")

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join("..", "data", "2D", "OAT15", "vertices_and_masks.pt"), weights_only=False)
    cell_area_orig = xz[f"area_{area}"].unsqueeze(-1).sqrt()
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)

    # load the airfoil(s) as overlay for contourf plots
    geometry = [load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "oat15_airfoil_no_TE.stl"), dimensions="xz")]
    if area == "large":
        geometry.append(load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "naca_airfoil_no_TE.stl"),
                        dimensions="xz"))

    # load the pressure field of the original CFD data, small area around the leading airfoil
    if area == "large":
        # load_path_ma_large = join("/media", "janis", "Elements", "FOR_data", "oat15_aoa5_tandem_Johannes")
        load_path_ma_large = join("..", "data", "2D", "OAT15")
        orig_field = pt.load(join(load_path_ma_large, f"ma_{area}_every10.pt"), weights_only=False)
    else:
        orig_field = pt.load(join("..", "data", "2D", "OAT15", "p_small_every10.pt"), weights_only=False)

    # compute the metric
    metric = pt.std(orig_field, dim=1)

    # scale both fields with free stream quantities
    param_infinity = 75229.6 if field_name == "p" else 0.72

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # plot the metric of the original grid
    # plot_metric_original_grid(xz[:, 0], xz[:, 1], metric / param_infinity, save_path_results, geometry_=geometry,
    #                           field_=field_name)

    # weigh the field with its cell area
    orig_field *= cell_area_orig

    # compute the normalized L2-norms
    l2_total_orig = pt.linalg.norm(orig_field, ord=2)
    l2_time_orig = pt.linalg.norm(orig_field, ord=2, dim=0)

    # get all the generated grids in the directory
    files_hdf = sorted([f for f in glob(join(load_path, f"*.h5")) if "svd" not in f])
    variances = [re.findall(r"\d+.\d+", f)[0] for f in files_hdf]
    """
    files_hdf = [join(load_path, "OAT15_large_area_variance_0.25.h5"),
                 join(load_path, "OAT15_large_area_variance_0.50.h5"),
                 join(load_path, "OAT15_large_area_variance_0.75.h5")]
    variances = ["0.25", "0.50", "0.75"]
    # """

    # create empty lists for L2-errors vs. metrics
    error_time_vs_metric, error_total_vs_metric = [], []

    # create KNN for interpolating the generated field back to the original one
    knn = KNeighborsRegressor(n_neighbors=8 if xz.size(-1) == 2 else 26, weights="distance", n_jobs=8)

    for v, h in zip(variances, files_hdf):
        # load the generated grid and its values at the cell center from HDF5 file and construct the data matrix
        dataloader = Dataloader("/".join(h.split("/")[:-1]), h.split("/")[-1])
        cell_area_inter = dataloader.weights.sqrt()

        # plot the grid along with the metric from the original field as overlay (uncomment if wanted)
        # plot_grid_and_metric(dataloader.faces, dataloader.nodes, xz[:, 0], xz[:, 1], metric, f"grid_metric_{v}",
        #                      save_path_results, geometry_=geometry)

        # interpolate the fields back onto the original grid. In case the data is not fitting into the RAM all at once,
        # then the data has to be loaded and interpolated as it is done in the fit method of S^3's export routine
        knn.fit(dataloader.vertices, dataloader.load_snapshot(field_name))
        fields_fitted = pt.from_numpy(knn.predict(xz)) * cell_area_orig

        # compute the L2 error wrt time and normalize it with number of time steps
        error_time_vs_metric.append(pt.linalg.norm(fields_fitted - orig_field, ord=2, dim=0) / l2_time_orig)
        # compute the total L2-error and normalize it with l2 norm of the original field
        error_total_vs_metric.append(pt.linalg.norm(fields_fitted - orig_field, ord=2) / l2_total_orig)

        # compute the avg. & std. error of the metric for each cell (= wrt space) and scale it with the free stream
        # parameter
        error_space_vs_metric_avg = pt.mean((fields_fitted - orig_field).abs(), dim=1) / param_infinity
        error_space_vs_metric_std = pt.std((fields_fitted - orig_field).abs(), dim=1)/ param_infinity

        # plot the L2-error wrt each cell
        plot_error_in_space(xz[:, 0], xz[:, 1], [error_space_vs_metric_avg, error_space_vs_metric_std],
                            f"error_metric_{v}_{field_name}", save_path_results, geometry_=geometry,
                            field=field_name)

    # save the errors
    pt.save({"L2_error_time": error_time_vs_metric, "L2_error_total": error_total_vs_metric},
            join(load_path, "l2_errors.pt"))

    # plot L2 error vs. time (each variance is a different line)
    plot_error_in_time(range(orig_field.size(-1)), error_time_vs_metric, variances, f"error_vs_t_and_metric_{field_name}",
                       save_path_results, field_=field_name)

    # plot total L2-error vs. metric
    plot_total_error(error_total_vs_metric, list(map(float, variances)), f"total_error_vs_metric_{field_name}",
                     save_path_results, field_=field_name)

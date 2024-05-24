"""
    compute the error between the generated grid and the original grid wrt space, time and space & time
"""
import h5py
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from stl import mesh
from glob import glob
from os.path import join
from os import path, makedirs
from matplotlib.patches import Polygon
from sklearn.neighbors import KNeighborsRegressor


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


def plot_grid_and_metric(load_dir: str, coord_x_orig: pt.Tensor, coord_y_orig: pt.Tensor, metric_orig: pt.Tensor,
                         save_name: str, save_dir: str, geometry: list = None) -> None:
    # we need to reconstruct the cells, because otherwise we can't plot it nicely
    vn = h5py.File(load_dir, "r")

    # TODO: try meshgrid with node coord
    fig, ax = plt.subplots(figsize=(6, 3))
    for i in range(vn["grid"]["faces"].shape[0]):
        node_idx = vn["grid"]["faces"][i, :]
        ax.plot([vn["grid"]["vertices"][j, 0] for j in node_idx] + [vn["grid"]["vertices"][node_idx[0], 0]],
                [vn["grid"]["vertices"][j, 1] for j in node_idx] + [vn["grid"]["vertices"][node_idx[0], 1]],
                color="red", lw=0.5)

    ax.tricontourf(coord_x_orig, coord_y_orig, metric_orig, extend=True, alpha=0.75)
    if geometry is not None:
        for g in geometry:
            ax.add_patch(Polygon(g, facecolor="white"))

    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_error_in_space(coord_x: pt.Tensor, coord_y: pt.Tensor, error_field: pt.Tensor, save_name: str, save_dir: str,
                        geometry: list = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    # TODO: colorbars limits are not affected by vmin / vmax
    # tcf = ax.tricontourf(coord_x, coord_y, error_field, vmin=0.8, vmax=1.2)
    tcf = ax.tricontourf(coord_x, coord_y, error_field)
    fig.colorbar(tcf, label=r"$L_2 / L_{2, orig}$", shrink=0.5)
    if geometry is not None:
        for g in geometry:
            ax.add_patch(Polygon(g, facecolor="white"))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_error_in_time(time_steps: any, errors: list, metrics: list, save_name: str, save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for e, m in zip(errors, metrics):
        ax.plot(time_steps, e, label=rf"$\sigma(p) = {m}$")
    ax.set_xlabel(r"$snapshot$ $no. \#$")
    ax.set_ylabel(r"$L_2 / L_{2, orig}$")
    ax.set_xlim(min(time_steps), max(time_steps))
    fig.tight_layout()
    fig.legend(loc="upper right", framealpha=1.0, ncol=4)
    fig.subplots_adjust(top=0.75)
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_total_error(errors: list, metrics: list, save_name: str, save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(metrics, errors)
    ax.set_xlabel(r"$\sigma(p) / \sigma(p_{orig})$")
    ax.set_ylabel(r"$L_2 / L_{2, orig}$")
    ax.hlines(1, min(metrics), max(metrics), "red", ls="-.")
    ax.set_xlim(min(metrics), max(metrics))
    ax.set_ylim(0.999, 1.001)
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_dir, f"{save_name}.png"), dpi=340)
    plt.close("all")


if __name__ == "__main__":
    # path to the CFD data and path to directory the results should be saved to
    field_name = "p"
    area = "small"
    load_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                     f"results_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                             f"plots_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join("..", "data", "2D", "OAT15", "vertices_and_masks.pt"))
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)

    # load the airfoil(s) as overlay for contourf plots
    oat15 = load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "oat15_airfoil_no_TE.stl"), dimensions="xz")
    # naca = load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "naca_airfoil_no_TE.stl"), dimensions="xz")

    # load the pressure field of the original CFD data, small area around the leading airfoil
    orig_field = pt.load(join("..", "data", "2D", "OAT15", "p_small_every10.pt"))
    metric = pt.std(orig_field, dim=1)

    # compute the normalized L2-norms
    l2_total_orig = pt.linalg.norm(orig_field, ord=2)
    l2_time_orig = pt.linalg.norm(orig_field, ord=2, dim=0)
    l2_space_orig = pt.linalg.norm(orig_field, ord=2, dim=1)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # get all the generated grids in the directory
    files = sorted(glob(join(load_path, "OAT15_*.pt")), key=lambda x: float(x.split("_")[-1].split(".pt")[0]))
    files_hdf = sorted(glob(join(load_path, "*.h5")), key=lambda x: float(x.split("_")[-1].split(".h5")[0]))
    variances = [f.split("_")[-1].split(".pt")[0] for f in files]

    # create empty lists for L2-errors vs. metrics
    error_time_vs_metric, error_total_vs_metric = [], []

    # create KNN for interpolating the generated field back to the original one
    knn = KNeighborsRegressor(n_neighbors=8 if xz.size(-1) == 2 else 26, weights="distance")

    for v, grid, h in zip(variances, files, files_hdf):
        # plot the grid along with the metric from the original field as overlay (uncomment if wanted)
        # plot_grid_and_metric(h, xz[:, 0], xz[:, 1], metric, f"grid_metric_{v}", save_path_results, geometry=[oat15])

        # load the generated grid and its values at the cell center
        data = pt.load(grid)

        # interpolate the fields back onto the original grid. In case the data is not fitting into the RAM all at once,
        # then the data has to be loaded and interpolated as it is done in the fit method of S^3's export routine
        knn.fit(data["coordinates"], data[f"{field_name}"])
        fields_fitted = pt.from_numpy(knn.predict(xz))

        # compute the L2 error wrt time and normalize it with number of time steps
        error_time_vs_metric.append(pt.linalg.norm(fields_fitted, ord=2, dim=0) / l2_time_orig)

        # compute the total L2-error and normalize it with l2 norm of the original field
        error_total_vs_metric.append(pt.linalg.norm(fields_fitted, ord=2) / l2_total_orig)

        # compute the L2-error of the metric for each cell (= wrt space) and normalize it with the number of cells
        error_space_vs_metric = pt.linalg.norm(fields_fitted, ord=2, dim=1) / l2_space_orig

        # plot the L2-error wrt each cell
        plot_error_in_space(xz[:, 0], xz[:, 1], error_space_vs_metric, f"error_metric_{v}_{field_name}",
                            save_path_results, geometry=[oat15])

    # plot L2 error vs. time (each variance is a different line)
    plot_error_in_time(range(orig_field.size(-1)), error_time_vs_metric, variances, f"error_vs_t_and_metric_{field_name}",
                       save_path_results)

    # plot total L2-error vs. metric
    plot_total_error(error_total_vs_metric, list(map(float, variances)), f"total_error_vs_metric_{field_name}",
                     save_path_results)

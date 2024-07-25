"""
    Compute an SVD of the interpolated field (generated grid with S^3) and compare it to the original field from CFD

    this script is adopted from the flowtorch tutorial:

        https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/notebooks/svd_cylinder.html

    For more information, it is referred to the flowtorch documentation
"""

import torch as pt
import matplotlib.pyplot as plt

from os.path import join
from os import path, makedirs
from scipy.signal import welch
from typing import Tuple, Union
from flowtorch.analysis import SVD
from matplotlib.patches import Polygon

from s_cube.data import Dataloader
from post_processing.compute_error import load_airfoil_as_stl_file


def plot_singular_values(sv: list, _save_path: str, _save_name: str, legend: list, n_values: int = 100) -> None:
    # singular values original field
    s_var = [s * (100 / s.sum()) for s in sv]
    s_cum = [pt.cumsum(s, dim=0) for s in s_var]

    fig, ax = plt.subplots(nrows=2, figsize=(6, 4), sharex="col")
    for i, s in enumerate(zip(s_var, s_cum)):
        ax[0].plot(s[0][:n_values], label=legend[i])
        ax[1].plot(s[1])
    ax[1].set_xlabel(r"$no. \#$")
    ax[0].set_ylabel(r"$\%$ $\sqrt{\sigma_i^2}$")
    ax[1].set_ylabel(r"$cumulative$ $\%$")
    ax[1].set_xlim(0, n_values)
    ax[0].set_ylim(0, max([s[:n_values].max() for s in s_var]))
    # ax[1].set_ylim(0, max([s.max() for s in s_cum]))
    ax[0].legend(loc="upper right", framealpha=1.0, ncols=3)
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(_save_path, f"{_save_name}.png"), dpi=340)
    plt.close("all")


def plot_psd(V: list, dt: float, n_samples: int, _save_path: str, _save_name: str, legend: list, chord: float = 0.15,
             u_inf: float = 238.59, xlim: Union[Tuple, list] = (0, 1)) -> None:
    # adapted from @AndreWeiner
    fig, ax = plt.subplots(figsize=(6, 4))

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ls = 10 * ["-", "--", "-.", ":"]

    for m in range(len(color) - 1):
        for i, v in enumerate(V):
            freq, amp = welch(v[:, m], fs=1 / dt, nperseg=int(n_samples / 2), nfft=2 * n_samples)
            if i == 0:
                ax.plot(freq * chord / u_inf, amp, color=color[m], label=f"$mode$ ${m + 1}$", ls=ls[i])
            else:
                ax.plot(freq * chord / u_inf, amp, color=color[m], ls=ls[i])
    ax.set_xlabel(r"$Sr$")
    ax.set_ylabel("$PSD$")
    ax.set_xlim(xlim)
    ax.legend()
    fig.legend(legend, loc="upper center", ncols=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(join(_save_path, f"{_save_name}.png"), dpi=340)
    plt.close("all")


def plot_pod_modes(coord_original, coord_inter, U_orig, U_inter, _save_path: str, _save_name: str, n_modes: int = 4,
                   _geometry: list = None) -> None:
    vmin = min(U_orig[:, :n_modes].min().item(), U_inter[:, :n_modes].min().item())
    vmax = max(U_orig[:, :n_modes].max().item(), U_inter[:, :n_modes].max().item())
    levels = pt.linspace(vmin, vmax, 100)

    fig, ax = plt.subplots(n_modes, 2, sharex="all", sharey="all")
    for row in range(n_modes):
        ax[row][0].tricontourf(coord_original[:, 0], coord_original[:, 1], U_orig[:, row], vmin=vmin, vmax=vmax,
                               levels=levels, cmap="seismic", extend="both")
        if row == 2:
            ax[row][1].tricontourf(coord_inter[:, 0], coord_inter[:, 1], U_inter[:, row], vmin=vmin, vmax=vmax,
                                   levels=levels, cmap="seismic", extend="both")
        else:
            ax[row][1].tricontourf(coord_inter[:, 0], coord_inter[:, 1], U_inter[:, row] * -1, vmin=vmin, vmax=vmax,
                                   levels=levels, cmap="seismic", extend="both")
        if _geometry is not None:
            for g in _geometry:
                ax[row][0].add_patch(Polygon(g, facecolor="black"))
                ax[row][1].add_patch(Polygon(g, facecolor="black"))
    ax[0][0].set_title("$original$")
    ax[0][1].set_title("$interpolated$")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(_save_path, f"{_save_name}.png"), dpi=340)
    plt.close("all")


def plot_mode_coefficients(write_times, V: list, _save_path: str, _save_name: str, legend: list,
                           n_modes: int = 4) -> None:
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ls = 10 * ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(nrows=n_modes, figsize=(6, 4), sharex="all", sharey="all")
    for row in range(n_modes):
        for i, v in enumerate(V):
            if row == 0:
                ax[row].plot(write_times, v[:, row], color=color[row], label=legend[i], ls=ls[i])
            else:
                ax[row].plot(write_times, v[:, row], color=color[row], ls=ls[i])
        ax[row].text(write_times.max() + 1e-3, 0, f"$mode$ ${row}$", va="center")
        ax[row].set_xlim(write_times.min(), write_times.max())
    fig.legend(loc="upper center", framealpha=1.0, ncols=3)
    ax[-1].set_xlabel("$t$ $[s]$")
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(join(_save_path, f"{_save_name}.png"),
                dpi=340)
    plt.close("all")


if __name__ == "__main__":
    # which fields and settings to use
    field_name = "p"
    area = "small"
    metric = "0.75"

    # path to the HDF5 file
    load_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                     f"results_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")

    # path to the directory to which the plots should be saved to
    file_name = f"OAT15_{area}_area_variance_{metric}.h5"
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                             f"plots_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")

    # load the field of the original CFD data
    # orig_field = pt.load(join("/media", "janis", "Elements", "FOR_data", "oat15_aoa5_tandem_Johannes",
    #                           f"ma_{area}_every10.pt"))
    orig_field = pt.load(join("..", "data", "2D", "OAT15", "p_small_every10.pt"))

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join("..", "data", "2D", "OAT15", "vertices_and_masks.pt"))
    cell_area_orig = xz[f"area_{area}"].unsqueeze(-1).sqrt()
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)

    # field from generated grid
    dataloader = Dataloader(load_path, file_name)
    interpolated_field = dataloader.load_snapshot(field_name)

    # load the airfoil(s) as overlay for contourf plots
    geometry = [load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "oat15_airfoil_no_TE.stl"), dimensions="xz")]
    # geometry.append(load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "naca_airfoil_no_TE.stl"),
    #                 dimensions="xz"))

    # load the corresponding write times and stack the coordinates
    times = pt.load(join("..", "data", "2D", "OAT15", "oat15_tandem_times.pt"))[::10]

    # compute the cell areas (2D) / volumes (3D) for the interpolated field
    cell_area_inter = dataloader.weights.sqrt().unsqueeze(-1)

    # multiply by the sqrt of the cell areas to weight their contribution
    orig_field *= cell_area_orig
    interpolated_field *= cell_area_inter

    # make SVD of original flow field data
    orig_field -= pt.mean(orig_field, dim=1).unsqueeze(-1)
    svd_orig = SVD(orig_field, rank=orig_field.size(-1))

    # make SVD of interpolated flow field data
    interpolated_field -= pt.mean(interpolated_field, dim=1).unsqueeze(-1)
    svd_inter = SVD(interpolated_field, rank=orig_field.size(-1))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # plot frequency spectrum
    plot_psd([svd_orig.V.numpy(), svd_inter.V.numpy()], (times[1] - times[0]).item(), len(times),
             save_path_results, f"comparison_psd_metric_{metric}_weighted", legend=["$original$", "$interpolated$"])

    # plot singular values
    plot_singular_values([svd_orig.s, svd_inter.s], save_path_results,
                         f"comparison_singular_values_metric_{metric}_weighted",
                         legend=["$original$", "$interpolated$"])

    # plot the first N POD modes (left singular vectors)
    plot_pod_modes(xz, dataloader.vertices, svd_orig.U / cell_area_orig, svd_inter.U / cell_area_inter,
                   save_path_results, f"comparison_pod_modes_metric_{metric}_weighted", _geometry=geometry)

    # plot POD mode coefficients (right singular vectors)
    plot_mode_coefficients(times, [svd_orig.V, svd_inter.V], save_path_results,
                           f"comparison_pod_mode_coefficients_metric_{metric}_weighted",
                           legend=["$original$", "$interpolated$"])

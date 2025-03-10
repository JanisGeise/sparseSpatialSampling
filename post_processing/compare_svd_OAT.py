"""
    Compute an SVD of the interpolated field (generated grid with S^3) and compare it to the original field from CFD
    for the OAT airfoil

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

from sparseSpatialSampling.data import Dataloader
from post_processing.compute_error_OAT import load_airfoil_as_stl_file

# use latex fonts
plt.rcParams.update({"text.usetex": True})


def plot_singular_values(sv: list, _save_path: str, _save_name: str, legend: list, n_values: int = 100) -> None:
    # singular values original field
    s_var = [s * (100 / s.sum()) for s in sv]
    s_cum = [pt.cumsum(s, dim=0) for s in s_var]

    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ls = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(6, 2))
    ax2 = ax.twinx()
    for i, s in enumerate(zip(sv, s_cum)):
        ax.plot(s[0][:n_values+1], label=legend[i], color=color[i], ls=ls[i])
        ax2.plot(s[1], color=color[i], ls=ls[i])
    ax.set_xlabel(r"$no. \#$")
    # ax.set_ylabel(r"$\sigma_{i, rel}$   $[\%]$")
    ax.set_ylabel(r"$\sigma_{i}$")
    ax.set_ylim(0, max([s[:n_values].max() for s in sv]))
    ax2.set_ylabel(r"$cumulative$ $[\%]$")
    ax2.set_xlim(0, n_values)
    # ax[1].set_ylim(0, max([s.max() for s in s_cum]))
    fig.legend(loc="upper center", framealpha=1.0, ncols=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.83)
    plt.savefig(join(_save_path, f"{_save_name}.png"), dpi=340)
    plt.close("all")


def plot_psd(V: list, dt: float, n_samples: int, _save_path: str, _save_name: str, legend: list, chord: float = 0.15,
             u_inf: float = 238.59, xlim: Union[Tuple, list] = (0, 1), n_modes: int = 4) -> None:
    # adapted from @AndreWeiner
    fig, ax = plt.subplots(figsize=(6, 3))

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ls = 10 * ["-", "--", "-.", ":"]

    for m in range(n_modes):
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
    ax.legend(loc="upper right", ncols=3)
    fig.legend(legend, loc="upper center", ncols=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    plt.savefig(join(_save_path, f"{_save_name}.png"), dpi=340)
    plt.close("all")


def plot_pod_modes(coord_original, coord_inter, U_orig, U_inter, _save_path: str, _save_name: str, n_modes: int = 4,
                   _geometry: list = None, chord: float = 0.15) -> None:
    vmax = max(U_orig[:, :n_modes].max().item(), U_inter[:, :n_modes].max().item())
    vmin = -vmax
    levels = pt.linspace(vmin, vmax, 100)

    fig, ax = plt.subplots(n_modes, 2, sharex="all", sharey="all")
    for row in range(n_modes):
        ax[row][0].tricontourf(coord_original[:, 0] / chord, coord_original[:, 1] / chord, U_orig[:, row], vmin=vmin,
                               vmax=vmax, levels=levels, cmap="seismic", extend="both")
        # flip sign of modes so it is consistent
        if 0 < row < n_modes-2:
            ax[row][1].tricontourf(coord_inter[:, 0] / chord, coord_inter[:, 1] / chord, U_inter[:, row] * -1,
                                   vmin=vmin, vmax=vmax, levels=levels, cmap="seismic", extend="both")
        else:
            ax[row][1].tricontourf(coord_inter[:, 0] / chord, coord_inter[:, 1] / chord, U_inter[:, row], vmin=vmin,
                                   vmax=vmax, levels=levels, cmap="seismic", extend="both")
        if _geometry is not None:
            for g in _geometry:
                ax[row][0].add_patch(Polygon(g / chord, facecolor="black"))
                ax[row][1].add_patch(Polygon(g / chord, facecolor="black"))
        # ax[row-1][0].set_ylabel("$z / c$")
    # ax[0][0].set_title("$original$")
    # ax[0][1].set_title("$interpolated$")
    fig.supylabel("$z / c$")
    ax[-1][0].set_xlabel("$x / c$")
    ax[-1][1].set_xlabel("$x / c$")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(_save_path, f"{_save_name}.png"), dpi=340)
    plt.close("all")


def plot_mode_coefficients(write_times: pt.Tensor, V: list, _save_path: str, _save_name: str, legend: list,
                           n_modes: int = 4, chord: float = 0.15, u_inf: float = 238.59) -> None:
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ls = 10 * ["-", "--", "-.", ":"]

    # reverse sign, so it is consistent with V[0] = original
    V[1][:, [1, 2, 3]] *= -1
    V[2][:, [0, 1, 2, 3]] *= -1
    V[3][:, [0, 4, 5]] *= -1

    # non-dimensionalize the time steps
    write_times = write_times * (u_inf / chord)

    fig, ax = plt.subplots(nrows=n_modes, figsize=(6, 4), sharex="all", sharey="all")
    for row in range(n_modes):
        for i, v in enumerate(V):
            if row == 0:
                ax[row].plot(write_times, v[:, row], color=color[i], label=legend[i], ls=ls[i])
            else:
                ax[row].plot(write_times, v[:, row], color=color[i], ls=ls[i])
        ax[row].set_xlim(write_times.min(), write_times.max())
    fig.legend(loc="upper center", framealpha=1.0, ncols=4)
    ax[-1].set_xlabel(r"$\tau$")
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(join(_save_path, f"{_save_name}.png"),
                dpi=340)
    plt.close("all")


if __name__ == "__main__":
    # which fields and settings to use
    field_name = "Ma"
    area = "large"
    metric = ["0.25", "0.50", "0.75"]

    # path to the HDF5 file
    load_path = join("..", "run", "final_benchmarks", f"OAT15_{area}_new",
                     "results_with_geometry_refinement_no_dl_constraint")
    file_name = [f"OAT15_{area}_area_variance_{m}.h5" for m in metric]

    # path to the directory to which the plots should be saved to
    save_path_results = join("..", "run", "final_benchmarks", f"OAT15_{area}_new",
                             "plots_with_geometry_refinement_no_dl_constraint")

    # the actual metric may differ from the filename
    legend = ["$original$"] + [r"$\mathcal{M} = " + f"{m}$" for m in ["0.27", "0.50", "0.75"]]

    # load the field of the original CFD data
    if area == "large":
        orig_field = pt.load(join("..", "data", "2D", "OAT15", f"ma_{area}_every10.pt"), weights_only=False)
    else:
        orig_field = pt.load(join("..", "data", "2D", "OAT15", "p_small_every10.pt"), weights_only=False)

    # load the airfoil(s) as overlay for contourf plots
    geometry = [load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "oat15_airfoil_no_TE.stl"), dimensions="xz")]

    if area == "large":
        geometry.append(load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "naca_airfoil_no_TE.stl"),
                        dimensions="xz"))

    # load the corresponding write times and stack the coordinates
    times = pt.load(join("..", "data", "2D", "OAT15", "oat15_tandem_times.pt"), weights_only=False)[::10]

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join("..", "data", "2D", "OAT15", "vertices_and_masks.pt"), weights_only=False)
    cell_area_orig = xz[f"area_{area}"].unsqueeze(-1).sqrt()
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)

    # field from generated grid
    dataloader = [Dataloader(load_path, f) for f in file_name]
    interpolated_field = [d.load_snapshot(field_name) for d in dataloader]

    # compute the cell areas (2D) / volumes (3D) for the interpolated field
    cell_area_inter = [d.weights.sqrt().unsqueeze(-1) for d in dataloader]

    # multiply by the sqrt of the cell areas to weight their contribution
    orig_field *= cell_area_orig
    interpolated_field = [f * cell_area_inter[i] for i, f in enumerate(interpolated_field)]

    # make SVD of original flow field data
    orig_field -= pt.mean(orig_field, dim=1).unsqueeze(-1)
    svd_orig = SVD(orig_field, rank=orig_field.size(-1))

    # make SVD of interpolated flow field data
    interpolated_field = [i - pt.mean(i, dim=1).unsqueeze(-1) for i in interpolated_field]
    svd_inter = [SVD(i, rank=orig_field.size(-1)) for i in interpolated_field]

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # plot frequency spectrum
    plot_psd([svd_orig.V.numpy()]+[sv.V.numpy() for sv in svd_inter], (times[1] - times[0]).item(), len(times),
             save_path_results, f"comparison_psd_metric_{metric[0]}_{metric[-1]}",
             legend=legend, xlim=(0, 2.6), n_modes=6)

    # plot singular values
    plot_singular_values([svd_orig.s]+[sv.s for sv in svd_inter], save_path_results,
                         f"comparison_singular_values_metric_{metric[0]}_{metric[-1]}", legend=legend)

    # plot POD mode coefficients (right singular vectors)
    plot_mode_coefficients(times, [svd_orig.V]+[sv.V for sv in svd_inter], save_path_results,
                           f"comparison_pod_mode_coefficients_metric_{metric[0]}_{metric[-1]}",
                           legend=legend, n_modes=6)

    # plot the first N POD modes (left singular vectors) for a specific case
    no = 0
    plot_pod_modes(xz, dataloader[no].vertices, svd_orig.U / cell_area_orig, svd_inter[no].U / cell_area_inter[no],
                   save_path_results, f"comparison_pod_modes_metric_{metric[no]}", _geometry=geometry,
                   n_modes=6)

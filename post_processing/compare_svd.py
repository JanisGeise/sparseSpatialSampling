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
from flowtorch.analysis import SVD
from matplotlib.patches import Polygon
# from scipy.signal import welch

from post_processing.compute_error import load_airfoil_as_stl_file


if __name__ == "__main__":
    # path to the CFD data and path to directory the results should be saved to
    field_name = "p"
    area = "small"
    load_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                     f"results_metric_based_on_{field_name}_stl_{area}_with_dl_constraint")
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                             f"plots_metric_based_on_{field_name}_stl_{area}_with_dl_constraint")

    # load the pressure field of the original CFD data, small area around the leading airfoil
    # orig_field = pt.load(join("/media", "janis", "Elements", "FOR_data", "oat15_aoa5_tandem_Johannes",
    #                           f"ma_{area}_every10.pt"))
    orig_field = pt.load(join("..", "data", "2D", "OAT15", "p_small_every10.pt"))

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join("..", "data", "2D", "OAT15", "vertices_and_masks.pt"))
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)

    # field from generated grid
    interpolated_field = pt.load(join(load_path, f"OAT15_{area}_area_variance_0.95.pt"))

    # load the airfoil(s) as overlay for contourf plots
    geometry = [load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "oat15_airfoil_no_TE.stl"), dimensions="xz")]
    # geometry.append(load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "naca_airfoil_no_TE.stl"),
    #                 dimensions="xz"))

    # load the corresponding write times and stack the coordinates
    times = pt.load(join("..", "data", "2D", "OAT15", "oat15_tandem_times.pt"))[::10]
    dt = (times[1] - times[0]).item()

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # make SVD of original flow field data and compute singular values
    orig_field -= pt.mean(orig_field, dim=1).unsqueeze(-1)
    svd_orig = SVD(orig_field, rank=orig_field.size(-1))
    s_var_orig = svd_orig.s * (100 / svd_orig.s.sum())
    s_cum_orig = pt.cumsum(s_var_orig, dim=0)

    # make SVD of interpolated flow field data and compute singular values
    interpolated_field[field_name] -= pt.mean(interpolated_field[field_name], dim=1).unsqueeze(-1)
    svd_inter = SVD(interpolated_field[field_name], rank=orig_field.size(-1))
    s_var_inter = svd_inter.s * (100 / svd_inter.s.sum())
    s_cum_inter = pt.cumsum(s_var_inter, dim=0)

    # plot singular values
    fig, ax = plt.subplots(nrows=2, figsize=(6, 4), sharex="col")
    ax[0].plot(s_var_orig, label="$original$")
    ax[0].plot(s_var_inter, label="$interpolated$")
    ax[1].plot(s_cum_orig)
    ax[1].plot(s_cum_inter)
    ax[1].set_xlabel(r"$no. \#$")
    ax[0].set_ylabel(r"$\%$ $\sqrt{\sigma_i^2}$")
    ax[1].set_ylabel(r"$cumulative$ $\%$")
    ax[0].set_xlim([0, s_var_orig.size(0)])
    ax[0].set_ylim([0, max([s_var_orig.max(), s_var_inter.max()])])
    ax[1].set_ylim([0, max([s_cum_orig.max(), s_cum_inter.max()])])
    ax[0].legend(loc="upper right", framealpha=1.0, ncols=2)
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_path_results, "comparison_singular_values.png"), dpi=340)
    plt.close("all")

    # plot the first 4 POD modes (left singular vectors)
    nrows = 4
    vmin = min(svd_orig.U[:, :nrows].min().item(), svd_inter.U[:, :nrows].min().item())
    vmax = max(svd_orig.U[:, :nrows].max().item(), svd_inter.U[:, :nrows].max().item())
    levels = pt.linspace(vmin, vmax, 100)

    fig, ax = plt.subplots(nrows, 2, sharex="all", sharey="all")
    for row in range(nrows):
        ax[row][0].tricontourf(xz[:, 0], xz[:, 1], svd_orig.U[:, row], vmin=vmin, vmax=vmax, levels=levels,
                               cmap="seismic")
        ax[row][1].tricontourf(interpolated_field["coordinates"][:, 0], interpolated_field["coordinates"][:, 1],
                               svd_inter.U[:, row], vmin=vmin, vmax=vmax, levels=levels, cmap="seismic")
        for g in geometry:
            ax[row][0].add_patch(Polygon(g, facecolor="black"))
            ax[row][1].add_patch(Polygon(g, facecolor="black"))
    ax[0][0].set_title("$original$")
    ax[0][1].set_title("$interpolated$")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_path_results, f"comparison_pod_modes.png"), dpi=340)
    plt.close("all")

    # plot POD mode coefficients (right singular vectors)
    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    fig, ax = plt.subplots(nrows=4, figsize=(6, 4), sharex="all", sharey="all")
    for row in range(4):
        if row == 0:
            ax[row].plot(times, svd_orig.V[:, row] * svd_orig.s[row], color=color[row], label="$original$")
            ax[row].plot(times, svd_inter.V[:, row] * svd_inter.s[row], color=color[row], ls="--",
                         label="$interpolated$")
        else:
            ax[row].plot(times, svd_orig.V[:, row] * svd_orig.s[row], color=color[row])
            ax[row].plot(times, svd_inter.V[:, row] * svd_inter.s[row], color=color[row], ls="--")
        ax[row].text(times.max() + 1e-3, 0, f"$mode$ ${row}$", va="center")
        ax[row].set_xlim(times.min(), times.max())
    fig.legend(loc="upper center", framealpha=1.0, ncols=2)
    ax[-1].set_xlabel("$t$ $[s]$")
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(join(save_path_results, f"comparison_pod_mode_coefficients.png"), dpi=340)
    plt.close("all")

    # plot frequency spectrum
    # freq_orig, amp_orig = welch(svd_orig.V.numpy(), fs=1/dt)
    # plt.plot(freq_orig, amp_orig[0, :])
    # plt.show()

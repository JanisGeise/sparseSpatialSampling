"""
    Compute a DMD of the interpolated field (generated grid with S^3) and compare it to the original field from CFD

    this script is adopted from the flowtorch tutorial:

        https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/notebooks/dmd_cylinder.html

    For more information, it is referred to the flowtorch documentation
"""
import torch as pt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pydmd import DMD
from os.path import join
from os import path, makedirs

from post_processing.compute_error import load_airfoil_as_stl_file, construct_data_matrix_from_hdf5

if __name__ == "__main__":
    # which fields and settings to use
    field_name = "p"
    area = "small"
    metric = "0.95"

    # parameter for scaling the reconstructed fields, e.g., pressure / Mach number of free stream
    param_infinity = 75229.6
    # param_infinity = 0.72

    # path to the HDF5 file
    load_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                     f"results_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")
    file_name = f"OAT15_{area}_area_variance_{metric}_{field_name}.h5"

    # path to the directory to which the plots should be saved to
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                             f"plots_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")

    # load the field of the original CFD data
    # orig_field = pt.load(join("/media", "janis", "Elements", "FOR_data", "oat15_aoa5_tandem_Johannes",
    #                           f"ma_{area}_every10.pt"))
    orig_field = pt.load(join("..", "data", "2D", "OAT15", "p_small_every10.pt"))

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join("..", "data", "2D", "OAT15", "vertices_and_masks.pt"))
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)

    # field from generated grid
    interpolated_field = construct_data_matrix_from_hdf5(join(load_path, file_name), field_name)

    # scale both fields with free stream quantities (for reconstructed fields)
    orig_field /= param_infinity
    interpolated_field[field_name] /= param_infinity

    # load the airfoil(s) as overlay for contourf plots
    geometry = [load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "oat15_airfoil_no_TE.stl"), dimensions="xz")]
    # geometry.append(load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "naca_airfoil_no_TE.stl"),
    #                 dimensions="xz"))

    # load the corresponding write times and stack the coordinates
    times = pt.load(join("..", "data", "2D", "OAT15", "oat15_tandem_times.pt"))[::10]

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # perform DMD
    rank = 50
    dmd_orig = DMD(svd_rank=rank, sorted_eigs="abs")
    dmd_inter = DMD(svd_rank=rank, sorted_eigs="abs")
    dmd_orig.fit(orig_field)
    dmd_inter.fit(interpolated_field[field_name])

    # plot original, reconstructed data and their error
    t_i = 100
    levels = 100
    # vmin, vmax = orig_field[:, t_i].min(), orig_field[:, t_i].max()
    fig, ax = plt.subplots(3, 2, sharex="all", sharey="all")
    tcf1 = ax[0][0].tricontourf(xz[:, 0], xz[:, 1], orig_field[:, t_i], levels=levels)
    tcf2 = ax[1][0].tricontourf(xz[:, 0], xz[:, 1], dmd_orig.reconstructed_data[:, t_i].real, levels=levels)
    tcf3 = ax[2][0].tricontourf(xz[:, 0], xz[:, 1], orig_field[:, t_i] - dmd_orig.reconstructed_data[:, t_i].real,
                                levels=levels)
    ax[0][1].tricontourf(interpolated_field["coordinates"][:, 0], interpolated_field["coordinates"][:, 1],
                         interpolated_field[field_name][:, t_i], levels=levels)
    ax[1][1].tricontourf(interpolated_field["coordinates"][:, 0], interpolated_field["coordinates"][:, 1],
                         dmd_inter.reconstructed_data[:, t_i].real, levels=levels)
    ax[2][1].tricontourf(interpolated_field["coordinates"][:, 0], interpolated_field["coordinates"][:, 1],
                         interpolated_field[field_name][:, t_i] - dmd_inter.reconstructed_data[:, t_i].real,
                         levels=levels)
    [fig.colorbar(tcf1, ax=ax[i][1], shrink=0.75, label=r"$p / p_{\infty}$") for i in range(3)]
    ax[0][0].set_ylabel("$original$")
    ax[1][0].set_ylabel("$reconstructed$")
    ax[2][0].set_ylabel("$difference$")
    for row in range(3):
        for g in geometry:
            ax[row][0].add_patch(Polygon(g, facecolor="white"))
            ax[row][1].add_patch(Polygon(g, facecolor="white"))
    ax[0][0].set_title("$original$")
    ax[0][1].set_title("$interpolated$")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_path_results, f"comparison_reconstructed_fields_dmd_metric_{metric}_rank_{rank}_t_" +
                     "{:.4f}.png".format(times[t_i])), dpi=340)
    plt.close("all")

    # plot the first few modes
    nrows = 4
    vmin = min(dmd_orig.modes.real[:, :nrows].min().item(), dmd_inter.modes.real[:, :nrows].min().item())
    vmax = max(dmd_orig.modes.real[:, :nrows].max().item(), dmd_inter.modes.real[:, :nrows].max().item())
    levels = pt.linspace(vmin, vmax, 100)

    fig, ax = plt.subplots(nrows, 2, sharex="all", sharey="all")
    for row in range(nrows):
        ax[row][0].tricontourf(xz[:, 0], xz[:, 1], dmd_orig.modes.real[:, row], vmin=vmin, vmax=vmax, levels=levels,
                               cmap="seismic")
        ax[row][1].tricontourf(interpolated_field["coordinates"][:, 0], interpolated_field["coordinates"][:, 1],
                               dmd_inter.modes.real[:, row], vmin=vmin, vmax=vmax, levels=levels, cmap="seismic")
        for g in geometry:
            ax[row][0].add_patch(Polygon(g, facecolor="black"))
            ax[row][1].add_patch(Polygon(g, facecolor="black"))
    ax[0][0].set_title("$original$")
    ax[0][1].set_title("$interpolated$")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_path_results, f"comparison_dmd_modes_metric_{metric}_rank_{rank}.png"), dpi=340)
    plt.close("all")

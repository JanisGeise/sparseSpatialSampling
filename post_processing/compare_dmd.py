"""
    Compute a DMD of the interpolated field (generated grid with S^3) and compare it to the original field from CFD

    this script is adopted from the flowtorch tutorial:

        https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/notebooks/dmd_cylinder.html

    For more information, it is referred to the flowtorch documentation
"""
import torch as pt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from os.path import join
from os import path, makedirs
from flowtorch.analysis import DMD

from s_cube.data import Dataloader
from post_processing.compute_error import load_airfoil_as_stl_file


def plot_dmd_modes(coord_orig: pt.tensor, coord_inter: pt.Tensor, modes_orig, modes_inter, _save_path: str,
                   _save_name: str, n_modes: int = 4, _geometry: list = None) -> None:
    vmin = min(modes_orig[:, :n_modes].min().item(), modes_inter[:, :n_modes].min().item())
    vmax = max(modes_orig[:, :n_modes].max().item(), modes_inter[:, :n_modes].max().item())
    levels = pt.linspace(vmin, vmax, 100)

    fig, ax = plt.subplots(n_modes, 2, sharex="all", sharey="all")
    for row in range(n_modes):
        ax[row][0].tricontourf(coord_orig[:, 0], coord_orig[:, 1], modes_orig[:, row], vmin=vmin, vmax=vmax,
                               levels=levels, cmap="seismic")
        ax[row][1].tricontourf(coord_inter[:, 0], coord_inter[:, 1], modes_inter[:, row], vmin=vmin, vmax=vmax,
                               levels=levels, cmap="seismic")
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


def plot_reconstructed_fields(coord_orig: pt.tensor, coord_inter: pt.Tensor, field_orig, field_inter, field_dmd_orig,
                              field_dmd_inter, _save_path: str, _save_name: str, t_i: int = 100,
                              levels: int = 100, label: str = "", _geometry: list = None) -> None:

    fig, ax = plt.subplots(3, 2, sharex="all", sharey="all")
    tcf1 = ax[0][0].tricontourf(coord_orig[:, 0], coord_orig[:, 1], field_orig[:, t_i], levels=levels)
    ax[1][0].tricontourf(coord_orig[:, 0], coord_orig[:, 1], field_dmd_orig[:, t_i], levels=levels)
    ax[2][0].tricontourf(coord_orig[:, 0], coord_orig[:, 1], field_orig[:, t_i] - field_dmd_orig[:, t_i], levels=levels)
    ax[0][1].tricontourf(coord_inter[:, 0], coord_inter[:, 1], field_inter[:, t_i], levels=levels)
    ax[1][1].tricontourf(coord_inter[:, 0], coord_inter[:, 1], field_dmd_inter[:, t_i], levels=levels)
    ax[2][1].tricontourf(coord_inter[:, 0], coord_inter[:, 1], field_inter[:, t_i] - field_dmd_inter[:, t_i],
                         levels=levels)
    [fig.colorbar(tcf1, ax=ax[i][1], shrink=0.75, label=label) for i in range(3)]
    ax[0][0].set_ylabel("$original$")
    ax[1][0].set_ylabel("$reconstructed$")
    ax[2][0].set_ylabel("$difference$")
    for row in range(3):
        if _geometry is not None:
            for g in _geometry:
                ax[row][0].add_patch(Polygon(g, facecolor="white"))
                ax[row][1].add_patch(Polygon(g, facecolor="white"))
    ax[0][0].set_title("$original$")
    ax[0][1].set_title("$interpolated$")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(_save_path, f"{_save_name}.png"), dpi=340)
    plt.close("all")


if __name__ == "__main__":
    # which fields and settings to use
    field_name = "p"
    area = "small"
    metric = 0.95

    # parameter for scaling the reconstructed fields, e.g., pressure / Mach number of free stream
    param_infinity = 75229.6
    # param_infinity = 0.72

    # path to the HDF5 file
    load_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                     f"results_metric_based_on_{field_name}_stl_{area}_no_dl_constraint_TEST")
    file_name = f"OAT15_{area}_area_variance_{metric}.h5"

    # path to the directory to which the plots should be saved to
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

    # field from generated grid & compute the cell areas (2D) / volumes (3D) for the interpolated field
    dataloader = Dataloader(load_path, file_name)
    interpolated_field = dataloader.load_snapshot(field_name)
    cell_area_inter = dataloader.weights.sqrt().unsqueeze(-1)

    # scale with cell area
    orig_field *= cell_area_orig
    interpolated_field *= cell_area_inter

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
    dmd_orig = DMD(orig_field, dt=times[1] - times[0], rank=rank)
    dmd_inter = DMD(interpolated_field, dt=times[1] - times[0], rank=rank)

    # sort the modes based on their integral contribution (every 2nd because we have complex-conjugated pairs and start
    # at 1 to ignore the mean)
    modes_orig_sorted = dmd_orig.modes[:, pt.argsort(dmd_orig.integral_contribution, descending=True)].real[:, 1::2]
    modes_inter_sorted = dmd_inter.modes[:, pt.argsort(dmd_inter.integral_contribution, descending=True)].real[:, 1::2]

    # plot the first few modes
    plot_dmd_modes(xz, dataloader.vertices, modes_orig_sorted, modes_inter_sorted, save_path_results,
                   f"comparison_dmd_modes_metric_{metric}_rank_{rank}_TEST", _geometry=geometry)

    # plot original, reconstructed data and their error and scale with free stream quantity
    orig_field /= param_infinity
    interpolated_field /= param_infinity

    # plot the original and reconstructed fields along with the reconstruction error
    plot_reconstructed_fields(xz, dataloader.vertices, orig_field / cell_area_orig,
                              interpolated_field / cell_area_inter,
                              dmd_orig.reconstruction.real / cell_area_orig,
                              dmd_inter.reconstruction.real / cell_area_inter,
                              save_path_results, f"comparison_reconstructed_fields_dmd_metric_{metric}_rank_{rank}_t_"
                              + "{:.4f}_TEST.png".format(times[100]), t_i=100, label=r"$p / p_{\infty}$", _geometry=geometry)

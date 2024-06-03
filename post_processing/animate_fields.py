"""
    create an animation of the original flow field and the interpolated one
"""
import torch as pt
import matplotlib.pyplot as plt

from os.path import join
from os import path, makedirs
from matplotlib.patches import Polygon
from matplotlib.animation import FFMpegWriter, FuncAnimation

from compute_error import load_airfoil_as_stl_file, construct_data_matrix_from_hdf5

# increase resolution
plt.rcParams['figure.dpi'] = 640


def compare_fields(orig_coord: pt.Tensor, interpolated_coord: pt.Tensor, orig_field: pt.Tensor,
                   interpolated_field: pt.Tensor, n_frames: int, geometry_: list = None):

    fig, ax = plt.subplots(nrows=2, figsize=(6, 6), sharex="col")
    for a in range(2):
        ax[a].set_ylabel("$z$")
        ax[a].set_aspect("equal")
    ax[1].set_xlabel("$x$")
    fig.text(0.8, 0.9, "$original$", ha="center", va="center", backgroundcolor="white")
    fig.text(0.8, 0.45, "$interpolated$", ha="center", va="center", backgroundcolor="white")
    fig.tight_layout()
    fig.subplots_adjust()
    vmin = min(orig_field.min().item(), interpolated_field.min().item())
    vmax = max(orig_field.max().item(), interpolated_field.max().item())
    levels = pt.linspace(vmin, vmax, 100)

    # animate function adopted from @AndreWeiner
    def animate(i):
        print("\r", f"Creating frame {i+1:03d} / {n_frames}", end="")
        ax[0].clear()
        ax[1].clear()
        tcf1 = ax[0].tricontourf(orig_coord[:, 0], orig_coord[:, 1], orig_field[:, i], levels=levels, vmin=vmin,
                                 vmax=vmax)
        tcf2 = ax[1].tricontourf(interpolated_coord[:, 0], interpolated_coord[:, 1], interpolated_field[:, i],
                                 levels=levels, vmin=vmin, vmax=vmax)

        # add colorbar
        # fig.colorbar(tcf1, ax=ax[0], label=f"${name}$" + "$_{orig}$", shrink=0.5, format="{x:.2f}")
        # fig.colorbar(tcf2, ax=ax[1], label=f"${name}$" + "$_{interpolated}$", shrink=0.5, format="{x:.2f}")

        if geometry_ is not None:
            for g in geometry_:
                ax[0].add_patch(Polygon(g, facecolor="white"))
                ax[1].add_patch(Polygon(g, facecolor="white"))

        ax[0].set_title("$t = {:.6f}s$".format(times[i]))
        for j in range(2):
            ax[j].set_ylabel("$z$")
            ax[j].set_aspect("equal")
        ax[1].set_xlabel("$x$")
        fig.tight_layout()
        fig.subplots_adjust()

    return FuncAnimation(fig, animate, frames=n_frames, repeat=True)


if __name__ == "__main__":
    # path to the CFD data and path to directory the results should be saved to
    field_name = "p"
    area = "small"
    file_name = f"OAT15_{area}_area_variance_0.95_{field_name}.h5"
    load_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                     f"results_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                             f"plots_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join("..", "data", "2D", "OAT15", "vertices_and_masks.pt"))
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)

    # load the airfoil(s) as overlay for contourf plots
    geometry = [load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "oat15_airfoil_no_TE.stl"), dimensions="xz")]
    # geometry.append(load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "naca_airfoil_no_TE.stl"),
    #                 dimensions="xz"))

    # load the pressure field of the original CFD data, small area around the leading airfoil
    field_orig = pt.load(join("..", "data", "2D", "OAT15", "p_small_every10.pt"))
    # field_orig = pt.load(join("/media", "janis", "Elements", "FOR_data", "oat15_aoa5_tandem_Johannes",
    #                           f"ma_{area}_every10.pt"))

    # load the corresponding write times and stack the coordinates
    times = pt.load(join("..", "data", "2D", "OAT15", "oat15_tandem_times.pt"))[::10]

    # load the field created by S^3. In case the data is not fitting into the RAM all at once, then the data has to be
    # loaded as it is done in the fit method of S^3's export routine
    data = construct_data_matrix_from_hdf5(join(load_path, file_name), field_name)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # create animation of fields
    ani = compare_fields(xz, data["coordinates"], field_orig, data[f"{field_name}"], len(times), geometry_=geometry)
    writer = FFMpegWriter(fps=40)
    ani.save(join(save_path_results, f"comparison_flow_fields_{field_name}.mp4"), writer=writer)

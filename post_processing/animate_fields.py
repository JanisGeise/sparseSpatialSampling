"""
    create an animation of the original flow field and the interpolated one
"""
import torch as pt
import matplotlib.pyplot as plt

from os.path import join
from os import path, makedirs
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Polygon

from compute_error import load_airfoil_as_stl_file

# increase resolution
plt.rcParams['figure.dpi'] = 640


def compare_fields(orig_coord: pt.Tensor, interpolated_coord: pt.Tensor, orig_field: pt.Tensor,
                   interpolated_field: pt.Tensor, n_frames: int, geometry: list = None):

    fig, ax = plt.subplots(nrows=2, figsize=(6, 4), sharex="col")
    for a in range(2):
        ax[a].set_ylabel("$z$")
        ax[a].set_aspect("equal")
    ax[1].set_xlabel("$x$")
    fig.text(0.8, 0.85, "$original$", ha="center", va="center", backgroundcolor="white")
    fig.text(0.8, 0.45, "$interpolated$", ha="center", va="center", backgroundcolor="white")
    fig.tight_layout()
    fig.subplots_adjust()

    # animate function adopted from @AndreWeiner
    def animate(i):
        print("\r", f"Creating frame {i+1:03d} / {n_frames}", end="")
        ax[0].clear()
        ax[1].clear()
        tcf1 = ax[0].tricontourf(orig_coord[:, 0], orig_coord[:, 1], orig_field[:, i], levels=25)
        tcf2 = ax[1].tricontourf(interpolated_coord[:, 0], interpolated_coord[:, 1], interpolated_field[:, i],
                                 levels=50)

        # add colorbar
        # fig.colorbar(tcf1, ax=ax[0], label=f"${name}$" + "$_{orig}$", shrink=0.5)
        # fig.colorbar(tcf2, ax=ax[1], label=f"${name}$" + "$_{interpolated}$", shrink=0.5)

        if geometry is not None:
            for g in geometry:
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
    field_name = "Ma"
    area = "large"
    file_name = f"OAT15_{area}_area_variance_0.95.pt"
    load_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                     f"results_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15",
                             f"plots_metric_based_on_{field_name}_stl_{area}_no_dl_constraint")

    # load the coordinates of the original grid used in CFD
    xz = pt.load(join("..", "data", "2D", "OAT15", "vertices_and_masks.pt"))
    xz = pt.stack([xz[f"x_{area}"], xz[f"z_{area}"]], dim=-1)

    # load the airfoil(s) as overlay for contourf plots
    oat15 = load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "oat15_airfoil_no_TE.stl"), dimensions="xz")
    naca = load_airfoil_as_stl_file(join("..", "data", "2D", "OAT15", "naca_airfoil_no_TE.stl"), dimensions="xz")

    # load the pressure field of the original CFD data, small area around the leading airfoil
    # field_orig = pt.load(join("..", "data", "2D", "OAT15", "p_small_every10.pt"))
    load_path_ma_large = join("/media", "janis", "Elements", "FOR_data", "oat15_aoa5_tandem_Johannes")
    field_orig = pt.load(join(load_path_ma_large, f"ma_{area}_every10.pt"))

    # load the corresponding write times and stack the coordinates
    times = pt.load(join("..", "data", "2D", "OAT15", "oat15_tandem_times.pt"))[::10]

    # load the field created by S^3. In case the data is not fitting into the RAM all at once, then the data has to be
    # loaded as it is done in the fit method of S^3's export routine
    data = pt.load(join(load_path, file_name))

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # create animation of fields
    ani = compare_fields(xz, data["coordinates"], field_orig, data[f"{field_name}"], len(times), geometry=[oat15, naca])
    writer = FFMpegWriter(fps=40)
    ani.save(join(save_path_results, f"comparison_flow_fields_{field_name}.mp4"), writer=writer)
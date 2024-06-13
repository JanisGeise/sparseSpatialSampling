"""
    compare the execution times and N_cells for different cases
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs

from post_processing.plot_results_parameter_study_variance import load_results


def plot_execution_times(_data: list, _save_path: str, case: list, save_name: str = "t_exec_vs_metric") -> None:
    # create directory for plots
    if not path.exists(_save_path):
        makedirs(_save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ls = ["-", "--", ":", "-."]

    # plot variance vs N_cells
    fig, ax = plt.subplots(ncols=2, figsize=(6, 4))
    for i, d in enumerate(_data):
        t_tot = pt.tensor(d["t_total"])
        ax[0].plot(d["final_metric"], pt.tensor(d["t_uniform"]) / t_tot, marker="x",
                   label=r"$t_{uniform}$" + f" $({case[i]})$", color=color[0], ls=ls[i])
        ax[0].plot(d["final_metric"], pt.tensor(d["t_adaptive"]) / t_tot, marker="x",
                   label=r"$t_{adaptive}$" + f" $({case[i]})$", color=color[1], ls=ls[i])
        ax[0].plot(d["final_metric"], pt.tensor(d["t_renumbering"]) / t_tot, marker="x",
                   label=r"$t_{renumbering}$" + f" $({case[i]})$", color=color[2], ls=ls[i])

        ax[1].plot(d["final_metric"], pt.tensor(d["t_uniform"]), marker="x", color=color[0], ls=ls[i])
        ax[1].plot(d["final_metric"], pt.tensor(d["t_adaptive"]), marker="x", color=color[1], ls=ls[i])
        ax[1].plot(d["final_metric"], pt.tensor(d["t_renumbering"]), marker="x", color=color[2], ls=ls[i])

        if d["t_geometry"]:
            ax[0].plot(d["final_metric"], pt.tensor(d["t_geometry"]) / t_tot, marker="x",
                       label=r"$t_{geometry}$" + f" $({case[i]})$", color=color[3], ls=ls[i])
            ax[1].plot(d["final_metric"], pt.tensor(d["t_geometry"]), marker="x", color=color[3], ls=ls[i])

    ax[0].set_xlabel(r"$metric$ / $metric_{orig}$")
    ax[1].set_xlabel(r"$metric$ / $metric_{orig}$")
    ax[0].set_ylabel(r"$t$ / $t_{total}$")
    ax[1].set_ylabel(r"$t$ $[s]$")
    fig.legend(ncols=3, loc="upper center")
    fig.tight_layout()
    fig.subplots_adjust(top=0.8, hspace=0.2)
    plt.savefig(join(save_path, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_n_cells(_data: list, _save_path: str, case: list, save_name: str = "n_cells_vs_metric") -> None:
    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # plot variance vs N_cells
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, d in enumerate(_data):
        ax.plot(d["final_metric"], pt.tensor(d["n_cells"]) / pt.tensor(d["n_cells_orig"]), marker="x",
                label=fr"${case[i]}$")

    ax.set_xlabel(r"$metric$ / $metric_{orig}$")
    ax.set_ylabel(r"$N_{cells} \, / \, N_{cells, orig}$")
    ax.legend(ncols=3, loc="upper left")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(join(_save_path, f"{save_name}.png"), dpi=340)
    plt.close("all")


if __name__ == "__main__":
    # -------------------------------------------- cylinder --------------------------------------------
    load_path = join("..", "run", "parameter_study_variance_as_stopping_criteria")
    save_path = join("..", "run", "parameter_study_variance_as_stopping_criteria", "plots_comparison_execution_times")
    cases = [join("OAT15", "results_metric_based_on_Ma_stl_large_no_dl_constraint"),
             join("surfaceMountedCube", "results_2000_snapshots_no_dl_constraint_metric_based_on_std_p_with_geometry_refinement")]

    # load the data
    data = [load_results(join(load_path, c)) for c in cases]

    # plot the results
    plot_execution_times(data, save_path, ["OAT", "cube"])
    plot_n_cells(data, save_path, ["OAT", "cube"])

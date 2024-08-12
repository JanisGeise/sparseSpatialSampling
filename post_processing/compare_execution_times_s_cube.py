"""
    compare the execution times and N_cells for different cases
"""
import torch as pt
import matplotlib.pyplot as plt

from os.path import join
from os import path, makedirs

from plot_results_parameter_study_variance import load_results

# use latex fonts
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": [r'\usepackage{amsmath}']})


def plot_execution_times(_data: list, _save_path: str, case: list, save_name: str = "t_exec_vs_metric") -> None:
    # create directory for plots
    if not path.exists(_save_path):
        makedirs(_save_path)

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ls = ["-", "--", ":", "-."]

    # plot variance vs N_cells
    fig, ax = plt.subplots(ncols=2, figsize=(6, 4))
    for i, d in enumerate(_data):
        t_tot = pt.tensor(d["t_total"])
        ax[0].plot(d["final_metric"], pt.tensor(d["t_uniform"]) / t_tot, marker="x",
                   label=r"$t_{uniform}$" + f" ({case[i]})", color=color[0], ls=ls[i])
        ax[0].plot(d["final_metric"], pt.tensor(d["t_adaptive"]) / t_tot, marker="x",
                   label=r"$t_{adaptive}$" + f" ({case[i]})", color=color[1], ls=ls[i])
        ax[0].plot(d["final_metric"], pt.tensor(d["t_renumbering"]) / t_tot, marker="x",
                   label=r"$t_{renumbering}$" + f" ({case[i]})", color=color[2], ls=ls[i])

        ax[1].plot(d["final_metric"], pt.tensor(d["t_uniform"]), marker="x", color=color[0], ls=ls[i])
        ax[1].plot(d["final_metric"], pt.tensor(d["t_adaptive"]), marker="x", color=color[1], ls=ls[i])
        ax[1].plot(d["final_metric"], pt.tensor(d["t_renumbering"]), marker="x", color=color[2], ls=ls[i])

        # only plot execution time of geometry refinement if available for all cases
        if d["t_geometry"][0] is not None:
            ax[0].plot(d["final_metric"], pt.tensor(d["t_geometry"]) / t_tot, marker="x",
                       label=r"$t_{geometry}$" + f" ({case[i]})", color=color[3], ls=ls[i])
            ax[1].plot(d["final_metric"], pt.tensor(d["t_geometry"]), marker="x", color=color[3], ls=ls[i])

    ax[0].set_xlabel(r"$\mathcal{M} \, / \, \mathcal{M}_{orig}$")
    ax[1].set_xlabel(r"$\mathcal{M} \, / \, \mathcal{M}_{orig}$")
    ax[0].set_ylabel(r"$t \, / \, t_{total}$")
    ax[1].set_ylabel(r"$t$ $[s]$")
    fig.legend(ncols=3, loc="upper center", fontsize=6)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8, hspace=0.2)
    plt.savefig(join(save_path, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_n_cells_and_t_exec(_data: list, _save_path: str, case: list, save_name: str = "n_cells_t_exec_vs_metric") -> None:
    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ls = ["-", "--", "-.", ":"]

    # plot variance vs N_cells
    fig, ax = plt.subplots(figsize=(6, 3), ncols=2, sharey="row")
    for i, d in enumerate(_data):
        ax[0].plot(d["final_metric"], pt.tensor(d["n_cells"]) / pt.tensor(d["n_cells_orig"]), marker="x",
                   label=fr"{case[i]}", color=color[i], ls=ls[0])

        t_tot = pt.tensor(d["t_total"])
        if i == 0:
            ax[1].plot(d["final_metric"], pt.tensor(d["t_uniform"]) / t_tot, marker="x",
                       label=r"$t_{u}$", color=color[i], ls=ls[0])
            ax[1].plot(d["final_metric"], pt.tensor(d["t_adaptive"]) / t_tot, marker="x",
                       label=r"$t_{a}$", color=color[i], ls=ls[1])
            ax[1].plot(d["final_metric"], pt.tensor(d["t_renumbering"]) / t_tot, marker="x",
                       label=r"$t_{rn}$", color=color[i], ls=ls[2])
            # only plot execution time of geometry refinement if available for all cases
            if d["t_geometry"][0] is not None:
                ax[1].plot(d["final_metric"], pt.tensor(d["t_geometry"]) / t_tot, marker="x",
                           label=r"$t_{g}$", color=color[i], ls=ls[3])
        else:
            ax[1].plot(d["final_metric"], pt.tensor(d["t_uniform"]) / t_tot, marker="x", color=color[i], ls=ls[0])
            ax[1].plot(d["final_metric"], pt.tensor(d["t_adaptive"]) / t_tot, marker="x", color=color[i], ls=ls[1])
            ax[1].plot(d["final_metric"], pt.tensor(d["t_renumbering"]) / t_tot, marker="x", color=color[i], ls=ls[2])
            # only plot execution time of geometry refinement if available for all cases
            if d["t_geometry"][0] is not None:
                ax[1].plot(d["final_metric"], pt.tensor(d["t_geometry"]) / t_tot, marker="x", color=color[i], ls=ls[3])

    ax[0].set_xlabel(r"$\mathcal{M} \, / \, \mathcal{M}_{orig}$")
    ax[1].set_xlabel(r"$\mathcal{M} \, / \, \mathcal{M}_{orig}$")
    ax[0].set_ylabel(r"$N_{cells} \, / \, N_{cells, orig}$")
    ax[1].set_ylabel(r"$t \, / \, t_{tot}$")
    ax[0].legend(ncols=1, loc="upper left")
    ax[1].legend(ncols=1)
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(_save_path, f"{save_name}.png"), dpi=340)
    plt.close("all")


if __name__ == "__main__":
    # -------------------------------------------- cylinder --------------------------------------------
    load_path = join("..", "run", "final_benchmarks")
    save_path = join("..", "run", "final_benchmarks", "plots")
    cases = [join("OAT15_large", "results_with_geometry_refinement_no_dl_constraint"),
             join("surfaceMountedCube_local_TKE", "results_no_geometry_refinement_no_dl_constraint")]
    legend = [r"$OAT$", r"$SurfaceMountedCube$"]

    # load the data
    data = [load_results(join(load_path, c)) for c in cases]

    # plot the results
    plot_n_cells_and_t_exec(data, save_path, legend)
    plot_execution_times(data, save_path, legend)

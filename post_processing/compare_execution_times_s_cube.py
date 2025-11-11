"""
    compare the execution times and N_cells for different cases
"""
import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs


# use latex fonts
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r'\usepackage{amsmath}'})


def load_results(_load_path: str) -> dict:
    """
    loads and re-sorts results from S^3 refinement

    :param _load_path: path to the results from the refinements with S^3
    :return: the loaded and sorted data
    """
    file_names = sorted(glob(join(_load_path, "mesh_info_*.pt")),
                        key=lambda x: float(x.split("_")[-1].split(".pt")[0]))
    _data = [pt.load(f, weights_only=False) for f in file_names]

    # re-sort results from list(dict) to dict(list) in order to plot the results easier / more efficient
    data_out = {}
    for key in list(_data[0].keys()):
        data_out[key] = [i[key] for i in _data]

    # add extra key for final variance of grid
    data_out["final_metric"] = [i[-1] for i in data_out["metric_per_iter"]]

    return data_out


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
    fig.subplots_adjust(top=0.84, hspace=0.2)
    plt.savefig(join(save_path, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_n_cells_and_t_exec(_data: list, _save_path: str, case: list, save_name: str = "n_cells_t_exec_vs_metric") -> None:
    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    ls = ["-", "--", ":", "-."]

    # plot variance vs N_cells
    fig, ax = plt.subplots(figsize=(6, 3), ncols=2)
    for i, d in enumerate(_data):
        ax[0].plot(d["final_metric"], pt.tensor(d["n_cells"]) / pt.tensor(d["n_cells_orig"]), marker="x",
                   label=fr"{case[i]}", color=color[i], ls=ls[0])

        t_tot = pt.tensor(d["t_total"])
        if i == 0:
            ax[1].plot(d["final_metric"], pt.tensor(d["t_adaptive"]) / t_tot, marker="x",
                       label=r"$t_{\mathrm{a}}$", color=color[i], ls=ls[0])

            # only plot execution time of geometry refinement if available for all cases
            if d["t_geometry"][0] is not None:
                ax[1].plot(d["final_metric"], pt.tensor(d["t_geometry"]) / t_tot, marker="x",
                           label=r"$t_{\mathrm{g}}$", color=color[i], ls=ls[2])

            # combine execution times for renumbering & uniform because they are very small
            ax[1].plot(d["final_metric"], (pt.tensor(d["t_renumbering"]) + pt.tensor(d["t_uniform"])) / t_tot,
                       marker="x", label=r"other", color=color[i], ls=ls[1])
        else:
            ax[1].plot(d["final_metric"], pt.tensor(d["t_adaptive"]) / t_tot, marker="x", color=color[i], ls=ls[0])

            # only plot execution time of geometry refinement if available for all cases
            if d["t_geometry"][0] is not None:
                ax[1].plot(d["final_metric"], pt.tensor(d["t_geometry"]) / t_tot, marker="x", color=color[i], ls=ls[2])

            ax[1].plot(d["final_metric"], (pt.tensor(d["t_renumbering"]) + pt.tensor(d["t_uniform"])) / t_tot,
                       marker="x", color=color[i], ls=ls[1])

    # ax[0].set_xlabel(r"$\mathcal{M}_{\mathrm{approx}}$")
    # ax[1].set_xlabel(r"$\mathcal{M}_{\mathrm{approx}}$")
    fig.supxlabel(r"$\mathcal{M}_{\mathrm{approx}}$")
    ax[0].set_ylabel(r"$N_\ell \, / \, N_{\ell, \mathrm{orig}}$")
    ax[1].set_ylabel(r"$t \, / \, t_{\mathrm{tot}}$")
    ax[1].set_ylim(0.01, 1)
    ax[0].set_xlim(0.2, 1)
    ax[1].set_xlim(0.2, 1.0)
    ax[1].set_yscale("log")
    fig.legend(case, ncols=2, loc="upper center")
    ax[1].legend(ncols=1, bbox_to_anchor=(0.35, 0.3), fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    plt.savefig(join(_save_path, f"{save_name}.png"), dpi=340)
    plt.close("all")


def plot_progress_metric(_data: list, _save_path: str, case: list, save_name: str = "approximation_of_metric",
                         no: int = -1) -> None:
    # create directory for plots
    if not path.exists(_save_path):
        makedirs(_save_path)

    # use default color cycle
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # plot variance vs N_cells
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, d in enumerate(_data):
        ax.plot(d["metric_per_iter"][no], marker="x", label=case[i], color=color[i])

    ax.set_xlabel(r"$\mathrm{iteration}$ $\mathrm{no.}$ $\#$")
    ax.set_ylabel(r"$\mathcal{M}_{\mathrm{approx}}$")
    ax.set_xlim(0, max([len(d["metric_per_iter"][no])-1 for d in _data]))
    ax.legend(ncols=1, loc="lower right")
    fig.tight_layout()
    # fig.subplots_adjust(top=0.9)
    plt.savefig(join(save_path, f"{save_name}_{round(data[0]['final_metric'][no], 2)}.png"), dpi=340)
    plt.close("all")


if __name__ == "__main__":
    # -------------------------------------------- cylinder --------------------------------------------
    load_path = join("..", "run", "final_benchmarks")
    save_path = join("..", "run", "final_benchmarks", "plots_final")
    cases = [join("OAT15_large_new", "results_geometry_refinement_no_dl_constraint_fully_parallelized"),
             join("cylinder3D_Re3900_local_TKE", "results_with_geometry_refinement_no_dl_constraint_fully_parallel")]
    legend = [r"$\mathrm{tandem}$", r"$\mathrm{cylinder}$"]

    # load the data
    data = [load_results(join(load_path, c)) for c in cases]

    # plot the results
    for n in range(len(data[0])):
        plot_progress_metric(data, save_path, legend, no=n)
    plot_n_cells_and_t_exec(data, save_path, legend)
    plot_execution_times(data, save_path, legend)

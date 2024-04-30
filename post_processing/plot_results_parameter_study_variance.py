"""
    plot the results of the parameter study executed with the 'parameter_study_variance_vs_cells.py' script
"""
import os.path

import torch as pt
import matplotlib.pyplot as plt

from glob import glob
from os.path import join
from os import path, makedirs


def load_results(_load_path: str) -> dict:
    """
    loads and re-sorts results from S^3 refinement

    :param _load_path: path to the results from the refinements with S^3
    :return: the loaded and sorted data
    """
    file_names = sorted(glob(join(_load_path, "mesh_info_variance_0.*.pt")),
                        key=lambda x: int(x.split("_")[-1].split(".")[1]))
    _data = [pt.load(f) for f in file_names]

    # re-sort results from list(dict) to dict(list) in order to plot the results easier / more efficient
    data_out = {}
    for key in list(_data[0].keys()):
        data_out[key] = [i[key] for i in _data]

    # add extra key for final variance of grid
    data_out["final_variance"] = [i[-1] for i in data_out["variance_per_iter"]]

    return data_out


def plot_results_parameter_study(_data: dict, save_path: str, _n_cells_orig: int,
                                 save_name: str = "parameters_vs_captured_variance"):
    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # plot variance vs N_cells
    fig, ax = plt.subplots(nrows=1, sharex="col")
    ax.plot(data["final_variance"], pt.tensor(_data["n_cells"]) / _n_cells_orig, marker="x",
            label=r"$N_{cells} \, / \, N_{cells, orig}$")
    ax.plot(data["final_variance"], pt.tensor(_data["t_uniform"]) / pt.tensor(_data["t_total"]), marker="x",
            label=r"$t_{uniform} \, / \, t_{total}$")
    ax.plot(data["final_variance"], pt.tensor(_data["t_adaptive"]) / pt.tensor(_data["t_total"]), marker="x",
            label=r"$t_{adaptive} \, / \, t_{total}$")
    ax.plot(data["final_variance"], pt.tensor(_data["t_renumbering"]) / pt.tensor(_data["t_total"]), marker="x",
            label=r"$t_{renumbering} \, / \, t_{total}$")
    # ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\sigma(p) \, / \, \sigma(p_{orig})$")
    ax.legend()
    plt.savefig(join(save_path, f"{save_name}.png"), dpi=340)
    plt.show()


if __name__ == "__main__":
    load_path_cylinder = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D", "results")
    save_path_cylinder = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D", "plots")
    n_cells_orig = 21250

    # load the data
    data = load_results(load_path_cylinder)

    # plot the results
    plot_results_parameter_study(data, save_path_cylinder, n_cells_orig)

    # cube
    load_path_cube = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube", "results")
    save_path_cube = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube", "plots")
    n_cells_orig = 1112000

    # load the data
    data = load_results(load_path_cube)

    # plot the results
    plot_results_parameter_study(data, save_path_cube, n_cells_orig)



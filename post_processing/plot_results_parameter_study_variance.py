"""
    plot the results of the parameter study executed with the 'parameter_study_variance_vs_cells.py' script
"""
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
    file_names = sorted(glob(join(_load_path, "mesh_info_*.pt")),
                        key=lambda x: float(x.split("_")[-1].split(".pt")[0]))
    _data = [pt.load(f) for f in file_names]

    # re-sort results from list(dict) to dict(list) in order to plot the results easier / more efficient
    data_out = {}
    for key in list(_data[0].keys()):
        data_out[key] = [i[key] for i in _data]

    # add extra key for final variance of grid
    data_out["final_metric"] = [i[-1] for i in data_out["metric_per_iter"]]

    return data_out


def plot_results_parameter_study(_data: dict, save_path: str, save_name: str = "parameters_vs_metric",
                                 field: str = "p") -> None:
    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # plot variance vs N_cells
    fig, ax = plt.subplots(nrows=1, sharex="col")
    ax.plot(_data["final_metric"], pt.tensor(_data["n_cells"]) / pt.tensor(_data["n_cells_orig"]), marker="x",
            label=r"$N_{cells} \, / \, N_{cells, orig}$")
    ax.plot(_data["final_metric"], pt.tensor(_data["t_uniform"]) / pt.tensor(_data["t_total"]), marker="x",
            label=r"$t_{uniform} \, / \, t_{total}$")
    ax.plot(_data["final_metric"], pt.tensor(_data["t_adaptive"]) / pt.tensor(_data["t_total"]), marker="x",
            label=r"$t_{adaptive} \, / \, t_{total}$")
    ax.plot(_data["final_metric"], pt.tensor(_data["t_renumbering"]) / pt.tensor(_data["t_total"]), marker="x",
            label=r"$t_{renumbering} \, / \, t_{total}$")
    if _data["t_geometry"]:
        ax.plot(_data["final_metric"], pt.tensor(_data["t_geometry"]) / pt.tensor(_data["t_total"]), marker="x",
                label=r"$t_{geometry} \, / \, t_{total}$")

    # plot reference lines for variance ~ N_cells
    dx = max(_data["final_metric"]) - min(_data["final_metric"])
    ax.plot([min(_data["final_metric"]), max(_data["final_metric"])],
            [min(_data["n_cells"]) / _data["n_cells_orig"][0], min(_data["n_cells"]) / _data["n_cells_orig"][0] + dx],
            color="#1f77b4", ls="-.", label=r"$N_{cells} \propto \sigma(" + str(field) + ")$")
    ax.plot([0, 1], [0, 1], color="#1f77b4", ls=":", label=r"$N_{cells} \propto \sigma($" + str(field) + "$)$")

    ax.set_ylim(0, 1)
    ax.set_xlim(min(_data["final_metric"]) - 0.01, max(_data["final_metric"]) + 0.01)
    ax.set_xlabel(r"$\sigma(" + str(field) + r") \, / \, \sigma(" + str(field) + "_{orig})$")
    fig.legend(ncols=3, loc="upper center")
    fig.subplots_adjust(top=0.82)
    plt.savefig(join(save_path, f"{save_name}.png"), dpi=340)
    plt.show()


if __name__ == "__main__":
    # -------------------------------------------- cylinder --------------------------------------------
    load_path_cylinder = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D", "results")
    save_path_cylinder = join("..", "run", "parameter_study_variance_as_stopping_criteria", "cylinder2D", "plots")

    # load the data
    data = load_results(load_path_cylinder)

    # plot the results
    plot_results_parameter_study(data, save_path_cylinder)

    # -------------------------------------------- cube --------------------------------------------
    load_path_cube = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube", "results")
    save_path_cube = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube", "plots")

    # load the data
    data = load_results(load_path_cube)

    # plot the results
    plot_results_parameter_study(data, save_path_cube)

    # -------------------------------------------- OAT 15 airfoil --------------------------------------------
    load_path_oat15 = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15", "results")
    save_path_oat15 = join("..", "run", "parameter_study_variance_as_stopping_criteria", "OAT15", "plots")

    # load the data
    data = load_results(load_path_oat15)

    # plot the results
    plot_results_parameter_study(data, save_path_oat15, field="Ma")

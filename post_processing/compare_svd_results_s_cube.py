"""
    same as compare_svd, but here the results are already present in the HDF file created when executing S^3 and
    performing an SVD on the interpolated fields
"""
import h5py
import torch as pt
import matplotlib.pyplot as plt

from os.path import join
from os import path, makedirs

from compare_svd_OAT import plot_mode_coefficients, plot_singular_values, plot_psd

# use latex fonts
plt.rcParams.update({"text.usetex": True})


def load_svd_from_hdf5(_load_path: str, _file_name: str):
    data = h5py.File(join(_load_path, f"{_file_name}.h5"))

    # re-sort the modes into a single tensor
    modes = pt.stack([pt.from_numpy(data.get(f"constant/{d}")[()]) for d in data["constant"].keys() if
                      d.startswith("mode_")], dim=-1)
    return data.get("constant/s")[()], modes, data.get("constant/V")[()]


if __name__ == "__main__":
    # path to the data and to the directory the results should be saved to
    load_path = join("/media", "janis", "Elements", "FOR_data", "results_s_cube_Janis")
    save_path_results = join("..", "run", "final_benchmarks", "surfaceMountedCube_local_TKE",
                             "plots_no_geometry_refinement_no_dl_constraint")

    # field name and flag if scalar or vector field
    field = "U"
    metric = "0.50"
    scalar = False
    cases = [f"cube_2000_snapshots_svd_{field}",
             join("surfaceMountedCube_local_TKE", "results_no_geometry_refinement_no_dl_constraint",
                  f"surfaceMountedCube_metric_TKE_0.50_{field}_svd")]

    # list with legend entries
    legend = ["$original$", r"$\mathcal{M} = 0.51$"]

    # time step and number of time steps for computing the PSD
    dt, n_dt = 0.05, 2001

    # load SVD for the specified cases
    results_svd = {"s": [], "U": [], "V": []}
    for c in cases:
        s, U, V = load_svd_from_hdf5(load_path, c)
        results_svd["s"].append(pt.from_numpy(s))
        results_svd["U"].append(U)
        results_svd["V"].append(V)
    del s, U, V

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # plot the singular values
    plot_singular_values(results_svd["s"], save_path_results, f"comparison_singular_values_field_{field}_{metric}", legend)

    # compare the PSD
    plot_psd(results_svd["V"], dt, n_dt, save_path_results, f"comparison_psd_field_{field}", chord=1, u_inf=1,
             xlim=(0, 0.3), legend=legend)

    # compare mode coefficients
    plot_mode_coefficients(pt.linspace(30, 130, n_dt), results_svd["V"], save_path_results,
                           f"comparison_mode_coefficients_field_{field}", legend, chord=1, u_inf=1)

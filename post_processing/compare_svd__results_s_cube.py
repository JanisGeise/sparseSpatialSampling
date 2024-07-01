"""
    same as compare_svd, but here the results are already present in the HDF file created when executing S^3 and
    performing an SVD on the interpolated fields
"""
import h5py
import torch as pt
import matplotlib.pyplot as plt

from os.path import join
from os import path, makedirs

from compare_svd import plot_mode_coefficients, plot_singular_values, plot_psd


def load_svd_from_hdf5(_load_path: str, _file_name: str, _scalar: bool = True):
    data = h5py.File(join(_load_path, f"{_file_name}.h5"))
    if _scalar:
        return data.get("singular_values")[()], data.get("mode")[()], data.get("mode_coefficients")[()]
    else:
        modes = [data.get(f"mode_{d}")[()] for d in ["x", "y", "z"]]
        return data.get("singular_values")[()], modes, data.get("mode_coefficients")[()]


if __name__ == "__main__":
    # path to the data and to the directory the results should be saved to
    # load_path = join("..", "data", "3D", "surfaceMountedCube_s_cube_Janis", "hpc")
    load_path = join("/media", "janis", "Elements", "FOR_data", "surfaceMountedCube_s_cube_Janis")
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube",
                             "plots_surfaceMountedCube_s_cube_2000_snapshots", "comparison_psd_metrics")

    # field name and flag if scalar or vector field
    field = "p"
    scalar = True
    cases = [f"cube_2000_snapshots_svd_{field}",
             join("hpc", "results_metric_0.50", f"surfaceMountedCube_metric_std_pressure_0.50_svd_{field}"),
             join("hpc", "results_metric_0.50", f"surfaceMountedCube_metric_std_mag_velocity_0.50_svd_{field}")]

    # list with legend entries
    legend = ["$original$", r"$\sigma(p) = 66.54\%$", r"$\sigma(L_2(U)) = 50.16\%$"]

    # time step and number of time steps for computing the PSD
    dt, n_dt = 0.05, 2001

    # load SVD for the specified cases
    results_svd = {"s": [], "U": [], "V": []}
    for c in cases:
        s, U, V = load_svd_from_hdf5(load_path, c, _scalar=scalar)
        results_svd["s"].append(pt.from_numpy(s))
        results_svd["U"].append(U)
        results_svd["V"].append(V)
    del s, U, V

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # plot the singular values
    plot_singular_values(results_svd["s"], save_path_results, f"comparison_singular_values_field_{field}", legend)

    # compare the PSD
    plot_psd(results_svd["V"], dt, n_dt, save_path_results, f"comparison_psd_field_{field}", chord=1, u_inf=1,
             xlim=(0, 0.3), legend=legend)

    # compare mode coefficients
    plot_mode_coefficients(pt.linspace(30, 130, n_dt), results_svd["V"], save_path_results,
                           f"comparison_mode_coefficients_field_{field}", legend)

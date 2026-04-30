"""
compare the SVD results of the DDES for the XRF-1 case
"""
import torch as pt

from h5py import File
from os.path import join
from os import path, makedirs

from compare_svd_OAT import plot_mode_coefficients, plot_error_map_coefficients, plot_singular_values
from sparseSpatialSampling.data import Dataloader

if __name__ == "__main__":
    # which fields and settings to use
    load_path = join("..", "run", "final_benchmarks", f"DDES_XRF1", "results_SVD")
    save_path = join("..", "run", "final_benchmarks", f"DDES_XRF1", "plots_SVD")
    file_name = ""

    # load scube info
    # scube_info = pt.load(join(load_path, "mesh_info_grid_s_cube.pt"), weights_only=False)
    # scube_info = File(join(load_path, "grid_s_cube.h5"), mode="r")

    # legend entries
    legend = [r"$\mathrm{original}$", r"$\mathcal{M}_\mathrm{min} = 0.95$"]

    # load the time steps
    times = pt.tensor(list(map(float, pt.load(join(load_path, "times.pt"), weights_only=False))))

    # create directory for plots
    if not path.exists(save_path):
        makedirs(save_path)

    # load the singular values and right singular vectors of the original DDES
    s_orig = pt.load(join(load_path, "scube_output_compareSVD_normFalse/s_orig.pt"), weights_only=False)
    V_orig = pt.load(join(load_path, "scube_output_compareSVD_normFalse/V_orig.pt"), weights_only=False)

    # load the ones from Scube
    file = File(join(load_path, "grid_s_cube_pressure_svd.h5"), mode="r")
    sv = [s_orig, pt.from_numpy(file["constant"]["s"][()])]
    V_scube = pt.from_numpy(file["constant"]["V"][()])

    # plot relative error within the mode coefficients
    error = (V_orig.abs() - V_scube.abs()).abs() / V_orig.norm()
    plot_error_map_coefficients(error, times, save_path, f"relative_error_map_coefficients", chord=0.1965,
                                u_inf=269.836237015814)

    # plot singular values
    plot_singular_values(sv, save_path, f"comparison_singular_values", legend=legend)

    # plot POD mode coefficients (right singular vectors)
    plot_mode_coefficients(times, [V_orig, V_scube], save_path, f"comparison_pod_mode_coefficients",
                           legend=legend, n_modes=6, chord=0.1965, u_inf=269.836237015814)
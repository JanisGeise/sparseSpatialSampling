"""
    Compute an SVD of the interpolated field (generated grid with S^3) and compare it to the original field from CFD
    for the surfaceMountedCube test case
"""
import h5py
import torch as pt

from os.path import join
from os import path, makedirs

from flowtorch.data import FOAMDataloader

from post_processing.compare_svd_OAT import plot_psd, plot_singular_values, plot_mode_coefficients

if __name__ == "__main__":
    # which fields and settings to use
    field_name = "U"
    metric = ["0.25", "0.50", "0.95"]

    # path to the HDF5 file
    load_path = join("..", "run", "final_benchmarks", f"surfaceMountedCube_local_TKE",
                     "results_no_geometry_refinement_no_dl_constraint")
    file_name = [f"surfaceMountedCube_metric_TKE_{m}_{field_name}_svd.h5" for m in metric]

    # the actual metric may differ from the filename
    legend = ["$original$"] + [r"$\mathcal{M} = " + f"{m}$" for m in ["0.34", "0.51", "0.95"]]

    # path to the directory to which the plots should be saved to
    save_path_results = join("..", "run", "final_benchmarks", "surfaceMountedCube_local_TKE",
                             "plots_no_geometry_refinement_no_dl_constraint")

    # load the field of the original CFD data
    svd_orig = h5py.File(join("..", "data", "3D", "surfaceMountedCube_s_cube_Janis",
                              f"cube_2000_snapshots_{field_name}_svd.h5"))

    # field from generated grid
    svd_inter = [h5py.File(join(load_path, f)) for f in file_name]

    # load the corresponding write time
    # foamloader = FOAMDataloader(join("/media", "janis", "Elements", "FOR_data", "surfaceMountedCube_Janis", "fullCase"))
    # times = pt.tensor(list(map(float, foamloader.write_times))[1:])
    times = pt.arange(30, 130.04, 0.05)

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # plot frequency spectrum
    plot_psd([svd_orig.get("constant/V")[()]]+[sv.get("constant/V")[()] for sv in svd_inter],
             (times[1] - times[0]).item(), len(times), save_path_results,
             f"comparison_psd_metric_{metric[0]}_{metric[-1]}", legend=legend, xlim=(0, 0.3), u_inf=1, chord=1,
             n_modes=6)

    # plot singular values
    plot_singular_values([pt.from_numpy(svd_orig.get("constant/s")[()])] +
                         [pt.from_numpy(sv.get("constant/s")[()]) for sv in svd_inter], save_path_results,
                         f"comparison_singular_values_metric_{metric[0]}_{metric[-1]}", legend=legend)

    # plot POD mode coefficients (right singular vectors)
    plot_mode_coefficients(times, [svd_orig.get("constant/V")[()]] +
                           [sv.get("constant/V")[()] for sv in svd_inter], save_path_results,
                           f"comparison_pod_mode_coefficients_metric_{metric[0]}_{metric[-1]}",
                           legend=legend, u_inf=1, chord=1, n_modes=6)

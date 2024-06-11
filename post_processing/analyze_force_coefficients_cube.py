"""
    Analyze the force coefficients from the surfaceMountedCube simulation. This script is taken from @AndreWeiner.
    The original notebook can be found under:

    https://github.com/AndreWeiner/ofw2022_dmd_training/blob/main/preliminary_flow_analysis.ipynb
"""
import matplotlib.pyplot as plt

from os.path import join
from pandas import read_csv
from os import path, makedirs
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d

if __name__ == "__main__":
    # paths to the data & save directory
    load_path = join("..", "data", "3D", "surfaceMountedCube_s_cube_Janis", "postProcessing")
    save_path_results = join("..", "run", "parameter_study_variance_as_stopping_criteria", "surfaceMountedCube",
                             "plots_surfaceMountedCube_s_cube_Janis")

    # append to save name
    metric = "0.95"

    # load the coefficients
    columns = ["t", "cx", "cy", "cz"]
    coeffs = read_csv(join(load_path, "forces", "0", "coefficient.dat"), sep=r"\s+", comment="#", names=columns)

    # time step between samples
    dt = coeffs.t.values[1] - coeffs.t.values[0]

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # create directory for plots
    if not path.exists(save_path_results):
        makedirs(save_path_results)

    # plot the coefficients
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, c in enumerate(zip([coeffs.cx, coeffs.cy, coeffs.cz], [r"$c_x$", r"$c_y$", r"$c_z$"])):
        ax.scatter(coeffs.t, c[0], s=1, marker="x", label=c[1])

        if i == 2:
            ax.plot(coeffs.t, gaussian_filter1d(c[0], 20), c="k", label="smoothed")
        else:
            ax.plot(coeffs.t, gaussian_filter1d(c[0], 20), c="k")
    ax.set_xlim(coeffs.t.min(), coeffs.t.max())
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r"$t$ in $s$")
    ax.set_ylabel(r"$c_i$")
    fig.legend(ncol=4, loc="upper center")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_path_results, f"force_coefficients_metric_{metric}.png"), dpi=340)
    plt.close("all")

    # compute & plot PSD
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in zip([coeffs.cx, coeffs.cy, coeffs.cz], [r"$c_x$", r"$c_y$", r"$c_z$"]):
        ci_smooth = gaussian_filter1d(c[0], 20)
        freq, amp = welch(ci_smooth - ci_smooth.mean(), fs=1/dt, nperseg=int(len(ci_smooth)/2), nfft=2 * len(ci_smooth))
        ax.plot(freq, amp, label=c[1])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$f$ in $Hz$")
    ax.set_ylabel(r"$PSD$")
    ax.set_xlim(1e-3, 0.5 / dt)
    fig.legend(ncol=4, loc="upper center")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(join(save_path_results, f"psd_force_coefficients_metric_{metric}.png"), dpi=340)
    plt.close("all")

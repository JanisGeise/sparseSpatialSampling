"""
compute the temporal error for the generic metrics.
"""
import torch as pt
import matplotlib.pyplot as plt

from os import makedirs
from os.path import exists, join
from sklearn.neighbors import KNeighborsRegressor

from sparseSpatialSampling.data import Dataloader

# use latex fonts
plt.rcParams.update({"text.usetex": True})

def load_original_field(load_dir: str, field_name: str) -> pt.Tensor:
    return pt.load(join(load_dir, f"{field_name}_large_every10.pt"), weights_only=False)


if __name__ == "__main__":
    # paths
    load_path_orig_fields = join("..", "data", "2D", "OAT15")
    load_path_scube_fields = join("..", "run", "final_benchmarks", "OAT15_large_new", "results_generic_metrics")
    save_path = join("..", "run", "final_benchmarks", "OAT15_large_new", "plots_generic_metrics")

    # file names and plot settings
    fields = ["p", "rho", "u", "ma"]
    cases = ["mu_only", "sigma_only"]
    legend = ["$p$", r"$\rho$", "$u_1$", "$u_2$", "$u_3$", "$Ma$"]
    chord, u_inf = 0.15, 238.59
    weighted = True

    # create directory
    if not exists(save_path):
        makedirs(save_path)

    # load the coordinates
    xz = pt.load(join("..", "data", "2D", "OAT15", "vertices_and_masks.pt"), weights_only=False)
    cell_area_orig = xz[f"area_large"].unsqueeze(-1).sqrt()
    xz = pt.stack([xz[f"x_large"], xz[f"z_large"]], dim=-1)

    # interpolate Scube data onto the original grid in order to compute the error
    knn = KNeighborsRegressor(n_neighbors=8 if xz.size(-1) == 2 else 26, weights="distance", n_jobs=8)

    # load the original flow fields and the Scube data, compute the norm wrt time for each field
    all_errors = []
    for c in cases:
        print(f"Loading case '{c}'.")
        loader = Dataloader(load_path_scube_fields, f"metric_{c}_variance_0.75.h5")
        error = {}

        for field in fields:
            # load the original data
            print(f"Loading original field '{field}'.")
            if weighted:
                _f_orig = load_original_field(load_path_orig_fields, field)
                if len(_f_orig.size()) > 2:
                    _f_orig *= cell_area_orig.unsqueeze(-1)
                else:
                    _f_orig *= cell_area_orig
            else:
                _f_orig = load_original_field(load_path_orig_fields, field)
            _norm_f_orig = pt.linalg.norm(_f_orig, dim=0)

            # load the Scube data
            print(f"Loading and interpolating field '{field}'.")
            _fields = loader.load_snapshot(field)
            if len(_fields.size()) > 2:
                _f = []
                for cmp in range(_fields.size(1)):
                    knn.fit(loader.vertices, _fields[:, cmp, :])

                    if weighted:
                        _f.append(pt.from_numpy(knn.predict(xz)) * cell_area_orig)
                    else:
                        _f.append(pt.from_numpy(knn.predict(xz)))
            else:
                knn.fit(loader.vertices, _fields)

                if weighted:
                    _f = pt.from_numpy(knn.predict(xz)) * cell_area_orig
                else:
                    _f = pt.from_numpy(knn.predict(xz))
            del _fields

            # compute error, then free up some space
            if isinstance(_f, list):
                for cmp in range(len(_f)):
                    # print the global error
                    print(f"global error {field}_{cmp}:\t",
                          (pt.linalg.norm(_f[cmp] - _f_orig[:, cmp, :], ord=2) / _f_orig[:, cmp, :].norm(p=2)).item())
                    error[f"{field}_{cmp}"] = pt.linalg.norm(_f[cmp] - _f_orig[:, cmp, :], dim=0, ord=2) / _norm_f_orig[cmp, :]

            else:
                print(f"global error {field}:\t", (pt.linalg.norm(_f - _f_orig, ord=2) / _f_orig.norm(p=2)).item())
                error[field] = pt.linalg.norm(_f - _f_orig, dim=0) / _norm_f_orig
            del _norm_f_orig, _f_orig, _f

        all_errors.append(error)
    del loader, error, cell_area_orig

    # load the corresponding write times for plotting
    times = pt.load(join(load_path_orig_fields, "oat15_tandem_times.pt"), weights_only=False)[::10] * (u_inf / chord)

    # plot temporal evolution of error for all fields
    ls = ["-", "--", "-.", ":"]
    color = ["C0", "C1", "C2", "C3", "C4", "C5"]
    fig, ax = plt.subplots(figsize=(6, 3))
    for c in range(len(cases)):
        for i, f in enumerate(zip(all_errors[c].keys(), legend)):
            # discard u_y, since 2D case
            if f[0] == "u_1":
                continue
            if c == 0:
                ax.plot(times, all_errors[c][f[0]], label=f"${f[1]}$", ls=ls[c], color=color[i])
            else:
                ax.plot(times, all_errors[c][f[0]], ls=ls[c], color=color[i])

    ax.set_xlim(times.min(), times.max())
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e-1)
    ax.set_xlabel(r"$\tau$")
    # ax.set_ylabel(r"$|| \Delta \mathbf{f} ||_2 \, / \, || \mathbf{f} ||_2$")
    ax.set_ylabel(r"$\Delta f_n$")
    fig.tight_layout()
    fig.legend(loc="upper center", framealpha=1, ncol=6)
    fig.subplots_adjust(top=0.86)
    if weighted:
        plt.savefig(join(save_path, f"comparison_temporal_error_weighted_new.png"), dpi=340)
    else:
        plt.savefig(join(save_path, f"comparison_temporal_error.png"), dpi=340)
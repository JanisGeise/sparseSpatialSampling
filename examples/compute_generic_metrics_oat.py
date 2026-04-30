"""
script for computing the generic metrics for the OAT15 airfoil, in particular:

        1. mu(p/p_infty) + mu(rho/rho_infty) + mu(U/U_infty)
        2. sigma(p/p_infty) + sigma(rho/rho_infty) + sigma(U/U_infty)
        3. ((1.) + (2.))

to assess the robustness of S^3 wrt the metric.
"""
import torch as pt

from typing import Tuple
from os import makedirs
from os.path import exists, join

def compute_mean_and_std(load_dir: str, field_name: str) -> Tuple[pt.Tensor, pt.Tensor]:
    print(f"Loading snapshots for field {field_name}.")

    if field_name == "vel":
        _field = []
        for cmp in ["x", "y", "z"]:
            _field.append(pt.load(join(load_dir, f"{field_name}_{cmp}_large_every10.pt"), weights_only=False).unsqueeze(1))
        _field = pt.stack(_field, dim=1).squeeze().pow(2).sum(1).sqrt()
    elif field_name == "u":
        _field = pt.load(join(load_dir, f"{field_name}_large_every10.pt"), weights_only=False).pow(2).sum(1).sqrt()
    else:
        _field = pt.load(join(load_dir, f"{field_name}_large_every10.pt"), weights_only=False)

    # pt.save(_field, join(save_path, f"{field_name}_large_every10.pt"))

    return _field.mean(-1), _field.std(-1)


if __name__ == "__main__":
    # free stream quantities
    u_mag_infty = 238.59
    rho_infty = 0.959635
    p_infty = 75230

    # paths
    load_path = join("..", "data", "2D", "OAT15")
    save_path = join("..", "run", "final_benchmarks", "OAT15_large_new", "results_generic_metrics")

    # create directory
    if not exists(save_path):
        makedirs(save_path)

    # load the fields
    p_mean, p_std = compute_mean_and_std(load_path, "p")
    rho_mean, rho_std = compute_mean_and_std(load_path, "rho")
    u_mag_mean, u_mag_std = compute_mean_and_std(load_path, "u")

    # compute the metrics
    metric_mu_only = p_mean/p_infty + rho_mean/rho_infty + u_mag_mean/u_mag_infty
    metric_sigma_only = p_std/p_infty + rho_std/rho_infty + u_mag_std/u_mag_infty
    metric_mu_and_sigma = metric_mu_only + metric_sigma_only

    # save the metrics for S^3 execution
    pt.save(metric_mu_only, join(save_path, "metric_mu_only.pt"))
    pt.save(metric_sigma_only, join(save_path, "metric_sigma_only.pt"))
    pt.save(metric_mu_and_sigma, join(save_path,"metric_mu_and_sigma.pt"))
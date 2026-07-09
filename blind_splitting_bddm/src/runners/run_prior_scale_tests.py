"""BDDM prior scale inference tests."""

from __future__ import annotations

import numpy as np
import torch

from blind_splitting_bddm.src.plotting import save_line_plot
from blind_splitting_bddm.src.priors_gmm import make_prior, make_sigma_grid
from blind_splitting_bddm.src.utils import common_parser, ensure_dirs, prepare_config, seed_all, write_markdown
from posterior_bddm_oracle.src.torch_oracle_tools import make_generator


STAT_NAMES = ["mean", "median", "std", "q05", "q95", "min_hit", "max_hit", "entropy"]


def run(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 123))
    seed_all(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if cfg.get("dtype") == "float32" else torch.float64
    paths = ensure_dirs(cfg.get("save_dir", "blind_splitting_bddm/results/prior_scale"))

    prior_cfg = dict(cfg.get("prior", {}))
    exp = cfg.get("experiments", {})
    d_values = cfg.get("d_values", exp.get("d_values", [2, 10, 50, 100, 500]))
    sigmas = exp.get("sigmas", [0.05, 0.1, 0.3, 1.0])
    n_trials = int(exp.get("n_trials", 128))
    sigma_grid = make_sigma_grid(cfg.get("sigma_grid", {}), device, dtype)

    stats = np.full((len(d_values), len(sigmas), len(STAT_NAMES)), np.nan, dtype=np.float64)
    for di, d in enumerate(d_values):
        local_prior_cfg = dict(prior_cfg)
        local_prior_cfg["ambient_dim"] = int(d)
        prior = make_prior(local_prior_cfg, sigma_grid, device, dtype, seed + 1000 * di)
        gen = make_generator(device, seed + 2000 * di)
        x, _ = prior.sample(n_trials, gen)
        for si, sigma in enumerate(sigmas):
            sigma_t = torch.as_tensor(float(sigma), device=device, dtype=dtype)
            y = x + sigma_t * torch.randn(x.shape, device=device, dtype=dtype, generator=gen)
            sigma_hat = prior.sigma_mle(y)
            ratio = (sigma_hat / sigma_t).detach().cpu().numpy()
            entropy = prior.sigma_entropy(y).detach().cpu().numpy()
            stats[di, si] = np.asarray(
                [
                    np.mean(ratio),
                    np.median(ratio),
                    np.std(ratio),
                    np.quantile(ratio, 0.05),
                    np.quantile(ratio, 0.95),
                    np.mean(sigma_hat.detach().cpu().numpy() <= float(sigma_grid[0].detach().cpu()) * 1.000001),
                    np.mean(sigma_hat.detach().cpu().numpy() >= float(sigma_grid[-1].detach().cpu()) * 0.999999),
                    np.mean(entropy),
                ]
            )
            print(f"scale d={d} sigma={sigma:g}: median ratio={stats[di, si, 1]:.3f}")

    np.savez_compressed(
        paths["data"] / "prior_scale_results.npz",
        stats=stats,
        stat_names=np.asarray(STAT_NAMES),
        d_values=np.asarray(d_values),
        sigmas=np.asarray(sigmas),
    )
    for si, sigma in enumerate(sigmas):
        save_line_plot(
            paths["figures"] / f"scale_ratio_sigma_{sigma:g}.png",
            np.asarray(d_values),
            {
                "mean": stats[:, si, STAT_NAMES.index("mean")],
                "median": stats[:, si, STAT_NAMES.index("median")],
            },
            "ambient dimension d",
            "sigma_hat / sigma",
            f"Blind scale ratio, sigma={sigma:g}",
            logx=True,
        )
    lines = ["# Prior Scale Inference", ""]
    lines.append("| d | sigma | median ratio | std | q05 | q95 | min hit | max hit | entropy |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for di, d in enumerate(d_values):
        for si, sigma in enumerate(sigmas):
            row = stats[di, si]
            lines.append(
                f"| {int(d)} | {float(sigma):g} | {row[1]:.3f} | {row[2]:.3f} | "
                f"{row[3]:.3f} | {row[4]:.3f} | {row[5]:.3f} | {row[6]:.3f} | {row[7]:.3f} |"
            )
    write_markdown(paths["reports"] / "prior_scale_report.md", lines)
    return {"stats": stats, "paths": paths}


def main() -> None:
    parser = common_parser("Run BDDM prior scale inference tests")
    args = parser.parse_args()
    run(prepare_config(args))


if __name__ == "__main__":
    main()

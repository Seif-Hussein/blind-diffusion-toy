"""Closed-loop inverse-problem tests for blind diffusion splitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .algorithms import (
    blind_forced_diffusion,
    covariance_filtered_clean_update,
    initial_noisy_state,
    log_schedule,
    oracle_scale_nonblind_forced_diffusion,
    raw_pnp_splitting,
    scheduled_nonblind_forced_diffusion,
)
from .measurements import exact_linear_posterior, make_measurement
from .metrics import (
    measurement_mse,
    negative_log_joint,
    posterior_mean_mse,
    summarize_final_metrics,
)
from .plotting import (
    ensure_dir,
    plot_final_boxplots,
    plot_latent_trajectories,
    plot_metric_trajectories,
)
from .priors import (
    GaussianMixturePrior,
    make_ellipse_gmm,
    make_full_gaussian_control,
    make_subspace_gmm,
)


FINAL_METRIC_NAMES = [
    "mse_true",
    "measurement_mse",
    "posterior_mean_mse",
    "neg_log_joint",
    "nearest_component_distance",
    "final_sigma_hat",
]

HISTORY_METRIC_NAMES = [
    "mse_true",
    "measurement_mse",
    "posterior_mean_mse",
    "neg_log_joint",
    "sigma_used",
    "sigma_hat",
    "grad_norm",
    "cov_grad_norm",
    "raw_normal_ratio",
    "cov_normal_ratio",
]


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _make_prior(
    name: str,
    d: int,
    rng: np.random.Generator,
    sigma_grid: np.ndarray,
    curve_components: int,
) -> GaussianMixturePrior:
    if name == "subspace":
        return make_subspace_gmm(d, n_components=min(8, curve_components), rng=rng, sigma_grid=sigma_grid)
    if name == "ellipse":
        return make_ellipse_gmm(d, n_components=curve_components, rng=rng, sigma_grid=sigma_grid)
    if name == "full":
        return make_full_gaussian_control(d, rng=rng, sigma_grid=sigma_grid)
    raise ValueError(f"unknown prior: {name}")


def _method_names(include_bayes: bool) -> list[str]:
    names = [
        "scheduled_nonblind",
        "blind_mle",
    ]
    if include_bayes:
        names.append("blind_bayes")
    names.extend(
        [
            "oracle_scale",
            "raw_pnp_rho0",
            "raw_pnp_rho0p1",
            "raw_pnp_rho1",
            "cov_filtered",
        ]
    )
    return names


def _run_method(
    name: str,
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    schedule: np.ndarray,
    eta: float,
    beta: float,
    h: float,
    sigma_min: float,
    y0: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    if name == "scheduled_nonblind":
        return scheduled_nonblind_forced_diffusion(
            prior, A, y_obs, noise_std, schedule, eta=eta, h=h, beta=beta, rng=rng, y0=y0
        )
    if name == "blind_mle":
        return blind_forced_diffusion(
            prior,
            A,
            y_obs,
            noise_std,
            eta=eta,
            h=h,
            beta=beta,
            n_steps=schedule.size,
            sigma_min=sigma_min,
            rng=rng,
            y0=y0,
            mode="mle",
        )
    if name == "blind_bayes":
        return blind_forced_diffusion(
            prior,
            A,
            y_obs,
            noise_std,
            eta=eta,
            h=h,
            beta=beta,
            n_steps=schedule.size,
            sigma_min=sigma_min,
            rng=rng,
            y0=y0,
            mode="bayes",
        )
    if name == "oracle_scale":
        return oracle_scale_nonblind_forced_diffusion(
            prior,
            A,
            y_obs,
            noise_std,
            eta=eta,
            h=h,
            beta=beta,
            n_steps=schedule.size,
            sigma_min=sigma_min,
            rng=rng,
            y0=y0,
        )
    if name == "raw_pnp_rho0":
        return raw_pnp_splitting(
            prior, A, y_obs, noise_std, schedule, eta=eta, rng=rng, y0_for_init=y0, rho_factor=0.0
        )
    if name == "raw_pnp_rho0p1":
        return raw_pnp_splitting(
            prior, A, y_obs, noise_std, schedule, eta=eta, rng=rng, y0_for_init=y0, rho_factor=0.1
        )
    if name == "raw_pnp_rho1":
        return raw_pnp_splitting(
            prior, A, y_obs, noise_std, schedule, eta=eta, rng=rng, y0_for_init=y0, rho_factor=1.0
        )
    if name == "cov_filtered":
        return covariance_filtered_clean_update(
            prior, A, y_obs, noise_std, schedule, eta=eta, rng=rng, y0=y0, sigma_mode="schedule"
        )
    raise ValueError(f"unknown method: {name}")


def _fill_history_array(
    dest: np.ndarray,
    result: dict,
    x_true: np.ndarray,
    posterior_mean: np.ndarray,
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
) -> None:
    x_hist = np.asarray(result["history"]["x"], dtype=float)
    length = min(dest.shape[0], x_hist.shape[0])
    for k in range(length):
        x = x_hist[k]
        dest[k, 0] = np.mean((x - x_true) ** 2)
        dest[k, 1] = measurement_mse(x, y_obs, A)
        dest[k, 2] = posterior_mean_mse(x, posterior_mean)
        dest[k, 3] = negative_log_joint(x, prior, A, y_obs, noise_std)
    for j, key in enumerate(HISTORY_METRIC_NAMES[4:], start=4):
        values = np.asarray(result["history"][key], dtype=float)
        dest[: min(dest.shape[0], values.size), j] = values[: dest.shape[0]]


def _final_metric_vector(
    result: dict,
    x_true: np.ndarray,
    posterior_mean: np.ndarray,
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    summary = summarize_final_metrics(result["x_final"], x_true, posterior_mean, prior, A, y_obs, noise_std)
    sigma_hist = np.asarray(result["history"]["sigma_hat"], dtype=float)
    finite = sigma_hist[np.isfinite(sigma_hist)]
    final_sigma = finite[-1] if finite.size else np.nan
    return np.asarray([summary[name] for name in FINAL_METRIC_NAMES[:-1]] + [final_sigma], dtype=float)


def write_report(
    out_path: Path,
    metrics: np.ndarray,
    d_values: list[int],
    measurement_ratios: list[float],
    eta_values: list[float],
    beta_values: list[float],
    method_names: list[str],
) -> None:
    lines = ["# Closed-loop report", ""]
    pm_idx = FINAL_METRIC_NAMES.index("posterior_mean_mse")
    meas_idx = FINAL_METRIC_NAMES.index("measurement_mse")
    for di, d in enumerate(d_values):
        for ri, ratio in enumerate(measurement_ratios):
            lines.append(f"## d={d}, measurement_ratio={ratio:g}")
            for ei, eta in enumerate(eta_values):
                for bi, beta in enumerate(beta_values):
                    means = np.nanmean(metrics[di, ri, ei, bi, :, :, pm_idx], axis=0)
                    meas = np.nanmean(metrics[di, ri, ei, bi, :, :, meas_idx], axis=0)
                    best = int(np.nanargmin(means))
                    lines.append(
                        f"- eta={eta:g}, beta={beta:g}: best posterior-mean MSE is "
                        f"{method_names[best]} ({means[best]:.4e}); measurement MSE mean for that method "
                        f"is {meas[best]:.4e}."
                    )
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="toy_blind_splitting/results")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--prior", choices=["subspace", "ellipse", "full"], default="ellipse")
    parser.add_argument("--A-type", choices=["random", "mask"], default="random")
    parser.add_argument("--d-values", default="20")
    parser.add_argument("--measurement-ratios", default="0.5")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--curve-components", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--sigma-min", type=float, default=0.02)
    parser.add_argument("--sigma-max", type=float, default=1.5)
    parser.add_argument("--sigma0", type=float, default=1.2)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--eta-values", default="0.01")
    parser.add_argument("--beta-values", default="0.1")
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--skip-bayes", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.d_values = "12"
        args.measurement_ratios = "0.5"
        args.n_trials = min(args.n_trials, 2)
        args.n_steps = min(args.n_steps, 20)
        args.curve_components = min(args.curve_components, 10)
        args.grid_size = min(args.grid_size, 21)
        args.eta_values = "0.01"
        args.beta_values = "0.0"

    d_values = _parse_int_list(args.d_values)
    measurement_ratios = _parse_float_list(args.measurement_ratios)
    eta_values = _parse_float_list(args.eta_values)
    beta_values = _parse_float_list(args.beta_values)
    method_names = _method_names(include_bayes=not args.skip_bayes)

    out_dir = ensure_dir(args.out)
    figure_dir = ensure_dir(out_dir / "figures")
    data_dir = ensure_dir(out_dir / "data")

    metrics = np.full(
        (
            len(d_values),
            len(measurement_ratios),
            len(eta_values),
            len(beta_values),
            args.n_trials,
            len(method_names),
            len(FINAL_METRIC_NAMES),
        ),
        np.nan,
        dtype=float,
    )
    histories = np.full(
        (
            len(d_values),
            len(measurement_ratios),
            len(eta_values),
            len(beta_values),
            args.n_trials,
            len(method_names),
            args.n_steps + 1,
            len(HISTORY_METRIC_NAMES),
        ),
        np.nan,
        dtype=float,
    )

    first_prior = None
    first_trajectories = {}

    for di, d in enumerate(d_values):
        prior_rng = np.random.default_rng(args.seed + 1000 * di)
        sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)
        prior = _make_prior(args.prior, d, prior_rng, sigma_grid, args.curve_components)
        if first_prior is None:
            first_prior = prior
        schedule = log_schedule(args.sigma0, args.sigma_min, args.n_steps)
        for ri, measurement_ratio in enumerate(measurement_ratios):
            m = max(1, int(round(measurement_ratio * d)))
            for trial in range(args.n_trials):
                trial_seed = args.seed + 100000 * di + 10000 * ri + trial
                trial_rng = np.random.default_rng(trial_seed)
                meas = make_measurement(prior, args.A_type, m, args.noise_std, trial_rng)
                posterior = exact_linear_posterior(prior, meas["A"], meas["y_obs"], args.noise_std)
                y0 = initial_noisy_state(prior, args.sigma0, np.random.default_rng(trial_seed + 55))
                for ei, eta in enumerate(eta_values):
                    for bi, beta in enumerate(beta_values):
                        for mi, method_name in enumerate(method_names):
                            method_rng = np.random.default_rng(trial_seed + 1000 * mi + 17)
                            result = _run_method(
                                method_name,
                                prior,
                                meas["A"],
                                meas["y_obs"],
                                args.noise_std,
                                schedule,
                                eta,
                                beta,
                                args.h,
                                args.sigma_min,
                                y0,
                                method_rng,
                            )
                            metrics[di, ri, ei, bi, trial, mi] = _final_metric_vector(
                                result,
                                meas["x_true"],
                                posterior["mean"],
                                prior,
                                meas["A"],
                                meas["y_obs"],
                                args.noise_std,
                            )
                            _fill_history_array(
                                histories[di, ri, ei, bi, trial, mi],
                                result,
                                meas["x_true"],
                                posterior["mean"],
                                prior,
                                meas["A"],
                                meas["y_obs"],
                                args.noise_std,
                            )
                            if di == ri == ei == bi == trial == 0:
                                first_trajectories[method_name] = np.asarray(result["history"]["x"], dtype=float)

    np.savez_compressed(
        data_dir / "closed_loop_results.npz",
        metrics=metrics,
        histories=histories,
        d_values=np.asarray(d_values),
        measurement_ratios=np.asarray(measurement_ratios),
        eta_values=np.asarray(eta_values),
        beta_values=np.asarray(beta_values),
        method_names=np.asarray(method_names),
        final_metric_names=np.asarray(FINAL_METRIC_NAMES),
        history_metric_names=np.asarray(HISTORY_METRIC_NAMES),
    )

    base_metrics = metrics[0, 0, 0, 0]
    plot_final_boxplots(
        base_metrics,
        method_names,
        FINAL_METRIC_NAMES,
        "posterior_mean_mse",
        figure_dir / "final_posterior_mean_mse.png",
        "Final posterior mean error",
    )
    history_pm_idx = HISTORY_METRIC_NAMES.index("posterior_mean_mse")
    plot_metric_trajectories(
        {
            method_names[mi]: histories[0, 0, 0, 0, :, mi, :, history_pm_idx]
            for mi in range(len(method_names))
        },
        figure_dir / "posterior_mean_mse_trajectory.png",
        "posterior_mean_mse",
        "Posterior mean error trajectory",
    )
    history_meas_idx = HISTORY_METRIC_NAMES.index("measurement_mse")
    plot_metric_trajectories(
        {
            method_names[mi]: histories[0, 0, 0, 0, :, mi, :, history_meas_idx]
            for mi in range(len(method_names))
        },
        figure_dir / "measurement_mse_trajectory.png",
        "measurement_mse",
        "Measurement residual trajectory",
    )
    sigma_idx = HISTORY_METRIC_NAMES.index("sigma_hat")
    sigma_used_idx = HISTORY_METRIC_NAMES.index("sigma_used")
    plot_metric_trajectories(
        {
            "scheduled sigma": histories[0, 0, 0, 0, :, 0, :, sigma_used_idx],
            "scheduled inferred sigma": histories[0, 0, 0, 0, :, 0, :, sigma_idx],
            "blind mle inferred sigma": histories[0, 0, 0, 0, :, 1, :, sigma_idx],
        },
        figure_dir / "scale_trajectory.png",
        "sigma",
        "Scale trajectory",
    )
    if first_prior is not None and first_prior.metadata.get("U") is not None:
        plot_latent_trajectories(
            first_prior,
            first_trajectories,
            figure_dir / "latent_trajectories.png",
            "Example latent trajectories",
        )

    write_report(
        data_dir / "closed_loop_report.md",
        metrics,
        d_values,
        measurement_ratios,
        eta_values,
        beta_values,
        method_names,
    )
    print(f"saved closed-loop results to {data_dir / 'closed_loop_results.npz'}")


if __name__ == "__main__":
    main()


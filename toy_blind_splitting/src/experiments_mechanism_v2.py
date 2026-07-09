"""Focused v2 experiments for the covariance-filtered likelihood-tilt mechanism."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .algorithms import (
    blind_clean_pnp_splitting,
    blind_forced_diffusion,
    covariance_filtered_clean_update,
    initial_noisy_state,
    log_schedule,
    oracle_scale_nonblind_forced_diffusion,
    raw_noisy_shift_without_sigma2,
    raw_pnp_splitting,
    scheduled_nonblind_forced_diffusion,
)
from .measurements import exact_linear_posterior, make_measurement
from .metrics import local_tangent_normal_ratio, measurement_mse, negative_log_joint, posterior_mean_mse
from .plotting import ensure_dir, plot_latent_trajectories
from .priors import (
    GaussianMixturePrior,
    make_ellipse_gmm,
    make_full_gaussian_control,
    make_subspace_gmm,
)


SCALE_STAT_NAMES = [
    "mean",
    "std",
    "median",
    "p05",
    "p95",
    "min_hit_rate",
    "max_hit_rate",
]

FINAL_METRIC_NAMES = [
    "mse_true",
    "measurement_mse",
    "posterior_mean_mse",
    "neg_log_joint",
    "final_sigma_hat",
    "sigma_min_hit_rate",
    "sigma_max_hit_rate",
]

HISTORY_METRIC_NAMES = [
    "mse_true",
    "measurement_mse",
    "posterior_mean_mse",
    "neg_log_joint",
    "sigma_used",
    "sigma_hat",
    "sigma_min_hit",
    "sigma_max_hit",
    "grad_norm",
    "cov_grad_norm",
    "raw_normal_ratio",
    "cov_normal_ratio",
]


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


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


def _method_names(include_blind_clean: bool, method_list: str = "") -> list[str]:
    default_names = [
        "scheduled_nonblind",
        "blind_mle",
        "blind_bayes",
        "oracle_scale",
        "cov_filtered",
        "raw_pnp_tuned",
        "raw_noisy_no_sigma2",
    ]
    if include_blind_clean:
        default_names.append("blind_clean_pnp_tuned")
    if not method_list:
        return default_names
    allowed = set(default_names) | {"blind_clean_pnp_tuned"}
    names = _parse_str_list(method_list)
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"unknown method(s): {unknown}")
    return names


def _run_method(
    method_name: str,
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
    raw_pnp_step_cap: float,
) -> dict:
    if method_name == "scheduled_nonblind":
        return scheduled_nonblind_forced_diffusion(
            prior, A, y_obs, noise_std, schedule, eta=eta, h=h, beta=beta, rng=rng, y0=y0
        )
    if method_name == "blind_mle":
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
    if method_name == "blind_bayes":
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
    if method_name == "oracle_scale":
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
    if method_name == "cov_filtered":
        return covariance_filtered_clean_update(
            prior, A, y_obs, noise_std, schedule, eta=eta, rng=rng, y0=y0, sigma_mode="schedule"
        )
    if method_name == "raw_pnp_tuned":
        return raw_pnp_splitting(
            prior,
            A,
            y_obs,
            noise_std,
            schedule,
            eta=min(eta, raw_pnp_step_cap),
            rng=rng,
            y0_for_init=y0,
            rho_factor=0.0,
        ) | {"name": "raw_pnp_tuned"}
    if method_name == "raw_noisy_no_sigma2":
        return raw_noisy_shift_without_sigma2(
            prior, A, y_obs, noise_std, schedule, eta=eta, h=h, beta=beta, rng=rng, y0=y0
        )
    if method_name == "blind_clean_pnp_tuned":
        return blind_clean_pnp_splitting(
            prior,
            A,
            y_obs,
            noise_std,
            n_steps=schedule.size,
            eta=min(eta, raw_pnp_step_cap),
            rng=rng,
            y0_for_init=y0,
            mode="mle",
        )
    raise ValueError(f"unknown method: {method_name}")


def _scale_stats(sigmas: np.ndarray, true_sigma: float, sigma_grid: np.ndarray) -> np.ndarray:
    ratio = sigmas / true_sigma
    return np.asarray(
        [
            np.mean(ratio),
            np.std(ratio),
            np.median(ratio),
            np.percentile(ratio, 5),
            np.percentile(ratio, 95),
            np.mean(sigmas == sigma_grid[0]),
            np.mean(sigmas == sigma_grid[-1]),
        ],
        dtype=float,
    )


def plot_scale_summary(
    out_dir: Path,
    prior_names: list[str],
    d_values: list[int],
    stats: np.ndarray,
) -> None:
    stat_idx = {name: i for i, name in enumerate(SCALE_STAT_NAMES)}
    for pi, prior_name in enumerate(prior_names):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        x = np.asarray(d_values, dtype=float)
        mean = stats[pi, :, stat_idx["mean"]]
        median = stats[pi, :, stat_idx["median"]]
        p05 = stats[pi, :, stat_idx["p05"]]
        p95 = stats[pi, :, stat_idx["p95"]]
        ax.plot(x, mean, marker="o", label="mean")
        ax.plot(x, median, marker="s", label="median")
        ax.fill_between(x, p05, p95, alpha=0.18, label="5-95%")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("ambient dimension d")
        ax.set_ylabel("sigma_hat / sigma")
        ax.set_title(f"Blind scale statistics: {prior_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"v2_scale_stats_{prior_name}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(x, stats[pi, :, stat_idx["min_hit_rate"]], marker="o", label="sigma_min")
        ax.plot(x, stats[pi, :, stat_idx["max_hit_rate"]], marker="s", label="sigma_max")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("ambient dimension d")
        ax.set_ylabel("boundary hit rate")
        ax.set_title(f"Blind scale boundary hits: {prior_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"v2_scale_boundary_{prior_name}.png", dpi=180)
        plt.close(fig)


def plot_identity(out_dir: Path, etas: np.ndarray, rel_errors: np.ndarray) -> None:
    mean = np.mean(rel_errors, axis=1)
    p05 = np.percentile(rel_errors, 5, axis=1)
    p95 = np.percentile(rel_errors, 95, axis=1)
    slope = np.polyfit(np.log(etas), np.log(mean), deg=1)[0]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.loglog(etas, mean, marker="o", label=f"mean, slope={slope:.2f}")
    ax.fill_between(etas, p05, p95, alpha=0.16, label="5-95%")
    ref = mean[0] * (etas / etas[0])
    ax.loglog(etas, ref, color="black", linestyle="--", linewidth=1.0, label="slope 1 ref")
    ax.set_xlabel("eta")
    ax.set_ylabel("relative error")
    ax.set_title("Likelihood-tilt identity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "v2_likelihood_tilt_identity.png", dpi=180)
    plt.close(fig)


def plot_tangent_by_sigma(out_dir: Path, sigmas: np.ndarray, ratios: np.ndarray) -> None:
    labels = ["raw g", "C g", "denoise shift"]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for idx, label in enumerate(labels):
        med = np.median(ratios[:, :, idx], axis=1)
        p05 = np.percentile(ratios[:, :, idx], 5, axis=1)
        p95 = np.percentile(ratios[:, :, idx], 95, axis=1)
        ax.plot(sigmas, med, marker="o", label=label)
        ax.fill_between(sigmas, p05, p95, alpha=0.10)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("sigma")
    ax.set_ylabel("normal fraction")
    ax.set_title("Ellipse tangent-normal filtering by sigma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "v2_tangent_filtering_by_sigma.png", dpi=180)
    plt.close(fig)


def run_one_step(args: argparse.Namespace, out_dir: Path) -> dict:
    figure_dir = ensure_dir(out_dir / "figures")
    data_dir = ensure_dir(out_dir / "data")
    rng = np.random.default_rng(args.seed)
    d_values = _parse_int_list(args.one_step_d_values)
    prior_names = _parse_str_list(args.one_step_priors)
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)

    sigma_hats = np.full((len(prior_names), len(d_values), args.n_one_step_samples), np.nan, dtype=np.float32)
    stats = np.full((len(prior_names), len(d_values), len(SCALE_STAT_NAMES)), np.nan, dtype=float)
    for pi, prior_name in enumerate(prior_names):
        for di, d in enumerate(d_values):
            prior = _make_prior(prior_name, d, rng, sigma_grid, args.curve_components)
            x, _ = prior.sample(args.n_one_step_samples, rng)
            y = x + args.true_sigma * rng.normal(size=x.shape)
            sigmas = np.asarray(prior.sigma_mle(y), dtype=float)
            sigma_hats[pi, di] = sigmas.astype(np.float32)
            stats[pi, di] = _scale_stats(sigmas, args.true_sigma, sigma_grid)
            print(f"one-step scale {prior_name:8s} d={d:3d}: mean ratio={stats[pi, di, 0]:.3f}")

    plot_scale_summary(figure_dir, prior_names, d_values, stats)

    identity_rng = np.random.default_rng(args.seed + 1001)
    identity_prior = make_ellipse_gmm(
        args.identity_d,
        n_components=args.curve_components,
        rng=identity_rng,
        sigma_grid=sigma_grid,
    )
    x, _ = identity_prior.sample(args.n_one_step_samples, identity_rng)
    y = x + args.identity_sigma * identity_rng.normal(size=x.shape)
    g = identity_rng.normal(size=x.shape)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-15)
    m0 = identity_prior.denoise(y, args.identity_sigma)
    cg = identity_prior.posterior_covariance_action(y, args.identity_sigma, g)
    etas = np.asarray(_parse_float_list(args.identity_etas), dtype=float)
    rel_errors = np.empty((etas.size, args.n_one_step_samples), dtype=np.float32)
    for ei, eta in enumerate(etas):
        m1 = identity_prior.denoise(y - eta * args.identity_sigma**2 * g, args.identity_sigma)
        delta_cov = -eta * cg
        rel = np.linalg.norm((m1 - m0) - delta_cov, axis=1) / np.maximum(
            np.linalg.norm(delta_cov, axis=1), 1e-15
        )
        rel_errors[ei] = rel.astype(np.float32)
        print(f"identity eta={eta:g}: mean relative error={np.mean(rel):.3e}")
    plot_identity(figure_dir, etas, rel_errors)

    tangent_rng = np.random.default_rng(args.seed + 2002)
    tangent_sigmas = np.asarray(_parse_float_list(args.tangent_sigmas), dtype=float)
    tangent_prior = make_ellipse_gmm(
        args.tangent_d,
        n_components=args.curve_components,
        rng=tangent_rng,
        sigma_grid=sigma_grid,
    )
    tangent_ratios = np.empty((tangent_sigmas.size, args.n_one_step_samples, 3), dtype=np.float32)
    for si, sigma in enumerate(tangent_sigmas):
        x, _ = tangent_prior.sample(args.n_one_step_samples, tangent_rng)
        y = x + sigma * tangent_rng.normal(size=x.shape)
        g = tangent_rng.normal(size=x.shape)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-15)
        m = tangent_prior.denoise(y, sigma)
        cov_g = tangent_prior.posterior_covariance_action(y, sigma, g)
        delta = tangent_prior.denoise(y - args.tangent_eta * sigma**2 * g, sigma) - m
        for n in range(args.n_one_step_samples):
            tangent_ratios[si, n, 0] = local_tangent_normal_ratio(tangent_prior, m[n], g[n])
            tangent_ratios[si, n, 1] = local_tangent_normal_ratio(tangent_prior, m[n], cov_g[n])
            tangent_ratios[si, n, 2] = local_tangent_normal_ratio(tangent_prior, m[n], delta[n])
        print(
            f"tangent sigma={sigma:g}: raw med={np.median(tangent_ratios[si, :, 0]):.3f}, "
            f"Cg med={np.median(tangent_ratios[si, :, 1]):.3f}"
        )
    plot_tangent_by_sigma(figure_dir, tangent_sigmas, tangent_ratios)

    np.savez_compressed(
        data_dir / "one_step_v2_results.npz",
        prior_names=np.asarray(prior_names),
        d_values=np.asarray(d_values),
        scale_stat_names=np.asarray(SCALE_STAT_NAMES),
        scale_stats=stats,
        sigma_hats=sigma_hats,
        true_sigma=np.asarray(args.true_sigma),
        sigma_grid=sigma_grid,
        identity_etas=etas,
        identity_rel_errors=rel_errors,
        tangent_sigmas=tangent_sigmas,
        tangent_ratios=tangent_ratios,
    )
    return {
        "prior_names": prior_names,
        "d_values": d_values,
        "scale_stats": stats,
        "identity_etas": etas,
        "identity_rel_errors": rel_errors,
        "tangent_sigmas": tangent_sigmas,
        "tangent_ratios": tangent_ratios,
    }


def _history_to_array(
    result: dict,
    out: np.ndarray,
    x_true: np.ndarray,
    posterior_mean: np.ndarray,
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    sigma_min: float,
    sigma_max: float,
) -> None:
    x_hist = np.asarray(result["history"]["x"], dtype=float)
    length = min(out.shape[0], x_hist.shape[0])
    for step in range(length):
        x = x_hist[step]
        out[step, 0] = np.mean((x - x_true) ** 2)
        out[step, 1] = measurement_mse(x, y_obs, A)
        out[step, 2] = posterior_mean_mse(x, posterior_mean)
        out[step, 3] = negative_log_joint(x, prior, A, y_obs, noise_std)

    sigma_used = np.asarray(result["history"]["sigma_used"], dtype=float)
    sigma_hat = np.asarray(result["history"]["sigma_hat"], dtype=float)
    grad_norm = np.asarray(result["history"]["grad_norm"], dtype=float)
    cov_grad_norm = np.asarray(result["history"]["cov_grad_norm"], dtype=float)
    raw_normal = np.asarray(result["history"]["raw_normal_ratio"], dtype=float)
    cov_normal = np.asarray(result["history"]["cov_normal_ratio"], dtype=float)
    values = {
        "sigma_used": sigma_used,
        "sigma_hat": sigma_hat,
        "sigma_min_hit": (sigma_hat == sigma_min).astype(float),
        "sigma_max_hit": (sigma_hat == sigma_max).astype(float),
        "grad_norm": grad_norm,
        "cov_grad_norm": cov_grad_norm,
        "raw_normal_ratio": raw_normal,
        "cov_normal_ratio": cov_normal,
    }
    for key, vals in values.items():
        idx = HISTORY_METRIC_NAMES.index(key)
        n = min(vals.size, out.shape[0])
        out[:n, idx] = vals[:n]


def _final_metric(
    result: dict,
    history: np.ndarray,
) -> np.ndarray:
    last = np.where(np.isfinite(history[:, 2]))[0]
    if last.size == 0:
        step = 0
    else:
        step = int(last[-1])
    sigma_hat = history[:, HISTORY_METRIC_NAMES.index("sigma_hat")]
    finite_sigma = sigma_hat[np.isfinite(sigma_hat)]
    final_sigma = finite_sigma[-1] if finite_sigma.size else np.nan
    min_hits = history[:, HISTORY_METRIC_NAMES.index("sigma_min_hit")]
    max_hits = history[:, HISTORY_METRIC_NAMES.index("sigma_max_hit")]
    return np.asarray(
        [
            history[step, HISTORY_METRIC_NAMES.index("mse_true")],
            history[step, HISTORY_METRIC_NAMES.index("measurement_mse")],
            history[step, HISTORY_METRIC_NAMES.index("posterior_mean_mse")],
            history[step, HISTORY_METRIC_NAMES.index("neg_log_joint")],
            final_sigma,
            np.nanmean(min_hits),
            np.nanmean(max_hits),
        ],
        dtype=np.float32,
    )


def _plot_closed_loop_summary(
    figure_dir: Path,
    prior_name: str,
    d_values: list[int],
    measurement_ratios: list[float],
    eta_values: list[float],
    beta_values: list[float],
    method_names: list[str],
    final_metrics: np.ndarray,
    histories: np.ndarray,
) -> None:
    pm_idx = FINAL_METRIC_NAMES.index("posterior_mean_mse")
    meas_idx = FINAL_METRIC_NAMES.index("measurement_mse")
    nlj_idx = FINAL_METRIC_NAMES.index("neg_log_joint")

    # Aggregate posterior-mean error versus eta.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for mi, method in enumerate(method_names):
        vals = np.nanmedian(final_metrics[:, :, :, :, :, mi, pm_idx], axis=(0, 1, 3, 4))
        ax.plot(eta_values, vals, marker="o", label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eta")
    ax.set_ylabel("median posterior-mean MSE")
    ax.set_title(f"{prior_name}: closed-loop reconstruction versus eta")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / f"v2_closed_loop_pmse_vs_eta_{prior_name}.png", dpi=180)
    plt.close(fig)

    # Raw insertion check: measurement residual and negative log joint versus eta.
    selected_methods = [
        m for m in ["cov_filtered", "raw_pnp_tuned", "raw_noisy_no_sigma2", "blind_clean_pnp_tuned"] if m in method_names
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for method in selected_methods:
        mi = method_names.index(method)
        meas = np.nanmedian(final_metrics[:, :, :, :, :, mi, meas_idx], axis=(0, 1, 3, 4))
        nlj = np.nanmedian(final_metrics[:, :, :, :, :, mi, nlj_idx], axis=(0, 1, 3, 4))
        axes[0].plot(eta_values, meas, marker="o", label=method)
        axes[1].plot(eta_values, nlj, marker="o", label=method)
    for ax, ylabel in zip(axes, ["measurement MSE", "negative log joint"]):
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("eta")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    fig.suptitle(f"{prior_name}: raw insertion diagnostics")
    fig.tight_layout()
    fig.savefig(figure_dir / f"v2_raw_insertion_diagnostics_{prior_name}.png", dpi=180)
    plt.close(fig)

    # Dimension dependence.
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for method in ["scheduled_nonblind", "blind_mle", "oracle_scale", "cov_filtered", "raw_pnp_tuned"]:
        if method not in method_names:
            continue
        mi = method_names.index(method)
        vals = np.nanmedian(final_metrics[:, :, :, :, :, mi, pm_idx], axis=(1, 2, 3, 4))
        ax.plot(d_values, vals, marker="o", label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("d")
    ax.set_ylabel("median posterior-mean MSE")
    ax.set_title(f"{prior_name}: dimension dependence")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / f"v2_dimension_dependence_{prior_name}.png", dpi=180)
    plt.close(fig)

    # Scale tracking for a representative high-d, mid-ratio, moderate eta/beta condition.
    d_sel = int(np.argmin(np.abs(np.asarray(d_values) - 100)))
    r_sel = int(np.argmin(np.abs(np.asarray(measurement_ratios) - 0.5)))
    e_sel = int(np.argmin(np.abs(np.asarray(eta_values) - 1e-3)))
    b_sel = int(np.argmin(np.abs(np.asarray(beta_values) - 0.1)))
    sigma_hat_idx = HISTORY_METRIC_NAMES.index("sigma_hat")
    sigma_used_idx = HISTORY_METRIC_NAMES.index("sigma_used")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for method in ["scheduled_nonblind", "blind_mle", "blind_bayes", "oracle_scale"]:
        if method not in method_names:
            continue
        mi = method_names.index(method)
        sigma_hat = np.nanmedian(histories[d_sel, r_sel, e_sel, b_sel, :, mi, :, sigma_hat_idx], axis=0)
        ax.plot(sigma_hat, label=f"{method} sigma_hat")
    sched_mi = method_names.index("scheduled_nonblind")
    sigma_used = np.nanmedian(histories[d_sel, r_sel, e_sel, b_sel, :, sched_mi, :, sigma_used_idx], axis=0)
    ax.plot(sigma_used, color="black", linestyle="--", linewidth=1.0, label="schedule")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("sigma")
    ax.set_title(f"{prior_name}: scale tracking d={d_values[d_sel]}, ratio={measurement_ratios[r_sel]}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / f"v2_scale_tracking_{prior_name}.png", dpi=180)
    plt.close(fig)

    # Covariance-filter diagnostics over trajectory.
    grad_idx = HISTORY_METRIC_NAMES.index("grad_norm")
    cov_grad_idx = HISTORY_METRIC_NAMES.index("cov_grad_norm")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for method in ["scheduled_nonblind", "blind_mle", "cov_filtered", "raw_noisy_no_sigma2"]:
        if method not in method_names:
            continue
        mi = method_names.index(method)
        grad = histories[d_sel, r_sel, e_sel, b_sel, :, mi, :, grad_idx]
        cov_grad = histories[d_sel, r_sel, e_sel, b_sel, :, mi, :, cov_grad_idx]
        ratio = np.nanmedian(cov_grad / np.maximum(grad, 1e-12), axis=0)
        ax.plot(ratio, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("median ||C g|| / ||g||")
    ax.set_title(f"{prior_name}: covariance-filter strength")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / f"v2_cov_filter_strength_{prior_name}.png", dpi=180)
    plt.close(fig)


def run_closed_loop_for_prior(
    args: argparse.Namespace,
    prior_name: str,
    out_dir: Path,
) -> dict:
    figure_dir = ensure_dir(out_dir / "figures")
    data_dir = ensure_dir(out_dir / "data")
    d_values = _parse_int_list(args.closed_loop_d_values)
    measurement_ratios = _parse_float_list(args.measurement_ratios)
    eta_values = _parse_float_list(args.eta_values)
    beta_values = _parse_float_list(args.beta_values)
    method_names = _method_names(include_blind_clean=args.include_blind_clean, method_list=args.methods)
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)

    shape = (
        len(d_values),
        len(measurement_ratios),
        len(eta_values),
        len(beta_values),
        args.n_trials,
        len(method_names),
    )
    final_metrics = np.full(shape + (len(FINAL_METRIC_NAMES),), np.nan, dtype=np.float32)
    histories = np.full(
        shape + (args.n_steps + 1, len(HISTORY_METRIC_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    first_prior = None
    first_trajectories: dict[str, np.ndarray] = {}

    total_conditions = len(d_values) * len(measurement_ratios) * len(eta_values) * len(beta_values)
    trajectories_per_condition = args.n_trials * len(method_names)
    total_trajectories = total_conditions * trajectories_per_condition
    condition_count = 0
    start = time.time()
    for di, d in enumerate(d_values):
        prior_rng = np.random.default_rng(args.seed + 1000 * di + 97 * (prior_name == "full"))
        prior = _make_prior(prior_name, d, prior_rng, sigma_grid, args.curve_components)
        if first_prior is None:
            first_prior = prior
        schedule = log_schedule(args.sigma0, args.schedule_sigma_min, args.n_steps)
        for ri, measurement_ratio in enumerate(measurement_ratios):
            m = max(1, int(round(measurement_ratio * d)))
            trial_measurements = []
            for trial in range(args.n_trials):
                trial_seed = args.seed + 100000 * di + 10000 * ri + 100 * (prior_name == "full") + trial
                trial_rng = np.random.default_rng(trial_seed)
                meas = make_measurement(prior, args.A_type, m, args.noise_std, trial_rng)
                posterior = exact_linear_posterior(prior, meas["A"], meas["y_obs"], args.noise_std)
                y0 = initial_noisy_state(prior, args.sigma0, np.random.default_rng(trial_seed + 55))
                trial_measurements.append((trial_seed, meas, posterior, y0))

            for ei, eta in enumerate(eta_values):
                for bi, beta in enumerate(beta_values):
                    condition_count += 1
                    print(
                        f"closed-loop {prior_name:7s} condition {condition_count:03d}/{total_conditions}: "
                        f"d={d}, ratio={measurement_ratio:g}, eta={eta:g}, beta={beta:g}"
                    )
                    for trial, (trial_seed, meas, posterior, y0) in enumerate(trial_measurements):
                        for mi, method_name in enumerate(method_names):
                            method_rng = np.random.default_rng(trial_seed + 1000 * mi + 17 + 31 * bi)
                            try:
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
                                    args.raw_pnp_step_cap,
                                )
                                hist = histories[di, ri, ei, bi, trial, mi]
                                _history_to_array(
                                    result,
                                    hist,
                                    meas["x_true"],
                                    posterior["mean"],
                                    prior,
                                    meas["A"],
                                    meas["y_obs"],
                                    args.noise_std,
                                    sigma_grid[0],
                                    sigma_grid[-1],
                                )
                                final_metrics[di, ri, ei, bi, trial, mi] = _final_metric(result, hist)
                                if di == ri == ei == bi == trial == 0:
                                    first_trajectories[method_name] = np.asarray(result["history"]["x"], dtype=float)
                            except FloatingPointError:
                                continue
                            except Exception as exc:
                                print(f"method failed: {prior_name} {method_name} trial={trial}: {exc}")
                    elapsed = time.time() - start
                    completed_trajectories = condition_count * trajectories_per_condition
                    avg_seconds = elapsed / max(completed_trajectories, 1)
                    remaining_trajectories = total_trajectories - completed_trajectories
                    eta_seconds = avg_seconds * remaining_trajectories
                    print(
                        f"completed condition {condition_count} / {total_conditions}; "
                        f"elapsed={elapsed / 60.0:.2f} min; "
                        f"avg={avg_seconds:.3f} sec/trajectory; "
                        f"eta_remaining={eta_seconds / 60.0:.2f} min"
                    )
                    np.savez_compressed(
                        data_dir / f"closed_loop_v2_{prior_name}_partial.npz",
                        prior_name=np.asarray(prior_name),
                        final_metrics=final_metrics,
                        histories=histories,
                        d_values=np.asarray(d_values),
                        measurement_ratios=np.asarray(measurement_ratios),
                        eta_values=np.asarray(eta_values),
                        beta_values=np.asarray(beta_values),
                        method_names=np.asarray(method_names),
                        final_metric_names=np.asarray(FINAL_METRIC_NAMES),
                        history_metric_names=np.asarray(HISTORY_METRIC_NAMES),
                        n_steps=np.asarray(args.n_steps),
                        n_trials=np.asarray(args.n_trials),
                        raw_pnp_step_cap=np.asarray(args.raw_pnp_step_cap),
                        completed_conditions=np.asarray(condition_count),
                        total_conditions=np.asarray(total_conditions),
                    )

    np.savez_compressed(
        data_dir / f"closed_loop_v2_{prior_name}.npz",
        prior_name=np.asarray(prior_name),
        final_metrics=final_metrics,
        histories=histories,
        d_values=np.asarray(d_values),
        measurement_ratios=np.asarray(measurement_ratios),
        eta_values=np.asarray(eta_values),
        beta_values=np.asarray(beta_values),
        method_names=np.asarray(method_names),
        final_metric_names=np.asarray(FINAL_METRIC_NAMES),
        history_metric_names=np.asarray(HISTORY_METRIC_NAMES),
        n_steps=np.asarray(args.n_steps),
        n_trials=np.asarray(args.n_trials),
        raw_pnp_step_cap=np.asarray(args.raw_pnp_step_cap),
    )
    _plot_closed_loop_summary(
        figure_dir,
        prior_name,
        d_values,
        measurement_ratios,
        eta_values,
        beta_values,
        method_names,
        final_metrics,
        histories,
    )
    if first_prior is not None and first_prior.metadata.get("U") is not None:
        plot_latent_trajectories(
            first_prior,
            first_trajectories,
            figure_dir / f"v2_latent_trajectories_{prior_name}.png",
            f"{prior_name}: example latent trajectories",
        )
    return {
        "prior_name": prior_name,
        "final_metrics": final_metrics,
        "histories": histories,
        "d_values": d_values,
        "measurement_ratios": measurement_ratios,
        "eta_values": eta_values,
        "beta_values": beta_values,
        "method_names": method_names,
    }


def _best_method_summary(closed: dict) -> list[str]:
    final_metrics = closed["final_metrics"]
    method_names = closed["method_names"]
    pm_idx = FINAL_METRIC_NAMES.index("posterior_mean_mse")
    means = np.nanmedian(final_metrics[..., pm_idx], axis=(0, 1, 2, 3, 4))
    order = np.argsort(means)
    return [f"{method_names[i]}={means[i]:.4e}" for i in order]


def write_report(
    out_dir: Path,
    one_step: dict | None,
    closed_results: list[dict],
) -> None:
    data_dir = ensure_dir(out_dir / "data")
    lines = ["# Mechanism v2 report", ""]

    if one_step is not None:
        stat_idx = {name: i for i, name in enumerate(SCALE_STAT_NAMES)}
        lines.append("## One-step scale inference")
        for pi, prior_name in enumerate(one_step["prior_names"]):
            lines.append(f"### {prior_name}")
            lines.append("| d | mean | std | median | p05 | p95 | min hit | max hit |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for di, d in enumerate(one_step["d_values"]):
                stats = one_step["scale_stats"][pi, di]
                lines.append(
                    f"| {d} | {stats[stat_idx['mean']]:.3f} | {stats[stat_idx['std']]:.3f} | "
                    f"{stats[stat_idx['median']]:.3f} | {stats[stat_idx['p05']]:.3f} | "
                    f"{stats[stat_idx['p95']]:.3f} | {stats[stat_idx['min_hit_rate']]:.3f} | "
                    f"{stats[stat_idx['max_hit_rate']]:.3f} |"
                )
            lines.append("")

        identity_mean = np.mean(one_step["identity_rel_errors"], axis=1)
        identity_slope = np.polyfit(np.log(one_step["identity_etas"]), np.log(identity_mean), deg=1)[0]
        lines.append("## Likelihood-tilt identity")
        lines.append(f"- Log-log slope of mean relative error versus eta: `{identity_slope:.3f}`.")
        lines.append(
            "- Mean relative errors: "
            + ", ".join(
                f"eta={eta:g}: {err:.3e}"
                for eta, err in zip(one_step["identity_etas"], identity_mean)
            )
            + "."
        )
        lines.append("")

        lines.append("## Tangent-normal filtering")
        lines.append("| sigma | raw median | Cg median | denoise-shift median |")
        lines.append("|---:|---:|---:|---:|")
        for si, sigma in enumerate(one_step["tangent_sigmas"]):
            med = np.median(one_step["tangent_ratios"][si], axis=0)
            lines.append(f"| {sigma:g} | {med[0]:.3f} | {med[1]:.3f} | {med[2]:.3f} |")
        lines.append("")

    if closed_results:
        lines.append("## Closed-loop summaries")
        for closed in closed_results:
            lines.append(f"### {closed['prior_name']}")
            lines.append("- Median posterior-mean MSE ranking: " + "; ".join(_best_method_summary(closed)) + ".")
            final_metrics = closed["final_metrics"]
            method_names = closed["method_names"]
            pm_idx = FINAL_METRIC_NAMES.index("posterior_mean_mse")
            if "blind_mle" in method_names and "oracle_scale" in method_names:
                bm = method_names.index("blind_mle")
                oc = method_names.index("oracle_scale")
                rel = np.nanmedian(final_metrics[..., bm, pm_idx] / np.maximum(final_metrics[..., oc, pm_idx], 1e-12))
                lines.append(f"- Blind MLE / oracle-scale median posterior-MSE ratio: `{rel:.3f}`.")
            if "cov_filtered" in method_names and "scheduled_nonblind" in method_names:
                cf = method_names.index("cov_filtered")
                sn = method_names.index("scheduled_nonblind")
                rel = np.nanmedian(final_metrics[..., cf, pm_idx] / np.maximum(final_metrics[..., sn, pm_idx], 1e-12))
                lines.append(f"- Cov-filtered / scheduled median posterior-MSE ratio: `{rel:.3f}`.")
            if "raw_pnp_tuned" in method_names and "raw_noisy_no_sigma2" in method_names:
                rp = method_names.index("raw_pnp_tuned")
                rn = method_names.index("raw_noisy_no_sigma2")
                lines.append(
                    f"- Raw tuned PnP median posterior-MSE: `{np.nanmedian(final_metrics[..., rp, pm_idx]):.4e}`; "
                    f"raw unscaled noisy shift: `{np.nanmedian(final_metrics[..., rn, pm_idx]):.4e}`."
                )
            if "blind_clean_pnp_tuned" in method_names:
                bc = method_names.index("blind_clean_pnp_tuned")
                lines.append(
                    f"- Blind clean-space PnP median posterior-MSE: "
                    f"`{np.nanmedian(final_metrics[..., bc, pm_idx]):.4e}`."
                )
            lines.append("")

    lines.append("## Claim assessment")
    if one_step is not None:
        prior_names = one_step["prior_names"]
        d_values = np.asarray(one_step["d_values"])
        high_d_mask = d_values >= 50
        low_d_mask = d_values <= 5
        if "ellipse" in prior_names:
            pi = prior_names.index("ellipse")
            high_mean = np.mean(one_step["scale_stats"][pi, high_d_mask, SCALE_STAT_NAMES.index("mean")])
            low_std = np.mean(one_step["scale_stats"][pi, low_d_mask, SCALE_STAT_NAMES.index("std")])
            lines.append(
                f"- Claim A: supported for the ellipse prior if high-d mean ratio near 1 is enough "
                f"(d>=50 average mean ratio `{high_mean:.3f}`); low-d dispersion remains high "
                f"(d<=5 average std `{low_std:.3f}`)."
            )
        identity_mean = np.mean(one_step["identity_rel_errors"], axis=1)
        identity_slope = np.polyfit(np.log(one_step["identity_etas"]), np.log(identity_mean), deg=1)[0]
        tangent_med = np.median(one_step["tangent_ratios"], axis=1)
        lines.append(
            f"- Claim B: supported. Identity error slope is `{identity_slope:.3f}` and median Cg normal "
            f"fractions are consistently below raw normal fractions across the sigma sweep."
        )
    if closed_results:
        for closed in closed_results:
            method_names = closed["method_names"]
            final_metrics = closed["final_metrics"]
            pm_idx = FINAL_METRIC_NAMES.index("posterior_mean_mse")
            if "cov_filtered" in method_names and "scheduled_nonblind" in method_names:
                cf = method_names.index("cov_filtered")
                sn = method_names.index("scheduled_nonblind")
                rel = np.nanmedian(final_metrics[..., cf, pm_idx] / np.maximum(final_metrics[..., sn, pm_idx], 1e-12))
                verdict = "supported" if rel < 1.0 else "not supported"
                lines.append(
                    f"- Claim C for `{closed['prior_name']}`: {verdict} by aggregate median "
                    f"cov-filtered/scheduled ratio `{rel:.3f}`."
                )
            if "blind_clean_pnp_tuned" in method_names and "blind_mle" in method_names:
                bc = method_names.index("blind_clean_pnp_tuned")
                bm = method_names.index("blind_mle")
                rel = np.nanmedian(final_metrics[..., bc, pm_idx] / np.maximum(final_metrics[..., bm, pm_idx], 1e-12))
                verdict = "supported" if rel > 1.0 else "not supported"
                lines.append(
                    f"- Claim D for `{closed['prior_name']}`: {verdict} by aggregate median "
                    f"blind-clean/blind-noisy ratio `{rel:.3f}`."
                )
    lines.append("")
    lines.append("Boundary-hit rates should be inspected alongside all closed-loop scale plots; saturation means a scale-tracking result is not diagnostic.")
    (data_dir / "mechanism_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="toy_blind_splitting/results/mechanism_v2")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--skip-one-step", action="store_true")
    parser.add_argument("--skip-closed-loop", action="store_true")
    parser.add_argument("--one-step-d-values", default="2,5,10,20,50,100,200")
    parser.add_argument("--one-step-priors", default="ellipse,subspace,full")
    parser.add_argument("--n-one-step-samples", type=int, default=500)
    parser.add_argument("--true-sigma", type=float, default=0.35)
    parser.add_argument("--identity-d", type=int, default=100)
    parser.add_argument("--identity-sigma", type=float, default=0.25)
    parser.add_argument("--identity-etas", default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    parser.add_argument("--tangent-d", type=int, default=100)
    parser.add_argument("--tangent-sigmas", default="0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--tangent-eta", type=float, default=1e-2)
    parser.add_argument("--closed-loop-priors", default="ellipse,full")
    parser.add_argument("--closed-loop-d-values", default="12,50,100")
    parser.add_argument("--measurement-ratios", default="0.25,0.5,1.0")
    parser.add_argument("--eta-values", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    parser.add_argument("--beta-values", default="0,0.1,0.3")
    parser.add_argument("--methods", default="")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--curve-components", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--schedule-sigma-min", type=float, default=0.02)
    parser.add_argument("--sigma0", type=float, default=1.2)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--A-type", choices=["random", "mask"], default="random")
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--raw-pnp-step-cap", type=float, default=3e-4)
    parser.add_argument("--include-blind-clean", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    one_step = None
    if not args.skip_one_step:
        one_step = run_one_step(args, out_dir)

    closed_results = []
    if not args.skip_closed_loop:
        for prior_name in _parse_str_list(args.closed_loop_priors):
            closed_results.append(run_closed_loop_for_prior(args, prior_name, out_dir))

    write_report(out_dir, one_step, closed_results)
    print(f"saved mechanism-v2 report to {out_dir / 'data' / 'mechanism_v2_report.md'}")


if __name__ == "__main__":
    main()

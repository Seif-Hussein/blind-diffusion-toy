"""Posterior-correction oracle diagnostics for the Gaussian-mixture toy.

Run from the repository root, for example:

    python -m posterior_bddm_oracle.src.experiments_posterior_oracle --quick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from toy_blind_splitting.src.algorithms import log_schedule
from toy_blind_splitting.src.measurements import exact_linear_posterior, make_measurement
from toy_blind_splitting.src.metrics import measurement_mse
from toy_blind_splitting.src.plotting import ensure_dir
from toy_blind_splitting.src.priors import (
    GaussianMixturePrior,
    make_ellipse_gmm,
    make_full_gaussian_control,
    make_subspace_gmm,
)

from .oracle_tools import (
    EPS,
    SplitSpec,
    component_kl,
    component_weight_estimate,
    covariance_error,
    make_split_specs,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    posterior_as_gmm,
    posterior_mean_error,
    relative_error,
    safe_cosine,
    sigma_diagnostics,
    split_force,
    tangent_normal_fraction,
)


ONE_STEP_METRIC_NAMES = [
    "rel_error",
    "cosine",
    "c_star_norm",
    "c_split_norm",
    "c_star_normal_fraction",
    "c_split_normal_fraction",
]

ONE_STEP_SCALE_NAMES = [
    "sigma_hat_p",
    "sigma_hat_pi",
    "sigma_entropy_p",
    "sigma_entropy_pi",
    "p_sigma_min_hit",
    "p_sigma_max_hit",
    "pi_sigma_min_hit",
    "pi_sigma_max_hit",
]

CONDITION_NAMES = ["d", "intrinsic_k", "measurement_ratio", "m", "noise_std"]

CLOSED_METHOD_NAMES = [
    "posterior_oracle",
    "prior_exact_cstar",
    "blind_prior_split",
    "blind_naive_force",
    "scheduled_split",
    "posterior_scale_split",
    "raw_pnp_hqs",
]

CLOSED_HISTORY_NAMES = [
    "posterior_mean_mse",
    "measurement_mse",
    "sigma_used",
    "sigma_schedule",
    "sigma_hat_p",
    "sigma_hat_pi",
    "sigma_entropy_p",
    "sigma_entropy_pi",
    "p_sigma_min_hit",
    "p_sigma_max_hit",
    "pi_sigma_min_hit",
    "pi_sigma_max_hit",
    "corr_rel_error",
    "corr_cosine",
    "c_star_norm",
    "c_split_norm",
]

CLOSED_FINAL_NAMES = [
    "posterior_mean_mse",
    "measurement_mse",
    "final_sigma_hat_p",
    "final_sigma_hat_pi",
    "median_corr_rel_error",
    "median_corr_cosine",
]

CLOSED_AGGREGATE_NAMES = [
    "sample_mean_mse",
    "mean_posterior_mean_mse",
    "posterior_covariance_error",
    "component_weight_kl",
    "mean_measurement_mse",
    "median_corr_rel_error",
    "median_corr_cosine",
    "p_sigma_min_hit_rate",
    "p_sigma_max_hit_rate",
    "pi_sigma_min_hit_rate",
    "pi_sigma_max_hit_rate",
]


def _make_prior(
    name: str,
    d: int,
    rng: np.random.Generator,
    sigma_grid: np.ndarray,
    curve_components: int,
) -> GaussianMixturePrior:
    if name == "ellipse":
        return make_ellipse_gmm(d, n_components=curve_components, rng=rng, sigma_grid=sigma_grid)
    if name == "subspace":
        return make_subspace_gmm(d, n_components=min(8, curve_components), rng=rng, sigma_grid=sigma_grid)
    if name == "full":
        return make_full_gaussian_control(d, rng=rng, sigma_grid=sigma_grid)
    raise ValueError(f"unknown prior: {name}")


def _condition_table(
    d_values: list[int],
    measurement_ratios: list[float],
    noise_stds: list[float],
    intrinsic_k: int,
) -> np.ndarray:
    rows = []
    for d in d_values:
        for ratio in measurement_ratios:
            m = max(1, int(round(ratio * d)))
            for noise_std in noise_stds:
                rows.append((d, intrinsic_k, ratio, m, noise_std))
    return np.asarray(rows, dtype=float)


def _split_correction_batch(
    prior: GaussianMixturePrior,
    spec: SplitSpec,
    y_state: np.ndarray,
    sigma: float,
    m_prior: np.ndarray,
    y_obs: np.ndarray,
    A: np.ndarray,
    noise_std: float,
    eta: float,
) -> np.ndarray:
    force, _ = split_force(spec, m_prior, y_obs, A, noise_std, w=None)
    shifted = y_state - float(eta) * float(sigma) ** 2 * force
    return prior.denoise(shifted, float(sigma)) - m_prior


def _fill_one_step_metrics(
    out: np.ndarray,
    prior: GaussianMixturePrior,
    m_prior: np.ndarray,
    c_star: np.ndarray,
    c_split: np.ndarray,
) -> None:
    out[:, ONE_STEP_METRIC_NAMES.index("rel_error")] = relative_error(c_split, c_star)
    out[:, ONE_STEP_METRIC_NAMES.index("cosine")] = safe_cosine(c_split, c_star)
    out[:, ONE_STEP_METRIC_NAMES.index("c_star_norm")] = np.linalg.norm(c_star, axis=1)
    out[:, ONE_STEP_METRIC_NAMES.index("c_split_norm")] = np.linalg.norm(c_split, axis=1)
    out[:, ONE_STEP_METRIC_NAMES.index("c_star_normal_fraction")] = tangent_normal_fraction(
        prior,
        m_prior,
        c_star,
    )
    out[:, ONE_STEP_METRIC_NAMES.index("c_split_normal_fraction")] = tangent_normal_fraction(
        prior,
        m_prior,
        c_split,
    )


def _write_one_step_report(
    path: Path,
    conditions: np.ndarray,
    sigmas: np.ndarray,
    etas: np.ndarray,
    split_names: list[str],
    split_metrics: np.ndarray,
    scale_metrics: np.ndarray,
) -> None:
    rel_idx = ONE_STEP_METRIC_NAMES.index("rel_error")
    cos_idx = ONE_STEP_METRIC_NAMES.index("cosine")
    p_min_idx = ONE_STEP_SCALE_NAMES.index("p_sigma_min_hit")
    p_max_idx = ONE_STEP_SCALE_NAMES.index("p_sigma_max_hit")
    pi_min_idx = ONE_STEP_SCALE_NAMES.index("pi_sigma_min_hit")
    pi_max_idx = ONE_STEP_SCALE_NAMES.index("pi_sigma_max_hit")

    lines = ["# Posterior Correction Oracle: One-Step Diagnostics", ""]
    lines.append("## Aggregate Split Correction Alignment")
    lines.append("| split | median rel error | median cosine |")
    lines.append("|---|---:|---:|")
    for mi, name in enumerate(split_names):
        rel = np.nanmedian(split_metrics[:, :, :, mi, :, rel_idx])
        cos = np.nanmedian(split_metrics[:, :, :, mi, :, cos_idx])
        lines.append(f"| `{name}` | {rel:.4e} | {cos:.4f} |")
    lines.append("")

    lines.append("## Best Eta By Condition")
    lines.append("| d | ratio | noise | sigma | split | best eta | median rel error | median cosine |")
    lines.append("|---:|---:|---:|---:|---|---:|---:|---:|")
    for ci, cond in enumerate(conditions):
        d, _k, ratio, _m, noise_std = cond
        for si, sigma in enumerate(sigmas):
            for mi, name in enumerate(split_names):
                rel_by_eta = np.nanmedian(split_metrics[ci, si, :, mi, :, rel_idx], axis=1)
                best = int(np.nanargmin(rel_by_eta))
                cos = np.nanmedian(split_metrics[ci, si, best, mi, :, cos_idx])
                lines.append(
                    f"| {int(d)} | {ratio:g} | {noise_std:g} | {sigma:g} | `{name}` | "
                    f"{etas[best]:g} | {rel_by_eta[best]:.4e} | {cos:.4f} |"
                )
    lines.append("")

    lines.append("## Scale Boundary Diagnostics")
    lines.append("| d | ratio | noise | sigma | prior boundary hit | posterior boundary hit |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for ci, cond in enumerate(conditions):
        d, _k, ratio, _m, noise_std = cond
        for si, sigma in enumerate(sigmas):
            p_hit = np.nanmean(scale_metrics[ci, si, :, p_min_idx] + scale_metrics[ci, si, :, p_max_idx])
            pi_hit = np.nanmean(scale_metrics[ci, si, :, pi_min_idx] + scale_metrics[ci, si, :, pi_max_idx])
            lines.append(f"| {int(d)} | {ratio:g} | {noise_std:g} | {sigma:g} | {p_hit:.3f} | {pi_hit:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_step(args: argparse.Namespace, out_dir: Path) -> dict:
    data_dir = ensure_dir(out_dir / "data")
    d_values = parse_int_list(args.d_values)
    measurement_ratios = parse_float_list(args.measurement_ratios)
    noise_stds = parse_float_list(args.noise_stds)
    sigmas = np.asarray(parse_float_list(args.sigmas), dtype=float)
    etas = np.asarray(parse_float_list(args.eta_values), dtype=float)
    split_specs = make_split_specs(
        parse_str_list(args.split_methods),
        parse_float_list(args.pdhg_gammas),
        parse_float_list(args.hqs_taus),
    )
    split_names = [spec.name for spec in split_specs]
    conditions = _condition_table(d_values, measurement_ratios, noise_stds, args.intrinsic_k)

    split_metrics = np.full(
        (
            conditions.shape[0],
            sigmas.size,
            etas.size,
            len(split_specs),
            args.n_samples,
            len(ONE_STEP_METRIC_NAMES),
        ),
        np.nan,
        dtype=np.float32,
    )
    scale_metrics = np.full(
        (conditions.shape[0], sigmas.size, args.n_samples, len(ONE_STEP_SCALE_NAMES)),
        np.nan,
        dtype=np.float32,
    )

    start = time.perf_counter()
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)
    for ci, cond in enumerate(conditions):
        d, _k, ratio, m, noise_std = cond
        d = int(d)
        m = int(m)
        noise_std = float(noise_std)
        rng = np.random.default_rng(args.seed + 1000 * ci)
        prior = _make_prior(args.prior, d, rng, sigma_grid, args.curve_components)
        prior.precompute_sigmas(sigmas)
        meas = make_measurement(prior, args.A_type, m, noise_std, rng)
        posterior = exact_linear_posterior(prior, meas["A"], meas["y_obs"], noise_std)
        posterior_prior = posterior_as_gmm(prior, posterior, sigma_grid)
        posterior_prior.precompute_sigmas(sigmas)

        print(
            f"one-step condition {ci + 1}/{conditions.shape[0]}: "
            f"d={d}, ratio={ratio:g}, noise={noise_std:g}",
            flush=True,
        )
        for si, sigma in enumerate(sigmas):
            x, _ids = posterior_prior.sample(args.n_samples, rng)
            y_state = x + float(sigma) * rng.normal(size=x.shape)
            m_prior = prior.denoise(y_state, float(sigma))
            m_post = posterior_prior.denoise(y_state, float(sigma))
            c_star = m_post - m_prior

            sigma_p, entropy_p, boundary_p = sigma_diagnostics(prior, y_state)
            sigma_pi, entropy_pi, boundary_pi = sigma_diagnostics(posterior_prior, y_state)
            scale_metrics[ci, si, :, ONE_STEP_SCALE_NAMES.index("sigma_hat_p")] = sigma_p
            scale_metrics[ci, si, :, ONE_STEP_SCALE_NAMES.index("sigma_hat_pi")] = sigma_pi
            scale_metrics[ci, si, :, ONE_STEP_SCALE_NAMES.index("sigma_entropy_p")] = entropy_p
            scale_metrics[ci, si, :, ONE_STEP_SCALE_NAMES.index("sigma_entropy_pi")] = entropy_pi
            scale_metrics[ci, si, :, ONE_STEP_SCALE_NAMES.index("p_sigma_min_hit")] = boundary_p[:, 0]
            scale_metrics[ci, si, :, ONE_STEP_SCALE_NAMES.index("p_sigma_max_hit")] = boundary_p[:, 1]
            scale_metrics[ci, si, :, ONE_STEP_SCALE_NAMES.index("pi_sigma_min_hit")] = boundary_pi[:, 0]
            scale_metrics[ci, si, :, ONE_STEP_SCALE_NAMES.index("pi_sigma_max_hit")] = boundary_pi[:, 1]

            for ei, eta in enumerate(etas):
                for mi, spec in enumerate(split_specs):
                    c_split = _split_correction_batch(
                        prior,
                        spec,
                        y_state,
                        float(sigma),
                        m_prior,
                        meas["y_obs"],
                        meas["A"],
                        noise_std,
                        float(eta),
                    )
                    _fill_one_step_metrics(split_metrics[ci, si, ei, mi], prior, m_prior, c_star, c_split)
        elapsed = time.perf_counter() - start
        print(f"  completed in {elapsed / 60.0:.2f} min total", flush=True)

    np.savez_compressed(
        data_dir / "one_step_posterior_oracle.npz",
        conditions=conditions,
        condition_names=np.asarray(CONDITION_NAMES),
        sigmas=sigmas,
        etas=etas,
        split_names=np.asarray(split_names),
        split_metrics=split_metrics,
        split_metric_names=np.asarray(ONE_STEP_METRIC_NAMES),
        scale_metrics=scale_metrics,
        scale_metric_names=np.asarray(ONE_STEP_SCALE_NAMES),
        prior_name=np.asarray(args.prior),
        n_samples=np.asarray(args.n_samples),
    )
    _write_one_step_report(
        data_dir / "one_step_posterior_oracle_report.md",
        conditions,
        sigmas,
        etas,
        split_names,
        split_metrics,
        scale_metrics,
    )
    print(f"saved one-step diagnostics to {data_dir / 'one_step_posterior_oracle.npz'}", flush=True)
    return {
        "conditions": conditions,
        "sigmas": sigmas,
        "etas": etas,
        "split_names": split_names,
        "split_metrics": split_metrics,
        "scale_metrics": scale_metrics,
    }


def _nanmedian(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _history_row(
    x: np.ndarray,
    state: np.ndarray,
    sigma_used: float,
    sigma_schedule: float,
    prior: GaussianMixturePrior,
    posterior_prior: GaussianMixturePrior,
    posterior_mean: np.ndarray,
    A: np.ndarray,
    y_obs: np.ndarray,
    c_split: np.ndarray | None = None,
    c_star: np.ndarray | None = None,
) -> np.ndarray:
    row = np.full(len(CLOSED_HISTORY_NAMES), np.nan, dtype=np.float32)
    row[CLOSED_HISTORY_NAMES.index("posterior_mean_mse")] = np.mean((x - posterior_mean) ** 2)
    row[CLOSED_HISTORY_NAMES.index("measurement_mse")] = measurement_mse(x, y_obs, A)
    row[CLOSED_HISTORY_NAMES.index("sigma_used")] = sigma_used
    row[CLOSED_HISTORY_NAMES.index("sigma_schedule")] = sigma_schedule

    sigma_p, entropy_p, boundary_p = sigma_diagnostics(prior, state)
    sigma_pi, entropy_pi, boundary_pi = sigma_diagnostics(posterior_prior, state)
    row[CLOSED_HISTORY_NAMES.index("sigma_hat_p")] = sigma_p[0]
    row[CLOSED_HISTORY_NAMES.index("sigma_hat_pi")] = sigma_pi[0]
    row[CLOSED_HISTORY_NAMES.index("sigma_entropy_p")] = entropy_p[0]
    row[CLOSED_HISTORY_NAMES.index("sigma_entropy_pi")] = entropy_pi[0]
    row[CLOSED_HISTORY_NAMES.index("p_sigma_min_hit")] = boundary_p[0, 0]
    row[CLOSED_HISTORY_NAMES.index("p_sigma_max_hit")] = boundary_p[0, 1]
    row[CLOSED_HISTORY_NAMES.index("pi_sigma_min_hit")] = boundary_pi[0, 0]
    row[CLOSED_HISTORY_NAMES.index("pi_sigma_max_hit")] = boundary_pi[0, 1]

    if c_split is not None and c_star is not None:
        row[CLOSED_HISTORY_NAMES.index("corr_rel_error")] = relative_error(
            c_split[None, :],
            c_star[None, :],
        )[0]
        row[CLOSED_HISTORY_NAMES.index("corr_cosine")] = safe_cosine(
            c_split[None, :],
            c_star[None, :],
        )[0]
        row[CLOSED_HISTORY_NAMES.index("c_star_norm")] = np.linalg.norm(c_star)
        row[CLOSED_HISTORY_NAMES.index("c_split_norm")] = np.linalg.norm(c_split)
    return row


def _closed_split_correction(
    prior: GaussianMixturePrior,
    posterior_prior: GaussianMixturePrior,
    spec: SplitSpec,
    y_state: np.ndarray,
    sigma: float,
    y_obs: np.ndarray,
    A: np.ndarray,
    noise_std: float,
    eta: float,
    w: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    m_prior = prior.denoise(y_state, float(sigma))
    m_post = posterior_prior.denoise(y_state, float(sigma))
    c_star = m_post - m_prior
    force, w_next = split_force(spec, m_prior, y_obs, A, noise_std, w=w)
    shifted = y_state - float(eta) * float(sigma) ** 2 * force
    c_split = prior.denoise(shifted, float(sigma)) - m_prior
    return m_prior, c_star, c_split, force, w_next


def _run_closed_method(
    method: str,
    prior: GaussianMixturePrior,
    posterior_prior: GaussianMixturePrior,
    posterior: dict,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    y0: np.ndarray,
    schedule: np.ndarray,
    split_spec: SplitSpec,
    eta: float,
    h: float,
    beta: float,
    raw_pnp_tau: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    d = prior.d
    hist = np.full((schedule.size + 1, len(CLOSED_HISTORY_NAMES)), np.nan, dtype=np.float32)
    Y = y0.copy()
    w: np.ndarray | None = None

    if method == "raw_pnp_hqs":
        x = prior.denoise(Y, float(schedule[0]))
        hist[0] = _history_row(
            x,
            x,
            float(schedule[0]),
            float(schedule[0]),
            prior,
            posterior_prior,
            posterior["mean"],
            A,
            y_obs,
        )
        for step, sigma_sched in enumerate(schedule, start=1):
            # Reuse the HQS split force as a clean-space PnP/HQS prox.
            force, _ = split_force(
                SplitSpec("raw_hqs", "hqs", hqs_tau=raw_pnp_tau),
                x,
                y_obs,
                A,
                noise_std,
            )
            prox_x = x - raw_pnp_tau * force
            x = prior.denoise(prox_x, float(sigma_sched))
            hist[step] = _history_row(
                x,
                x,
                float(sigma_sched),
                float(sigma_sched),
                prior,
                posterior_prior,
                posterior["mean"],
                A,
                y_obs,
            )
        return x, hist

    x0 = posterior_prior.denoise(Y, float(schedule[0])) if method == "posterior_oracle" else prior.denoise(
        Y,
        float(schedule[0]),
    )
    hist[0] = _history_row(
        x0,
        Y,
        float(schedule[0]),
        float(schedule[0]),
        prior,
        posterior_prior,
        posterior["mean"],
        A,
        y_obs,
    )

    x_current = x0
    for step, sigma_sched in enumerate(schedule, start=1):
        sigma_sched = float(sigma_sched)
        c_star = None
        c_split = None
        if method == "posterior_oracle":
            sigma = sigma_sched
            m_post = posterior_prior.denoise(Y, sigma)
            drift = m_post - Y
            x_eval_after = lambda state: posterior_prior.denoise(state, sigma)
        elif method == "prior_exact_cstar":
            sigma = sigma_sched
            m_prior = prior.denoise(Y, sigma)
            m_post = posterior_prior.denoise(Y, sigma)
            c_star = m_post - m_prior
            c_split = c_star
            drift = (m_prior - Y) + c_star
            x_eval_after = lambda state: posterior_prior.denoise(state, sigma)
        elif method == "blind_prior_split":
            sigma = float(prior.sigma_mle(Y))
            m_prior, c_star, c_split, force, w = _closed_split_correction(
                prior,
                posterior_prior,
                split_spec,
                Y,
                sigma,
                y_obs,
                A,
                noise_std,
                eta,
                w,
            )
            drift = (m_prior - Y) + c_split
            x_eval_after = lambda state: prior.denoise(state, sigma)
        elif method == "blind_naive_force":
            sigma = float(prior.sigma_mle(Y))
            m_prior, c_star, c_split, force, w = _closed_split_correction(
                prior,
                posterior_prior,
                split_spec,
                Y,
                sigma,
                y_obs,
                A,
                noise_std,
                eta,
                w,
            )
            Y_tilde = Y - float(eta) * sigma**2 * force
            x_tilde = prior.denoise(Y_tilde, sigma)
            noise_scale = np.sqrt(max(0.0, 2.0 * h * beta * sigma**2))
            Y = Y_tilde + h * (x_tilde - Y_tilde) + noise_scale * rng.normal(size=d)
            x_current = prior.denoise(Y, sigma)
            hist[step] = _history_row(
                x_current,
                Y,
                sigma,
                sigma_sched,
                prior,
                posterior_prior,
                posterior["mean"],
                A,
                y_obs,
                c_split=c_split,
                c_star=c_star,
            )
            continue
        elif method == "scheduled_split":
            sigma = sigma_sched
            m_prior, c_star, c_split, force, w = _closed_split_correction(
                prior,
                posterior_prior,
                split_spec,
                Y,
                sigma,
                y_obs,
                A,
                noise_std,
                eta,
                w,
            )
            drift = (m_prior - Y) + c_split
            x_eval_after = lambda state: prior.denoise(state, sigma)
        elif method == "posterior_scale_split":
            sigma = float(posterior_prior.sigma_mle(Y))
            m_prior, c_star, c_split, force, w = _closed_split_correction(
                prior,
                posterior_prior,
                split_spec,
                Y,
                sigma,
                y_obs,
                A,
                noise_std,
                eta,
                w,
            )
            drift = (m_prior - Y) + c_split
            x_eval_after = lambda state: prior.denoise(state, sigma)
        else:
            raise ValueError(f"unknown closed-loop method: {method}")

        noise_scale = np.sqrt(max(0.0, 2.0 * h * beta * float(sigma) ** 2))
        Y = Y + h * drift + noise_scale * rng.normal(size=d)
        x_current = x_eval_after(Y)
        hist[step] = _history_row(
            x_current,
            Y,
            float(sigma),
            sigma_sched,
            prior,
            posterior_prior,
            posterior["mean"],
            A,
            y_obs,
            c_split=c_split,
            c_star=c_star,
        )
    return x_current, hist


def _final_metric_from_history(final_x: np.ndarray, hist: np.ndarray, posterior: dict, A: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    sigma_p = hist[:, CLOSED_HISTORY_NAMES.index("sigma_hat_p")]
    sigma_pi = hist[:, CLOSED_HISTORY_NAMES.index("sigma_hat_pi")]
    corr_rel = hist[:, CLOSED_HISTORY_NAMES.index("corr_rel_error")]
    corr_cos = hist[:, CLOSED_HISTORY_NAMES.index("corr_cosine")]
    finite_p = sigma_p[np.isfinite(sigma_p)]
    finite_pi = sigma_pi[np.isfinite(sigma_pi)]
    return np.asarray(
        [
            np.mean((final_x - posterior["mean"]) ** 2),
            measurement_mse(final_x, y_obs, A),
            finite_p[-1] if finite_p.size else np.nan,
            finite_pi[-1] if finite_pi.size else np.nan,
            _nanmedian(corr_rel),
            _nanmedian(corr_cos),
        ],
        dtype=np.float32,
    )


def _aggregate_closed_condition(
    final_samples: np.ndarray,
    final_metrics: np.ndarray,
    histories: np.ndarray,
    posterior: dict,
    posterior_prior: GaussianMixturePrior,
) -> np.ndarray:
    out = np.full((len(CLOSED_METHOD_NAMES), len(CLOSED_AGGREGATE_NAMES)), np.nan, dtype=np.float32)
    pm_idx = CLOSED_FINAL_NAMES.index("posterior_mean_mse")
    meas_idx = CLOSED_FINAL_NAMES.index("measurement_mse")
    corr_rel_idx = CLOSED_FINAL_NAMES.index("median_corr_rel_error")
    corr_cos_idx = CLOSED_FINAL_NAMES.index("median_corr_cosine")
    for mi in range(len(CLOSED_METHOD_NAMES)):
        samples = final_samples[:, mi, :]
        omega_hat = component_weight_estimate(posterior_prior, samples)
        out[mi, CLOSED_AGGREGATE_NAMES.index("sample_mean_mse")] = posterior_mean_error(
            samples,
            posterior["mean"],
        )
        out[mi, CLOSED_AGGREGATE_NAMES.index("mean_posterior_mean_mse")] = np.nanmean(final_metrics[:, mi, pm_idx])
        out[mi, CLOSED_AGGREGATE_NAMES.index("posterior_covariance_error")] = covariance_error(
            samples,
            posterior["covariance"],
        )
        out[mi, CLOSED_AGGREGATE_NAMES.index("component_weight_kl")] = component_kl(
            omega_hat,
            posterior["component_weights"],
        )
        out[mi, CLOSED_AGGREGATE_NAMES.index("mean_measurement_mse")] = np.nanmean(final_metrics[:, mi, meas_idx])
        out[mi, CLOSED_AGGREGATE_NAMES.index("median_corr_rel_error")] = _nanmedian(final_metrics[:, mi, corr_rel_idx])
        out[mi, CLOSED_AGGREGATE_NAMES.index("median_corr_cosine")] = _nanmedian(final_metrics[:, mi, corr_cos_idx])
        for source, target in [
            ("p_sigma_min_hit", "p_sigma_min_hit_rate"),
            ("p_sigma_max_hit", "p_sigma_max_hit_rate"),
            ("pi_sigma_min_hit", "pi_sigma_min_hit_rate"),
            ("pi_sigma_max_hit", "pi_sigma_max_hit_rate"),
        ]:
            out[mi, CLOSED_AGGREGATE_NAMES.index(target)] = np.nanmean(
                histories[:, mi, :, CLOSED_HISTORY_NAMES.index(source)]
            )
    return out


def _write_closed_report(
    path: Path,
    conditions: np.ndarray,
    aggregate_metrics: np.ndarray,
    final_samples_diff_ab: np.ndarray,
    split_spec: SplitSpec,
) -> None:
    sm_idx = CLOSED_AGGREGATE_NAMES.index("sample_mean_mse")
    pm_idx = CLOSED_AGGREGATE_NAMES.index("mean_posterior_mean_mse")
    cov_idx = CLOSED_AGGREGATE_NAMES.index("posterior_covariance_error")
    kl_idx = CLOSED_AGGREGATE_NAMES.index("component_weight_kl")
    rel_idx = CLOSED_AGGREGATE_NAMES.index("median_corr_rel_error")
    cos_idx = CLOSED_AGGREGATE_NAMES.index("median_corr_cosine")

    lines = ["# Posterior Correction Oracle: Closed-Loop Hierarchy", ""]
    lines.append(f"- Split correction used by candidate methods: `{split_spec.name}`.")
    lines.append(
        "- Decomposition check `posterior_oracle` vs `prior_exact_cstar` final sample MSE by condition: "
        + ", ".join(f"{v:.3e}" for v in final_samples_diff_ab)
        + "."
    )
    lines.append("")
    for ci, cond in enumerate(conditions):
        d, _k, ratio, _m, noise_std = cond
        lines.append(f"## d={int(d)}, ratio={ratio:g}, noise={noise_std:g}")
        lines.append(
            "| method | sample mean MSE | mean posterior MSE | cov error | weight KL | corr rel | corr cos |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        order = np.argsort(aggregate_metrics[ci, :, pm_idx])
        for mi in order:
            vals = aggregate_metrics[ci, mi]
            lines.append(
                f"| `{CLOSED_METHOD_NAMES[mi]}` | {vals[sm_idx]:.4e} | {vals[pm_idx]:.4e} | "
                f"{vals[cov_idx]:.4e} | {vals[kl_idx]:.4e} | {vals[rel_idx]:.4e} | {vals[cos_idx]:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_closed_loop(args: argparse.Namespace, out_dir: Path) -> dict:
    data_dir = ensure_dir(out_dir / "data")
    d_values = parse_int_list(args.closed_d_values)
    measurement_ratios = parse_float_list(args.closed_measurement_ratios)
    noise_stds = parse_float_list(args.closed_noise_stds)
    eta_values = parse_float_list(args.closed_eta_values)
    if len(eta_values) != 1:
        raise ValueError("closed-loop runner expects exactly one eta in --closed-eta-values")
    eta = float(eta_values[0])
    beta = float(args.closed_beta)
    conditions = _condition_table(d_values, measurement_ratios, noise_stds, args.intrinsic_k)
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)
    schedule = log_schedule(args.sigma0, args.schedule_sigma_min, args.n_steps)
    split_spec = make_split_specs([args.closed_split], [args.closed_pdhg_gamma], [args.closed_hqs_tau])[0]

    final_metrics = np.full(
        (conditions.shape[0], args.n_trials, len(CLOSED_METHOD_NAMES), len(CLOSED_FINAL_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    histories = np.full(
        (
            conditions.shape[0],
            args.n_trials,
            len(CLOSED_METHOD_NAMES),
            args.n_steps + 1,
            len(CLOSED_HISTORY_NAMES),
        ),
        np.nan,
        dtype=np.float32,
    )
    aggregate_metrics = np.full(
        (conditions.shape[0], len(CLOSED_METHOD_NAMES), len(CLOSED_AGGREGATE_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    final_samples_diff_ab = np.full(conditions.shape[0], np.nan, dtype=np.float32)

    start = time.perf_counter()
    for ci, cond in enumerate(conditions):
        d, _k, ratio, m, noise_std = cond
        d = int(d)
        m = int(m)
        noise_std = float(noise_std)
        rng = np.random.default_rng(args.seed + 50000 + ci)
        prior = _make_prior(args.prior, d, rng, sigma_grid, args.curve_components)
        prior.precompute_sigma_grid()
        prior.precompute_sigmas(schedule)
        meas = make_measurement(prior, args.A_type, m, noise_std, rng)
        posterior = exact_linear_posterior(prior, meas["A"], meas["y_obs"], noise_std)
        posterior_prior = posterior_as_gmm(prior, posterior, sigma_grid)
        posterior_prior.precompute_sigma_grid()
        posterior_prior.precompute_sigmas(schedule)

        print(
            f"closed-loop condition {ci + 1}/{conditions.shape[0]}: "
            f"d={d}, ratio={ratio:g}, noise={noise_std:g}, eta={eta:g}",
            flush=True,
        )
        final_samples = np.full((args.n_trials, len(CLOSED_METHOD_NAMES), d), np.nan, dtype=np.float32)
        for trial in range(args.n_trials):
            trial_seed = args.seed + 100000 * ci + trial
            y0 = prior.mean + args.sigma0 * np.random.default_rng(trial_seed + 19).normal(size=d)
            for mi, method in enumerate(CLOSED_METHOD_NAMES):
                final_x, hist = _run_closed_method(
                    method,
                    prior,
                    posterior_prior,
                    posterior,
                    meas["A"],
                    meas["y_obs"],
                    noise_std,
                    y0,
                    schedule,
                    split_spec,
                    eta,
                    args.h,
                    beta,
                    args.raw_pnp_tau,
                    np.random.default_rng(trial_seed + 999),
                )
                final_samples[trial, mi] = final_x.astype(np.float32)
                histories[ci, trial, mi] = hist
                final_metrics[ci, trial, mi] = _final_metric_from_history(
                    final_x,
                    hist,
                    posterior,
                    meas["A"],
                    meas["y_obs"],
                )
        a_idx = CLOSED_METHOD_NAMES.index("posterior_oracle")
        b_idx = CLOSED_METHOD_NAMES.index("prior_exact_cstar")
        final_samples_diff_ab[ci] = np.mean((final_samples[:, a_idx] - final_samples[:, b_idx]) ** 2)
        aggregate_metrics[ci] = _aggregate_closed_condition(
            final_samples,
            final_metrics[ci],
            histories[ci],
            posterior,
            posterior_prior,
        )
        elapsed = time.perf_counter() - start
        print(f"  completed in {elapsed / 60.0:.2f} min total", flush=True)

    np.savez_compressed(
        data_dir / "closed_loop_posterior_oracle.npz",
        conditions=conditions,
        condition_names=np.asarray(CONDITION_NAMES),
        method_names=np.asarray(CLOSED_METHOD_NAMES),
        final_metrics=final_metrics,
        final_metric_names=np.asarray(CLOSED_FINAL_NAMES),
        histories=histories,
        history_metric_names=np.asarray(CLOSED_HISTORY_NAMES),
        aggregate_metrics=aggregate_metrics,
        aggregate_metric_names=np.asarray(CLOSED_AGGREGATE_NAMES),
        posterior_oracle_exact_cstar_final_mse=final_samples_diff_ab,
        schedule=schedule,
        eta=np.asarray(eta),
        beta=np.asarray(beta),
        split_name=np.asarray(split_spec.name),
    )
    _write_closed_report(
        data_dir / "closed_loop_posterior_oracle_report.md",
        conditions,
        aggregate_metrics,
        final_samples_diff_ab,
        split_spec,
    )
    print(f"saved closed-loop hierarchy to {data_dir / 'closed_loop_posterior_oracle.npz'}", flush=True)
    return {
        "conditions": conditions,
        "final_metrics": final_metrics,
        "histories": histories,
        "aggregate_metrics": aggregate_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="posterior_bddm_oracle/results")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-one-step", action="store_true")
    parser.add_argument("--skip-closed-loop", action="store_true")

    parser.add_argument("--prior", choices=["ellipse", "subspace", "full"], default="ellipse")
    parser.add_argument("--A-type", choices=["random", "mask"], default="random")
    parser.add_argument("--intrinsic-k", type=int, default=2)
    parser.add_argument("--curve-components", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=3.0)

    parser.add_argument("--d-values", default="2,10,50,100,500")
    parser.add_argument("--measurement-ratios", default="0.5")
    parser.add_argument("--noise-stds", default="0.08")
    parser.add_argument("--sigmas", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--eta-values", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--split-methods", default="gradient,pdhg,hqs")
    parser.add_argument("--pdhg-gammas", default="100")
    parser.add_argument("--hqs-taus", default="1e-2")

    parser.add_argument("--closed-d-values", default="10,50")
    parser.add_argument("--closed-measurement-ratios", default="0.5")
    parser.add_argument("--closed-noise-stds", default="0.08")
    parser.add_argument("--closed-eta-values", default="1e-3")
    parser.add_argument("--closed-beta", type=float, default=0.0)
    parser.add_argument("--closed-split", choices=["gradient", "pdhg", "hqs"], default="gradient")
    parser.add_argument("--closed-pdhg-gamma", type=float, default=100.0)
    parser.add_argument("--closed-hqs-tau", type=float, default=1e-2)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--sigma0", type=float, default=1.2)
    parser.add_argument("--schedule-sigma-min", type=float, default=0.02)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--raw-pnp-tau", type=float, default=1e-2)

    args = parser.parse_args()
    if args.quick:
        args.d_values = "2,10"
        args.sigmas = "0.1,0.3"
        args.eta_values = "1e-3,1e-2"
        args.n_samples = min(args.n_samples, 32)
        args.curve_components = min(args.curve_components, 8)
        args.grid_size = min(args.grid_size, 25)
        args.split_methods = "gradient,pdhg,hqs"
        args.closed_d_values = "10"
        args.n_trials = min(args.n_trials, 3)
        args.n_steps = min(args.n_steps, 8)
        args.closed_eta_values = "1e-3"
        args.closed_beta = 0.0

    out_dir = ensure_dir(args.out)
    if not args.skip_one_step:
        run_one_step(args, out_dir)
    if not args.skip_closed_loop:
        run_closed_loop(args, out_dir)
    print(f"posterior oracle diagnostics complete; outputs under {out_dir / 'data'}", flush=True)


if __name__ == "__main__":
    main()




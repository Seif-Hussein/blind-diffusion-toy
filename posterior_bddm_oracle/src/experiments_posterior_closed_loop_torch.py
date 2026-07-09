"""CUDA closed-loop posterior-BDDM hierarchy for the Gaussian-mixture toy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .torch_oracle_tools import (
    cosine_similarity,
    exact_linear_posterior_gmm,
    hqs_force,
    make_device,
    make_ellipse_gmm_torch,
    make_full_gaussian_torch,
    make_generator,
    make_measurement,
    make_operator,
    parse_float_list,
    parse_int_list,
    relative_error,
    split_force,
    sync,
)


METHOD_NAMES = [
    "posterior_oracle",
    "prior_exact_cstar",
    "blind_split",
    "blind_naive_force",
    "scheduled_split",
    "posterior_scale_split",
    "raw_hqs",
]

HISTORY_NAMES = [
    "posterior_mean_mse",
    "measurement_mse",
    "sigma_used",
    "sigma_schedule",
    "sigma_hat_prior",
    "sigma_hat_posterior",
    "prior_boundary_hit",
    "posterior_boundary_hit",
    "corr_rel_error",
    "corr_cosine",
    "c_star_norm",
    "c_split_norm",
]

FINAL_NAMES = [
    "posterior_mean_mse",
    "measurement_mse",
    "final_sigma_hat_prior",
    "final_sigma_hat_posterior",
    "median_corr_rel_error",
    "median_corr_cosine",
    "prior_boundary_hit_rate",
    "posterior_boundary_hit_rate",
]

AGGREGATE_NAMES = [
    "sample_mean_mse",
    "mean_posterior_mean_mse",
    "posterior_covariance_error",
    "component_weight_kl",
    "mean_measurement_mse",
    "median_corr_rel_error",
    "median_corr_cosine",
    "prior_boundary_hit_rate",
    "posterior_boundary_hit_rate",
]

CONDITION_NAMES = ["d", "intrinsic_k", "measurement_ratio", "m", "noise_std", "eta"]


def _make_prior(prior_name: str, d: int, components: int, sigma_grid, device, dtype, seed: int):
    if prior_name == "ellipse":
        return make_ellipse_gmm_torch(d, components, sigma_grid, device, dtype, seed)
    if prior_name == "full":
        return make_full_gaussian_torch(d, 1.0, sigma_grid, device, dtype)
    raise ValueError(f"unknown prior: {prior_name}")


def _condition_table(
    d_values: list[int],
    measurement_ratios: list[float],
    noise_stds: list[float],
    eta_values: list[float],
    intrinsic_k: int,
) -> np.ndarray:
    rows = []
    for d in d_values:
        for ratio in measurement_ratios:
            m = max(1, int(round(float(ratio) * d)))
            for noise_std in noise_stds:
                for eta in eta_values:
                    rows.append((d, intrinsic_k, ratio, m, noise_std, eta))
    return np.asarray(rows, dtype=float)


def _log_schedule(sigma0: float, sigma_min: float, n_steps: int, device, dtype) -> torch.Tensor:
    return torch.tensor(np.geomspace(sigma0, sigma_min, int(n_steps)), device=device, dtype=dtype)


def _denoise_at_sigmas(prior, y: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    if sigmas.ndim == 0:
        return prior.denoise(y, sigmas)
    out = torch.empty_like(y)
    for sigma in torch.unique(sigmas):
        mask = sigmas == sigma
        out[mask] = prior.denoise(y[mask], sigma)
    return out


def _measurement_mse(x: torch.Tensor, y_obs: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    residual = x @ A.T - y_obs[None, :]
    return torch.mean(residual.square(), dim=1)


def _component_kl(estimated: np.ndarray, target: np.ndarray) -> float:
    eps = 1e-15
    estimated = np.maximum(estimated, eps)
    target = np.maximum(target, eps)
    estimated = estimated / np.sum(estimated)
    target = target / np.sum(target)
    return float(np.sum(estimated * (np.log(estimated) - np.log(target))))


def _covariance_error(samples: np.ndarray, target_cov: np.ndarray) -> float:
    if samples.shape[0] <= 1:
        return float("nan")
    sample_cov = np.cov(samples, rowvar=False, bias=False)
    return float(np.linalg.norm(sample_cov - target_cov, ord="fro") / max(np.linalg.norm(target_cov, ord="fro"), 1e-15))


def _nanmedian(values: np.ndarray) -> float | np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else float("nan")
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    for i in range(values.shape[0]):
        finite = values[i, np.isfinite(values[i])]
        if finite.size:
            out[i] = np.median(finite)
    return out


def _state_scale(prior, posterior_prior, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_prior = prior.sigma_mle(state)
    sigma_post = posterior_prior.sigma_mle(state)
    prior_hit = (
        (sigma_prior <= prior.sigma_grid[0] * 1.000001)
        | (sigma_prior >= prior.sigma_grid[-1] * 0.999999)
    ).to(state.dtype)
    post_hit = (
        (sigma_post <= posterior_prior.sigma_grid[0] * 1.000001)
        | (sigma_post >= posterior_prior.sigma_grid[-1] * 0.999999)
    ).to(state.dtype)
    return sigma_prior, sigma_post, prior_hit, post_hit


def _history_row(
    x: torch.Tensor,
    state: torch.Tensor,
    sigma_used: torch.Tensor,
    sigma_schedule: torch.Tensor,
    prior,
    posterior_prior,
    posterior_mean: torch.Tensor,
    y_obs: torch.Tensor,
    A: torch.Tensor,
    c_split: torch.Tensor | None = None,
    c_star: torch.Tensor | None = None,
) -> torch.Tensor:
    n = x.shape[0]
    row = torch.full((n, len(HISTORY_NAMES)), torch.nan, device=x.device, dtype=x.dtype)
    row[:, HISTORY_NAMES.index("posterior_mean_mse")] = torch.mean((x - posterior_mean[None, :]).square(), dim=1)
    row[:, HISTORY_NAMES.index("measurement_mse")] = _measurement_mse(x, y_obs, A)
    sigma_used_vec = sigma_used.expand(n) if sigma_used.ndim == 0 else sigma_used
    sigma_sched_vec = sigma_schedule.expand(n) if sigma_schedule.ndim == 0 else sigma_schedule
    row[:, HISTORY_NAMES.index("sigma_used")] = sigma_used_vec
    row[:, HISTORY_NAMES.index("sigma_schedule")] = sigma_sched_vec
    sigma_prior, sigma_post, prior_hit, post_hit = _state_scale(prior, posterior_prior, state)
    row[:, HISTORY_NAMES.index("sigma_hat_prior")] = sigma_prior
    row[:, HISTORY_NAMES.index("sigma_hat_posterior")] = sigma_post
    row[:, HISTORY_NAMES.index("prior_boundary_hit")] = prior_hit
    row[:, HISTORY_NAMES.index("posterior_boundary_hit")] = post_hit
    if c_split is not None and c_star is not None:
        row[:, HISTORY_NAMES.index("corr_rel_error")] = relative_error(c_split, c_star)
        row[:, HISTORY_NAMES.index("corr_cosine")] = cosine_similarity(c_split, c_star)
        row[:, HISTORY_NAMES.index("c_star_norm")] = torch.linalg.norm(c_star, dim=1)
        row[:, HISTORY_NAMES.index("c_split_norm")] = torch.linalg.norm(c_split, dim=1)
    return row


def _split_correction(
    prior,
    posterior_prior,
    Y: torch.Tensor,
    sigmas: torch.Tensor,
    split_name: str,
    y_obs: torch.Tensor,
    A: torch.Tensor,
    noise_std: float,
    eta: float,
    pdhg_gamma: float,
    hqs_tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m_prior = _denoise_at_sigmas(prior, Y, sigmas)
    m_post = _denoise_at_sigmas(posterior_prior, Y, sigmas)
    c_star = m_post - m_prior
    force = split_force(split_name, m_prior, y_obs, A, noise_std, pdhg_gamma, hqs_tau)
    sigma_vec = sigmas.expand(Y.shape[0]) if sigmas.ndim == 0 else sigmas
    shifted = Y - float(eta) * sigma_vec[:, None].square() * force
    c_split = _denoise_at_sigmas(prior, shifted, sigmas) - m_prior
    return m_prior, c_star, c_split, force


def _run_method(
    method: str,
    prior,
    posterior_prior,
    posterior: dict[str, torch.Tensor],
    y_obs: torch.Tensor,
    A: torch.Tensor,
    noise_std: float,
    y0: torch.Tensor,
    schedule: torch.Tensor,
    split_name: str,
    eta: float,
    h: float,
    beta: float,
    raw_hqs_tau: float,
    pdhg_gamma: float,
    hqs_tau: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, np.ndarray]:
    n, d = y0.shape
    hist = np.full((n, schedule.numel() + 1, len(HISTORY_NAMES)), np.nan, dtype=np.float32)
    Y = y0.clone()

    if method == "posterior_oracle":
        x = posterior_prior.denoise(Y, schedule[0])
    else:
        x = prior.denoise(Y, schedule[0])
    hist[:, 0] = _history_row(
        x, Y, schedule[0], schedule[0], prior, posterior_prior, posterior["mean"], y_obs, A
    ).detach().cpu().numpy().astype(np.float32)

    if method == "raw_hqs":
        for step, sigma_sched in enumerate(schedule, start=1):
            force = hqs_force(x, y_obs, A, noise_std, raw_hqs_tau)
            x = prior.denoise(x - float(raw_hqs_tau) * force, sigma_sched)
            hist[:, step] = _history_row(
                x, x, sigma_sched, sigma_sched, prior, posterior_prior, posterior["mean"], y_obs, A
            ).detach().cpu().numpy().astype(np.float32)
        return x, hist

    for step, sigma_sched in enumerate(schedule, start=1):
        c_star = None
        c_split = None
        if method == "posterior_oracle":
            sigmas = sigma_sched
            m_post = posterior_prior.denoise(Y, sigmas)
            drift = m_post - Y
            x_eval_prior = posterior_prior
        elif method == "prior_exact_cstar":
            sigmas = sigma_sched
            m_prior = prior.denoise(Y, sigmas)
            m_post = posterior_prior.denoise(Y, sigmas)
            c_star = m_post - m_prior
            c_split = c_star
            drift = (m_prior - Y) + c_star
            x_eval_prior = posterior_prior
        elif method == "blind_split":
            sigmas = prior.sigma_mle(Y)
            m_prior, c_star, c_split, _force = _split_correction(
                prior, posterior_prior, Y, sigmas, split_name, y_obs, A, noise_std, eta, pdhg_gamma, hqs_tau
            )
            drift = (m_prior - Y) + c_split
            x_eval_prior = prior
        elif method == "blind_naive_force":
            sigmas = prior.sigma_mle(Y)
            m_prior, c_star, c_split, force = _split_correction(
                prior, posterior_prior, Y, sigmas, split_name, y_obs, A, noise_std, eta, pdhg_gamma, hqs_tau
            )
            Y_tilde = Y - float(eta) * sigmas[:, None].square() * force
            x_tilde = _denoise_at_sigmas(prior, Y_tilde, sigmas)
            noise = torch.zeros_like(Y)
            if beta > 0.0:
                noise = torch.sqrt(2.0 * float(h) * float(beta) * sigmas[:, None].square()) * torch.randn(
                    Y.shape, device=Y.device, dtype=Y.dtype, generator=gen
                )
            Y = Y_tilde + float(h) * (x_tilde - Y_tilde) + noise
            x = _denoise_at_sigmas(prior, Y, sigmas)
            hist[:, step] = _history_row(
                x, Y, sigmas, sigma_sched, prior, posterior_prior, posterior["mean"], y_obs, A, c_split, c_star
            ).detach().cpu().numpy().astype(np.float32)
            continue
        elif method == "scheduled_split":
            sigmas = sigma_sched
            m_prior, c_star, c_split, _force = _split_correction(
                prior, posterior_prior, Y, sigmas, split_name, y_obs, A, noise_std, eta, pdhg_gamma, hqs_tau
            )
            drift = (m_prior - Y) + c_split
            x_eval_prior = prior
        elif method == "posterior_scale_split":
            sigmas = posterior_prior.sigma_mle(Y)
            m_prior, c_star, c_split, _force = _split_correction(
                prior, posterior_prior, Y, sigmas, split_name, y_obs, A, noise_std, eta, pdhg_gamma, hqs_tau
            )
            drift = (m_prior - Y) + c_split
            x_eval_prior = prior
        else:
            raise ValueError(f"unknown method: {method}")

        sigma_vec = sigmas.expand(n) if sigmas.ndim == 0 else sigmas
        noise = torch.zeros_like(Y)
        if beta > 0.0:
            noise = torch.sqrt(2.0 * float(h) * float(beta) * sigma_vec[:, None].square()) * torch.randn(
                (n, d), device=Y.device, dtype=Y.dtype, generator=gen
            )
        Y = Y + float(h) * drift + noise
        x = _denoise_at_sigmas(x_eval_prior, Y, sigmas)
        hist[:, step] = _history_row(
            x, Y, sigma_vec, sigma_sched, prior, posterior_prior, posterior["mean"], y_obs, A, c_split, c_star
        ).detach().cpu().numpy().astype(np.float32)
    return x, hist


def _final_from_history(final_x: torch.Tensor, hist: np.ndarray, posterior: dict[str, torch.Tensor], y_obs, A) -> np.ndarray:
    out = np.full((final_x.shape[0], len(FINAL_NAMES)), np.nan, dtype=np.float32)
    final_np = final_x.detach().cpu().numpy().astype(np.float32)
    posterior_mean = posterior["mean"].detach().cpu().numpy().astype(np.float32)
    residual = final_x @ A.T - y_obs[None, :]
    out[:, FINAL_NAMES.index("posterior_mean_mse")] = np.mean((final_np - posterior_mean[None, :]) ** 2, axis=1)
    out[:, FINAL_NAMES.index("measurement_mse")] = torch.mean(residual.square(), dim=1).detach().cpu().numpy()
    out[:, FINAL_NAMES.index("final_sigma_hat_prior")] = hist[:, -1, HISTORY_NAMES.index("sigma_hat_prior")]
    out[:, FINAL_NAMES.index("final_sigma_hat_posterior")] = hist[:, -1, HISTORY_NAMES.index("sigma_hat_posterior")]
    out[:, FINAL_NAMES.index("median_corr_rel_error")] = _nanmedian(
        hist[:, :, HISTORY_NAMES.index("corr_rel_error")]
    )
    out[:, FINAL_NAMES.index("median_corr_cosine")] = _nanmedian(
        hist[:, :, HISTORY_NAMES.index("corr_cosine")]
    )
    out[:, FINAL_NAMES.index("prior_boundary_hit_rate")] = np.nanmean(
        hist[:, :, HISTORY_NAMES.index("prior_boundary_hit")], axis=1
    )
    out[:, FINAL_NAMES.index("posterior_boundary_hit_rate")] = np.nanmean(
        hist[:, :, HISTORY_NAMES.index("posterior_boundary_hit")], axis=1
    )
    return out


def _aggregate(
    final_samples: np.ndarray,
    final_metrics: np.ndarray,
    histories: np.ndarray,
    posterior: dict[str, torch.Tensor],
    posterior_prior,
) -> np.ndarray:
    out = np.full((len(METHOD_NAMES), len(AGGREGATE_NAMES)), np.nan, dtype=np.float32)
    post_mean = posterior["mean"].detach().cpu().numpy().astype(np.float32)
    post_cov = posterior["covariance"].detach().cpu().numpy().astype(np.float32)
    target_weights = posterior["component_weights"].detach().cpu().numpy().astype(np.float64)
    for mi in range(len(METHOD_NAMES)):
        samples = final_samples[:, mi]
        resp = posterior_prior.posterior_component_weights(
            torch.as_tensor(samples, device=posterior_prior.device, dtype=posterior_prior.dtype),
            torch.as_tensor(1e-6, device=posterior_prior.device, dtype=posterior_prior.dtype),
        )
        omega_hat = torch.mean(resp, dim=0).detach().cpu().numpy().astype(np.float64)
        out[mi, AGGREGATE_NAMES.index("sample_mean_mse")] = np.mean((np.mean(samples, axis=0) - post_mean) ** 2)
        out[mi, AGGREGATE_NAMES.index("mean_posterior_mean_mse")] = np.nanmean(
            final_metrics[:, mi, FINAL_NAMES.index("posterior_mean_mse")]
        )
        out[mi, AGGREGATE_NAMES.index("posterior_covariance_error")] = _covariance_error(samples, post_cov)
        out[mi, AGGREGATE_NAMES.index("component_weight_kl")] = _component_kl(omega_hat, target_weights)
        out[mi, AGGREGATE_NAMES.index("mean_measurement_mse")] = np.nanmean(
            final_metrics[:, mi, FINAL_NAMES.index("measurement_mse")]
        )
        out[mi, AGGREGATE_NAMES.index("median_corr_rel_error")] = _nanmedian(
            final_metrics[:, mi, FINAL_NAMES.index("median_corr_rel_error")]
        )
        out[mi, AGGREGATE_NAMES.index("median_corr_cosine")] = _nanmedian(
            final_metrics[:, mi, FINAL_NAMES.index("median_corr_cosine")]
        )
        out[mi, AGGREGATE_NAMES.index("prior_boundary_hit_rate")] = np.nanmean(
            histories[:, mi, :, HISTORY_NAMES.index("prior_boundary_hit")]
        )
        out[mi, AGGREGATE_NAMES.index("posterior_boundary_hit_rate")] = np.nanmean(
            histories[:, mi, :, HISTORY_NAMES.index("posterior_boundary_hit")]
        )
    return out


def _write_report(path: Path, payload: dict) -> None:
    aggregate = payload["aggregate_metrics"]
    conditions = payload["conditions"]
    decomp = payload["posterior_oracle_exact_cstar_final_mse"]
    pm = AGGREGATE_NAMES.index("mean_posterior_mean_mse")
    sample_mean = AGGREGATE_NAMES.index("sample_mean_mse")
    cov = AGGREGATE_NAMES.index("posterior_covariance_error")
    kl = AGGREGATE_NAMES.index("component_weight_kl")
    rel = AGGREGATE_NAMES.index("median_corr_rel_error")
    cos = AGGREGATE_NAMES.index("median_corr_cosine")

    lines = ["# CUDA Closed-Loop Posterior-BDDM Hierarchy", ""]
    lines.append(f"Device: `{payload['device']}`")
    lines.append(f"Split correction: `{payload['split_name']}`")
    lines.append(
        "Posterior oracle vs prior+exact-cstar final MSE by condition: "
        + ", ".join(f"{v:.3e}" for v in decomp)
        + "."
    )
    lines.append(
        "Point-estimate metrics and posterior-distribution metrics should be read together; "
        "a method can have low posterior-mean MSE while collapsing covariance."
    )
    lines.append("")
    for ci, cond in enumerate(conditions):
        d, _k, ratio, _m, noise_std, eta = cond
        lines.append(f"## d={int(d)}, ratio={ratio:g}, noise={noise_std:g}, eta={eta:g}")
        lines.append("| method | sample mean MSE | mean posterior MSE | cov error | weight KL | corr rel | corr cos |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        order = np.argsort(aggregate[ci, :, pm])
        for mi in order:
            vals = aggregate[ci, mi]
            lines.append(
                f"| `{METHOD_NAMES[mi]}` | {vals[sample_mean]:.4e} | {vals[pm]:.4e} | "
                f"{vals[cov]:.4e} | {vals[kl]:.4e} | {vals[rel]:.4e} | {vals[cos]:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    out: str | Path,
    seed: int,
    prior_name: str,
    d_values: list[int],
    measurement_ratios: list[float],
    noise_stds: list[float],
    eta_values: list[float],
    intrinsic_k: int,
    components: int,
    n_trials: int,
    n_steps: int,
    grid_size: int,
    sigma_min: float,
    sigma_max: float,
    sigma0: float,
    schedule_sigma_min: float,
    A_type: str,
    split_name: str,
    h: float,
    beta: float,
    raw_hqs_tau: float,
    pdhg_gamma: float,
    hqs_tau: float,
    device_name: str,
    dtype_name: str,
) -> dict:
    device = make_device(device_name)
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    out = Path(out)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    conditions = _condition_table(d_values, measurement_ratios, noise_stds, eta_values, intrinsic_k)
    sigma_grid = torch.tensor(np.geomspace(sigma_min, sigma_max, grid_size), device=device, dtype=dtype)
    schedule = _log_schedule(sigma0, schedule_sigma_min, n_steps, device, dtype)
    final_metrics = np.full(
        (conditions.shape[0], n_trials, len(METHOD_NAMES), len(FINAL_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    histories = np.full(
        (conditions.shape[0], n_trials, len(METHOD_NAMES), n_steps + 1, len(HISTORY_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    aggregate_metrics = np.full((conditions.shape[0], len(METHOD_NAMES), len(AGGREGATE_NAMES)), np.nan, dtype=np.float32)
    decomp = np.full((conditions.shape[0],), np.nan, dtype=np.float32)
    condition_times = np.full((conditions.shape[0],), np.nan, dtype=np.float32)

    start_all = time.perf_counter()
    with torch.no_grad():
        for ci, cond in enumerate(conditions):
            d, _k, ratio, m, noise_std, eta = cond
            d = int(d)
            m = int(m)
            noise_std = float(noise_std)
            eta = float(eta)
            start = time.perf_counter()
            prior = _make_prior(prior_name, d, components, sigma_grid, device, dtype, seed + 1000 * ci)
            gen = make_generator(device, seed + 100000 * ci)
            A = make_operator(d, m, A_type, gen, device, dtype)
            _x_true, y_obs = make_measurement(prior, A, noise_std, gen)
            posterior_prior, posterior = exact_linear_posterior_gmm(prior, A, y_obs, noise_std)
            y0 = prior.mean[None, :] + float(sigma0) * torch.randn(
                (n_trials, d), device=device, dtype=dtype, generator=make_generator(device, seed + 200000 * ci)
            )
            final_samples = np.full((n_trials, len(METHOD_NAMES), d), np.nan, dtype=np.float32)
            print(
                f"condition {ci + 1}/{conditions.shape[0]}: d={d}, ratio={ratio:g}, noise={noise_std:g}, "
                f"eta={eta:g}, trials={n_trials}, steps={n_steps}, split={split_name}, device={device}",
                flush=True,
            )
            for mi, method in enumerate(METHOD_NAMES):
                final_x, hist = _run_method(
                    method,
                    prior,
                    posterior_prior,
                    posterior,
                    y_obs,
                    A,
                    noise_std,
                    y0,
                    schedule,
                    split_name,
                    eta,
                    h,
                    beta,
                    raw_hqs_tau,
                    pdhg_gamma,
                    hqs_tau,
                    make_generator(device, seed + 300000 * ci + 101 * mi),
                )
                final_samples[:, mi] = final_x.detach().cpu().numpy().astype(np.float32)
                histories[ci, :, mi] = hist
                final_metrics[ci, :, mi] = _final_from_history(final_x, hist, posterior, y_obs, A)
                sync(device)
            po = METHOD_NAMES.index("posterior_oracle")
            exact = METHOD_NAMES.index("prior_exact_cstar")
            decomp[ci] = np.mean((final_samples[:, po] - final_samples[:, exact]) ** 2)
            aggregate_metrics[ci] = _aggregate(final_samples, final_metrics[ci], histories[ci], posterior, posterior_prior)
            condition_times[ci] = time.perf_counter() - start
            elapsed = time.perf_counter() - start_all
            avg = elapsed / (ci + 1)
            remaining = avg * (conditions.shape[0] - ci - 1)
            print(
                f"  condition_time={condition_times[ci]:.1f}s; eta_remaining={remaining / 60.0:.1f}m",
                flush=True,
            )
            np.savez_compressed(
                data_dir / "closed_loop_cuda_partial.npz",
                final_metrics=final_metrics,
                histories=histories,
                aggregate_metrics=aggregate_metrics,
                posterior_oracle_exact_cstar_final_mse=decomp,
                conditions=conditions,
                condition_names=np.asarray(CONDITION_NAMES),
                method_names=np.asarray(METHOD_NAMES),
                final_names=np.asarray(FINAL_NAMES),
                history_names=np.asarray(HISTORY_NAMES),
                aggregate_names=np.asarray(AGGREGATE_NAMES),
                schedule=schedule.detach().cpu().numpy(),
                split_name=np.asarray(split_name),
                device=np.asarray(str(device)),
                dtype=np.asarray(dtype_name),
            )

    payload = {
        "final_metrics": final_metrics,
        "histories": histories,
        "aggregate_metrics": aggregate_metrics,
        "posterior_oracle_exact_cstar_final_mse": decomp,
        "conditions": conditions,
        "condition_names": np.asarray(CONDITION_NAMES),
        "method_names": np.asarray(METHOD_NAMES),
        "final_names": np.asarray(FINAL_NAMES),
        "history_names": np.asarray(HISTORY_NAMES),
        "aggregate_names": np.asarray(AGGREGATE_NAMES),
        "schedule": schedule.detach().cpu().numpy(),
        "condition_times": condition_times,
        "split_name": np.asarray(split_name),
        "device": np.asarray(str(device)),
        "dtype": np.asarray(dtype_name),
    }
    np.savez_compressed(data_dir / "closed_loop_cuda.npz", **payload)
    config = {
        "seed": seed,
        "prior": prior_name,
        "d_values": d_values,
        "measurement_ratios": measurement_ratios,
        "noise_stds": noise_stds,
        "eta_values": eta_values,
        "intrinsic_k": intrinsic_k,
        "components": components,
        "n_trials": n_trials,
        "n_steps": n_steps,
        "grid_size": grid_size,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "sigma0": sigma0,
        "schedule_sigma_min": schedule_sigma_min,
        "A_type": A_type,
        "split_name": split_name,
        "h": h,
        "beta": beta,
        "raw_hqs_tau": raw_hqs_tau,
        "pdhg_gamma": pdhg_gamma,
        "hqs_tau": hqs_tau,
        "device": str(device),
        "dtype": dtype_name,
    }
    (data_dir / "closed_loop_cuda_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _write_report(data_dir / "closed_loop_cuda_report.md", payload)
    print(f"saved CUDA closed-loop hierarchy to {data_dir / 'closed_loop_cuda.npz'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="posterior_bddm_oracle/results_cuda_closed_loop")
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--prior", choices=["ellipse", "full"], default="ellipse")
    parser.add_argument("--d-values", default="50,100,500")
    parser.add_argument("--measurement-ratios", default="0.5")
    parser.add_argument("--noise-stds", default="0.08")
    parser.add_argument("--eta-values", default="0.03")
    parser.add_argument("--intrinsic-k", type=int, default=2)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--n-trials", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--sigma0", type=float, default=1.2)
    parser.add_argument("--schedule-sigma-min", type=float, default=0.02)
    parser.add_argument("--A-type", choices=["random", "mask"], default="random")
    parser.add_argument("--split", choices=["gradient", "pdhg", "hqs"], default="gradient")
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--raw-hqs-tau", type=float, default=1e-2)
    parser.add_argument("--pdhg-gamma", type=float, default=100.0)
    parser.add_argument("--hqs-tau", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    if args.quick:
        args.d_values = "10"
        args.eta_values = "0.03"
        args.n_trials = min(args.n_trials, 4)
        args.n_steps = min(args.n_steps, 8)
        args.components = min(args.components, 8)
        args.grid_size = min(args.grid_size, 25)

    run_experiment(
        out=args.out,
        seed=args.seed,
        prior_name=args.prior,
        d_values=parse_int_list(args.d_values),
        measurement_ratios=parse_float_list(args.measurement_ratios),
        noise_stds=parse_float_list(args.noise_stds),
        eta_values=parse_float_list(args.eta_values),
        intrinsic_k=args.intrinsic_k,
        components=args.components,
        n_trials=args.n_trials,
        n_steps=args.n_steps,
        grid_size=args.grid_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma0=args.sigma0,
        schedule_sigma_min=args.schedule_sigma_min,
        A_type=args.A_type,
        split_name=args.split,
        h=args.h,
        beta=args.beta,
        raw_hqs_tau=args.raw_hqs_tau,
        pdhg_gamma=args.pdhg_gamma,
        hqs_tau=args.hqs_tau,
        device_name=args.device,
        dtype_name=args.dtype,
    )


if __name__ == "__main__":
    main()

"""Calibrate the CUDA posterior oracle before comparing split methods."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .experiments_posterior_closed_loop_torch import (
    HISTORY_NAMES,
    _component_kl,
    _covariance_error,
    _denoise_at_sigmas,
    _log_schedule,
    _make_prior,
    _measurement_mse,
    _run_method,
)
from .torch_oracle_tools import (
    exact_linear_posterior_gmm,
    make_device,
    make_generator,
    make_measurement,
    make_operator,
    parse_float_list,
    parse_int_list,
    sync,
)


METHODS = ["posterior_oracle", "prior_exact_cstar"]
METRIC_NAMES = [
    "sample_mean_mse",
    "mean_posterior_mean_mse",
    "posterior_covariance_error",
    "component_weight_kl",
    "mean_measurement_mse",
    "prior_boundary_hit_rate",
    "posterior_boundary_hit_rate",
]
CONDITION_NAMES = ["d", "measurement_ratio", "m", "noise_std", "h", "beta", "n_steps"]


def _conditions(
    d_values: list[int],
    measurement_ratios: list[float],
    noise_stds: list[float],
    h_values: list[float],
    beta_values: list[float],
    n_steps_values: list[int],
) -> np.ndarray:
    rows = []
    for d in d_values:
        for ratio in measurement_ratios:
            m = max(1, int(round(float(ratio) * d)))
            for noise_std in noise_stds:
                for h in h_values:
                    for beta in beta_values:
                        for n_steps in n_steps_values:
                            rows.append((d, ratio, m, noise_std, h, beta, n_steps))
    return np.asarray(rows, dtype=float)


def _aggregate(
    samples: np.ndarray,
    histories: np.ndarray,
    posterior: dict[str, torch.Tensor],
    posterior_prior,
    y_obs: torch.Tensor,
    A: torch.Tensor,
) -> np.ndarray:
    out = np.full((len(METHODS), len(METRIC_NAMES)), np.nan, dtype=np.float32)
    posterior_mean = posterior["mean"].detach().cpu().numpy().astype(np.float32)
    posterior_cov = posterior["covariance"].detach().cpu().numpy().astype(np.float32)
    target_weights = posterior["component_weights"].detach().cpu().numpy().astype(np.float64)

    for mi in range(len(METHODS)):
        method_samples = samples[:, mi]
        sample_t = torch.as_tensor(method_samples, device=posterior_prior.device, dtype=posterior_prior.dtype)
        resp = posterior_prior.posterior_component_weights(
            sample_t,
            torch.as_tensor(1e-6, device=posterior_prior.device, dtype=posterior_prior.dtype),
        )
        omega_hat = torch.mean(resp, dim=0).detach().cpu().numpy().astype(np.float64)
        residual = sample_t @ A.T - y_obs[None, :]
        out[mi, METRIC_NAMES.index("sample_mean_mse")] = np.mean(
            (np.mean(method_samples, axis=0) - posterior_mean) ** 2
        )
        out[mi, METRIC_NAMES.index("mean_posterior_mean_mse")] = np.mean(
            np.mean((method_samples - posterior_mean[None, :]) ** 2, axis=1)
        )
        out[mi, METRIC_NAMES.index("posterior_covariance_error")] = _covariance_error(
            method_samples,
            posterior_cov,
        )
        out[mi, METRIC_NAMES.index("component_weight_kl")] = _component_kl(omega_hat, target_weights)
        out[mi, METRIC_NAMES.index("mean_measurement_mse")] = torch.mean(
            torch.mean(residual.square(), dim=1)
        ).item()
        out[mi, METRIC_NAMES.index("prior_boundary_hit_rate")] = np.nanmean(
            histories[:, mi, :, HISTORY_NAMES.index("prior_boundary_hit")]
        )
        out[mi, METRIC_NAMES.index("posterior_boundary_hit_rate")] = np.nanmean(
            histories[:, mi, :, HISTORY_NAMES.index("posterior_boundary_hit")]
        )
    return out


def _write_report(path: Path, payload: dict) -> None:
    metrics = payload["metrics"]
    conditions = payload["conditions"]
    decomp = payload["decomposition_mse"]
    cov_idx = METRIC_NAMES.index("posterior_covariance_error")
    pm_idx = METRIC_NAMES.index("mean_posterior_mean_mse")
    kl_idx = METRIC_NAMES.index("component_weight_kl")

    lines = ["# CUDA Posterior Oracle Calibration", ""]
    lines.append(f"Device: `{payload['device']}`")
    lines.append("")
    lines.append("## Best Oracle Settings By Covariance Error")
    oracle_idx = METHODS.index("posterior_oracle")
    order = np.argsort(metrics[:, oracle_idx, cov_idx])
    lines.append("| rank | d | h | beta | n_steps | cov error | posterior MSE | weight KL | decomposition MSE |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rank, ci in enumerate(order[: min(20, order.size)], start=1):
        d, _ratio, _m, _noise, h, beta, n_steps = conditions[ci]
        vals = metrics[ci, oracle_idx]
        lines.append(
            f"| {rank} | {int(d)} | {h:g} | {beta:g} | {int(n_steps)} | "
            f"{vals[cov_idx]:.4e} | {vals[pm_idx]:.4e} | {vals[kl_idx]:.4e} | {decomp[ci]:.4e} |"
        )
    lines.append("")

    lines.append("## Full Table")
    lines.append("| d | ratio | noise | h | beta | n_steps | method | cov error | posterior MSE | weight KL |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    for ci, cond in enumerate(conditions):
        d, ratio, _m, noise_std, h, beta, n_steps = cond
        for mi, method in enumerate(METHODS):
            vals = metrics[ci, mi]
            lines.append(
                f"| {int(d)} | {ratio:g} | {noise_std:g} | {h:g} | {beta:g} | {int(n_steps)} | "
                f"`{method}` | {vals[cov_idx]:.4e} | {vals[pm_idx]:.4e} | {vals[kl_idx]:.4e} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    out: str | Path,
    seed: int,
    prior_name: str,
    d_values: list[int],
    measurement_ratios: list[float],
    noise_stds: list[float],
    h_values: list[float],
    beta_values: list[float],
    n_steps_values: list[int],
    components: int,
    n_trials: int,
    grid_size: int,
    sigma_min: float,
    sigma_max: float,
    sigma0: float,
    schedule_sigma_min: float,
    A_type: str,
    device_name: str,
    dtype_name: str,
) -> dict:
    device = make_device(device_name)
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    out = Path(out)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    conds = _conditions(d_values, measurement_ratios, noise_stds, h_values, beta_values, n_steps_values)
    sigma_grid = torch.tensor(np.geomspace(sigma_min, sigma_max, grid_size), device=device, dtype=dtype)
    metrics = np.full((conds.shape[0], len(METHODS), len(METRIC_NAMES)), np.nan, dtype=np.float32)
    decomp = np.full((conds.shape[0],), np.nan, dtype=np.float32)
    condition_times = np.full((conds.shape[0],), np.nan, dtype=np.float32)

    start_all = time.perf_counter()
    with torch.no_grad():
        for ci, cond in enumerate(conds):
            d, ratio, m, noise_std, h, beta, n_steps = cond
            d = int(d)
            m = int(m)
            noise_std = float(noise_std)
            h = float(h)
            beta = float(beta)
            n_steps = int(n_steps)
            start = time.perf_counter()

            prior = _make_prior(prior_name, d, components, sigma_grid, device, dtype, seed + 1000 * ci + d)
            gen = make_generator(device, seed + 100000 * ci)
            A = make_operator(d, m, A_type, gen, device, dtype)
            _x_true, y_obs = make_measurement(prior, A, noise_std, gen)
            posterior_prior, posterior = exact_linear_posterior_gmm(prior, A, y_obs, noise_std)
            init_gen = make_generator(device, seed + 200000 * ci)
            x0, _ids = posterior_prior.sample(n_trials, init_gen)
            y0 = x0 + float(sigma0) * torch.randn((n_trials, d), device=device, dtype=dtype, generator=init_gen)
            schedule = _log_schedule(sigma0, schedule_sigma_min, n_steps, device, dtype)

            final_samples = np.full((n_trials, len(METHODS), d), np.nan, dtype=np.float32)
            max_steps = n_steps + 1
            histories = np.full((n_trials, len(METHODS), max_steps, len(HISTORY_NAMES)), np.nan, dtype=np.float32)

            print(
                f"condition {ci + 1}/{conds.shape[0]}: d={d}, h={h:g}, beta={beta:g}, "
                f"steps={n_steps}, trials={n_trials}, device={device}",
                flush=True,
            )
            for mi, method in enumerate(METHODS):
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
                    "gradient",
                    0.0,
                    h,
                    beta,
                    1e-2,
                    100.0,
                    1e-2,
                    make_generator(device, seed + 300000 * ci),
                )
                final_samples[:, mi] = final_x.detach().cpu().numpy().astype(np.float32)
                histories[:, mi] = hist
                sync(device)

            metrics[ci] = _aggregate(final_samples, histories, posterior, posterior_prior, y_obs, A)
            decomp[ci] = np.mean(
                (final_samples[:, METHODS.index("posterior_oracle")] - final_samples[:, METHODS.index("prior_exact_cstar")])
                ** 2
            )
            condition_times[ci] = time.perf_counter() - start
            elapsed = time.perf_counter() - start_all
            remaining = elapsed / (ci + 1) * (conds.shape[0] - ci - 1)
            print(
                f"  condition_time={condition_times[ci]:.1f}s; eta_remaining={remaining / 60.0:.1f}m",
                flush=True,
            )
            np.savez_compressed(
                data_dir / "posterior_oracle_calibration_partial.npz",
                metrics=metrics,
                metric_names=np.asarray(METRIC_NAMES),
                conditions=conds,
                condition_names=np.asarray(CONDITION_NAMES),
                method_names=np.asarray(METHODS),
                decomposition_mse=decomp,
                condition_times=condition_times,
                device=np.asarray(str(device)),
                dtype=np.asarray(dtype_name),
            )

    payload = {
        "metrics": metrics,
        "metric_names": np.asarray(METRIC_NAMES),
        "conditions": conds,
        "condition_names": np.asarray(CONDITION_NAMES),
        "method_names": np.asarray(METHODS),
        "decomposition_mse": decomp,
        "condition_times": condition_times,
        "device": np.asarray(str(device)),
        "dtype": np.asarray(dtype_name),
    }
    np.savez_compressed(data_dir / "posterior_oracle_calibration.npz", **payload)
    config = {
        "seed": seed,
        "prior": prior_name,
        "d_values": d_values,
        "measurement_ratios": measurement_ratios,
        "noise_stds": noise_stds,
        "h_values": h_values,
        "beta_values": beta_values,
        "n_steps_values": n_steps_values,
        "components": components,
        "n_trials": n_trials,
        "grid_size": grid_size,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "sigma0": sigma0,
        "schedule_sigma_min": schedule_sigma_min,
        "A_type": A_type,
        "device": str(device),
        "dtype": dtype_name,
    }
    (data_dir / "posterior_oracle_calibration_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _write_report(data_dir / "posterior_oracle_calibration_report.md", payload)
    print(f"saved posterior oracle calibration to {data_dir / 'posterior_oracle_calibration.npz'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="posterior_bddm_oracle/results_cuda_oracle_calibration")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--prior", choices=["ellipse", "full"], default="ellipse")
    parser.add_argument("--d-values", default="100,500")
    parser.add_argument("--measurement-ratios", default="0.5")
    parser.add_argument("--noise-stds", default="0.08")
    parser.add_argument("--h-values", default="0.01,0.02,0.05")
    parser.add_argument("--beta-values", default="0,0.01,0.03,0.05")
    parser.add_argument("--n-steps-values", default="40,80")
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--n-trials", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--sigma0", type=float, default=1.2)
    parser.add_argument("--schedule-sigma-min", type=float, default=0.02)
    parser.add_argument("--A-type", choices=["random", "mask"], default="random")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    if args.quick:
        args.d_values = "10"
        args.h_values = "0.02,0.05"
        args.beta_values = "0,0.03"
        args.n_steps_values = "8"
        args.components = min(args.components, 8)
        args.n_trials = min(args.n_trials, 8)
        args.grid_size = min(args.grid_size, 25)

    run_experiment(
        out=args.out,
        seed=args.seed,
        prior_name=args.prior,
        d_values=parse_int_list(args.d_values),
        measurement_ratios=parse_float_list(args.measurement_ratios),
        noise_stds=parse_float_list(args.noise_stds),
        h_values=parse_float_list(args.h_values),
        beta_values=parse_float_list(args.beta_values),
        n_steps_values=parse_int_list(args.n_steps_values),
        components=args.components,
        n_trials=args.n_trials,
        grid_size=args.grid_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma0=args.sigma0,
        schedule_sigma_min=args.schedule_sigma_min,
        A_type=args.A_type,
        device_name=args.device,
        dtype_name=args.dtype,
    )


if __name__ == "__main__":
    main()


"""Lightweight closed-loop benchmark with checkpoints and timing."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from .metrics import measurement_mse, negative_log_joint, posterior_mean_mse
from .plotting import ensure_dir
from .priors import GaussianMixturePrior, make_ellipse_gmm, make_full_gaussian_control, make_subspace_gmm


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


def _make_prior(name: str, d: int, rng: np.random.Generator, sigma_grid: np.ndarray, components: int) -> GaussianMixturePrior:
    if name == "ellipse":
        return make_ellipse_gmm(d, n_components=components, rng=rng, sigma_grid=sigma_grid)
    if name == "subspace":
        return make_subspace_gmm(d, n_components=min(8, components), rng=rng, sigma_grid=sigma_grid)
    if name == "full":
        return make_full_gaussian_control(d, rng=rng, sigma_grid=sigma_grid)
    raise ValueError(f"unknown prior: {name}")


def _run_method(
    method: str,
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
    if method == "scheduled_nonblind":
        return scheduled_nonblind_forced_diffusion(
            prior, A, y_obs, noise_std, schedule, eta=eta, h=h, beta=beta, rng=rng, y0=y0
        )
    if method == "blind_mle":
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
    if method == "oracle_scale":
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
    if method == "cov_filtered":
        return covariance_filtered_clean_update(
            prior, A, y_obs, noise_std, schedule, eta=eta, rng=rng, y0=y0, sigma_mode="schedule"
        )
    if method == "raw_pnp_tuned":
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
        )
    raise ValueError(f"unknown method: {method}")


def _history_array(
    result: dict,
    x_true: np.ndarray,
    posterior_mean: np.ndarray,
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    sigma_min: float,
    sigma_max: float,
    n_steps: int,
) -> np.ndarray:
    out = np.full((n_steps + 1, len(HISTORY_METRIC_NAMES)), np.nan, dtype=np.float32)
    x_hist = np.asarray(result["history"]["x"], dtype=float)
    length = min(n_steps + 1, x_hist.shape[0])
    for step in range(length):
        x = x_hist[step]
        out[step, 0] = np.mean((x - x_true) ** 2)
        out[step, 1] = measurement_mse(x, y_obs, A)
        out[step, 2] = posterior_mean_mse(x, posterior_mean)
        out[step, 3] = negative_log_joint(x, prior, A, y_obs, noise_std)
    for key in ["sigma_used", "sigma_hat", "grad_norm", "cov_grad_norm", "raw_normal_ratio", "cov_normal_ratio"]:
        vals = np.asarray(result["history"][key], dtype=float)
        idx = HISTORY_METRIC_NAMES.index(key)
        n = min(vals.size, n_steps + 1)
        out[:n, idx] = vals[:n]
    sigma_hat = out[:, HISTORY_METRIC_NAMES.index("sigma_hat")]
    out[:, HISTORY_METRIC_NAMES.index("sigma_min_hit")] = (sigma_hat == sigma_min).astype(np.float32)
    out[:, HISTORY_METRIC_NAMES.index("sigma_max_hit")] = (sigma_hat == sigma_max).astype(np.float32)
    return out


def _final_from_history(history: np.ndarray) -> np.ndarray:
    valid = np.where(np.isfinite(history[:, HISTORY_METRIC_NAMES.index("posterior_mean_mse")]))[0]
    step = int(valid[-1]) if valid.size else 0
    sigma_hat = history[:, HISTORY_METRIC_NAMES.index("sigma_hat")]
    finite_sigma = sigma_hat[np.isfinite(sigma_hat)]
    return np.asarray(
        [
            history[step, HISTORY_METRIC_NAMES.index("mse_true")],
            history[step, HISTORY_METRIC_NAMES.index("measurement_mse")],
            history[step, HISTORY_METRIC_NAMES.index("posterior_mean_mse")],
            history[step, HISTORY_METRIC_NAMES.index("neg_log_joint")],
            finite_sigma[-1] if finite_sigma.size else np.nan,
            np.nanmean(history[:, HISTORY_METRIC_NAMES.index("sigma_min_hit")]),
            np.nanmean(history[:, HISTORY_METRIC_NAMES.index("sigma_max_hit")]),
        ],
        dtype=np.float32,
    )


def _run_trial(payload: dict) -> dict:
    prior = payload["prior"]
    prior.precompute_sigma_grid()
    prior.precompute_sigmas(payload["schedule"])
    trial_seed = payload["trial_seed"]
    rng = np.random.default_rng(trial_seed)
    meas = make_measurement(prior, payload["A_type"], payload["m"], payload["noise_std"], rng)
    posterior = exact_linear_posterior(prior, meas["A"], meas["y_obs"], payload["noise_std"])
    y0 = initial_noisy_state(prior, payload["sigma0"], np.random.default_rng(trial_seed + 55))

    methods = payload["methods"]
    final_metrics = np.full((len(methods), len(FINAL_METRIC_NAMES)), np.nan, dtype=np.float32)
    diag_history = np.full((len(methods), payload["n_steps"] + 1, len(HISTORY_METRIC_NAMES)), np.nan, dtype=np.float32)
    method_times = np.full(len(methods), np.nan, dtype=np.float32)

    for mi, method in enumerate(methods):
        start = time.perf_counter()
        result = _run_method(
            method,
            prior,
            meas["A"],
            meas["y_obs"],
            payload["noise_std"],
            payload["schedule"],
            payload["eta"],
            payload["beta"],
            payload["h"],
            payload["sigma_min"],
            y0,
            np.random.default_rng(trial_seed + 1000 * mi + 17),
            payload["raw_pnp_step_cap"],
        )
        method_times[mi] = time.perf_counter() - start
        hist = _history_array(
            result,
            meas["x_true"],
            posterior["mean"],
            prior,
            meas["A"],
            meas["y_obs"],
            payload["noise_std"],
            payload["sigma_grid"][0],
            payload["sigma_grid"][-1],
            payload["n_steps"],
        )
        final_metrics[mi] = _final_from_history(hist)
        if payload["store_diagnostic"]:
            diag_history[mi] = hist

    return {
        "trial_index": payload["trial_index"],
        "final_metrics": final_metrics,
        "diag_history": diag_history if payload["store_diagnostic"] else None,
        "method_times": method_times,
    }


def _condition_table(d_values: list[int], ratios: list[float], etas: list[float], betas: list[float]) -> np.ndarray:
    rows = []
    for d in d_values:
        for ratio in ratios:
            for eta in etas:
                for beta in betas:
                    rows.append((d, ratio, eta, beta))
    return np.asarray(rows, dtype=float)


def _save_checkpoint(path: Path, payload: dict) -> None:
    np.savez_compressed(path, **payload)


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _format_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    return f"{seconds / 60.0:.1f}m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="toy_blind_splitting/results/closed_loop_light")
    parser.add_argument("--prior", choices=["ellipse", "subspace", "full"], default="ellipse")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--d-values", default="12")
    parser.add_argument("--measurement-ratios", default="0.5")
    parser.add_argument("--eta-values", default="1e-3")
    parser.add_argument("--beta-values", default="0.0")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--methods", default="scheduled_nonblind,blind_mle,oracle_scale,cov_filtered,raw_pnp_tuned")
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
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--diagnostic-trials", type=int, default=1)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    data_dir = ensure_dir(out_dir / "data")
    ckpt_path = data_dir / f"closed_loop_light_{args.prior}_partial.npz"
    final_path = data_dir / f"closed_loop_light_{args.prior}.npz"

    d_values = _parse_int_list(args.d_values)
    ratios = _parse_float_list(args.measurement_ratios)
    etas = _parse_float_list(args.eta_values)
    betas = _parse_float_list(args.beta_values)
    methods = _parse_str_list(args.methods)
    conditions = _condition_table(d_values, ratios, etas, betas)
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)

    shape = (conditions.shape[0], args.n_trials, len(methods))
    checkpoint = _load_checkpoint(ckpt_path) if args.resume else None
    if checkpoint is None:
        final_metrics = np.full(shape + (len(FINAL_METRIC_NAMES),), np.nan, dtype=np.float32)
        trajectory_times = np.full(shape, np.nan, dtype=np.float32)
        method_times = np.full((conditions.shape[0], len(methods)), np.nan, dtype=np.float32)
        condition_times = np.full(conditions.shape[0], np.nan, dtype=np.float32)
        diagnostic_histories = np.full(
            (
                conditions.shape[0],
                min(args.diagnostic_trials, args.n_trials),
                len(methods),
                args.n_steps + 1,
                len(HISTORY_METRIC_NAMES),
            ),
            np.nan,
            dtype=np.float32,
        )
        completed = np.zeros(conditions.shape[0], dtype=bool)
    else:
        final_metrics = checkpoint["final_metrics"]
        trajectory_times = checkpoint["trajectory_times"]
        method_times = checkpoint["method_times"]
        condition_times = checkpoint["condition_times"]
        diagnostic_histories = checkpoint["diagnostic_histories"]
        completed = checkpoint["completed"].astype(bool)

    sweep_start = time.perf_counter()
    total_trajectories = conditions.shape[0] * args.n_trials * len(methods)
    completed_trajectories = int(np.sum(np.isfinite(trajectory_times)))

    for ci, (d_float, ratio, eta, beta) in enumerate(conditions):
        if completed[ci]:
            continue
        condition_start = time.perf_counter()
        d = int(d_float)
        m = max(1, int(round(ratio * d)))
        prior_rng = np.random.default_rng(args.seed + 1000 * ci + d)
        prior = _make_prior(args.prior, d, prior_rng, sigma_grid, args.curve_components)
        schedule = log_schedule(args.sigma0, args.schedule_sigma_min, args.n_steps)
        prior.precompute_sigma_grid()
        prior.precompute_sigmas(schedule)

        print(
            f"starting condition {ci + 1}/{conditions.shape[0]}: "
            f"d={d}, ratio={ratio:g}, eta={eta:g}, beta={beta:g}, "
            f"trials={args.n_trials}, methods={len(methods)}",
            flush=True,
        )

        trial_payloads = []
        for trial in range(args.n_trials):
            trial_payloads.append(
                {
                    "trial_index": trial,
                    "trial_seed": args.seed + 100000 * ci + trial,
                    "prior": prior,
                    "A_type": args.A_type,
                    "m": m,
                    "noise_std": args.noise_std,
                    "sigma0": args.sigma0,
                    "schedule": schedule,
                    "sigma_grid": sigma_grid,
                    "eta": float(eta),
                    "beta": float(beta),
                    "h": args.h,
                    "sigma_min": args.sigma_min,
                    "raw_pnp_step_cap": args.raw_pnp_step_cap,
                    "methods": methods,
                    "n_steps": args.n_steps,
                    "store_diagnostic": trial < diagnostic_histories.shape[1],
                }
            )

        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                futures = [pool.submit(_run_trial, payload) for payload in trial_payloads]
                for future in as_completed(futures):
                    result = future.result()
                    trial = int(result["trial_index"])
                    final_metrics[ci, trial] = result["final_metrics"]
                    trajectory_times[ci, trial] = result["method_times"]
                    if result["diag_history"] is not None and trial < diagnostic_histories.shape[1]:
                        diagnostic_histories[ci, trial] = result["diag_history"]
        else:
            for payload in trial_payloads:
                result = _run_trial(payload)
                trial = int(result["trial_index"])
                final_metrics[ci, trial] = result["final_metrics"]
                trajectory_times[ci, trial] = result["method_times"]
                if result["diag_history"] is not None and trial < diagnostic_histories.shape[1]:
                    diagnostic_histories[ci, trial] = result["diag_history"]

        condition_times[ci] = time.perf_counter() - condition_start
        method_times[ci] = np.nanmean(trajectory_times[ci], axis=0)
        completed[ci] = True
        completed_trajectories = int(np.sum(np.isfinite(trajectory_times)))
        elapsed = time.perf_counter() - sweep_start
        seconds_per_trajectory = elapsed / max(completed_trajectories, 1)
        remaining = total_trajectories - completed_trajectories
        eta_remaining = remaining * seconds_per_trajectory

        print(
            f"completed condition {ci + 1}/{conditions.shape[0]}; "
            f"condition_time={_format_seconds(condition_times[ci])}; "
            f"avg={seconds_per_trajectory:.3f}s/trajectory; "
            f"elapsed={_format_seconds(elapsed)}; "
            f"eta_remaining={_format_seconds(eta_remaining)}",
            flush=True,
        )
        for mi, method in enumerate(methods):
            print(f"  method_time {method}: {method_times[ci, mi]:.3f}s/trajectory", flush=True)

        _save_checkpoint(
            ckpt_path,
            {
                "final_metrics": final_metrics,
                "trajectory_times": trajectory_times,
                "method_times": method_times,
                "condition_times": condition_times,
                "diagnostic_histories": diagnostic_histories,
                "completed": completed,
                "conditions": conditions,
                "method_names": np.asarray(methods),
                "final_metric_names": np.asarray(FINAL_METRIC_NAMES),
                "history_metric_names": np.asarray(HISTORY_METRIC_NAMES),
                "d_values": np.asarray(d_values),
                "measurement_ratios": np.asarray(ratios),
                "eta_values": np.asarray(etas),
                "beta_values": np.asarray(betas),
                "n_steps": np.asarray(args.n_steps),
                "n_trials": np.asarray(args.n_trials),
            },
        )

    _save_checkpoint(
        final_path,
        {
            "final_metrics": final_metrics,
            "trajectory_times": trajectory_times,
            "method_times": method_times,
            "condition_times": condition_times,
            "diagnostic_histories": diagnostic_histories,
            "completed": completed,
            "conditions": conditions,
            "method_names": np.asarray(methods),
            "final_metric_names": np.asarray(FINAL_METRIC_NAMES),
            "history_metric_names": np.asarray(HISTORY_METRIC_NAMES),
            "d_values": np.asarray(d_values),
            "measurement_ratios": np.asarray(ratios),
            "eta_values": np.asarray(etas),
            "beta_values": np.asarray(betas),
            "n_steps": np.asarray(args.n_steps),
            "n_trials": np.asarray(args.n_trials),
        },
    )

    finite_times = trajectory_times[np.isfinite(trajectory_times)]
    cpu_seconds_per_trajectory = float(np.mean(finite_times)) if finite_times.size else float("nan")
    completed_conditions = np.isfinite(condition_times)
    completed_trajectories = int(np.sum(np.isfinite(trajectory_times)))
    wall_seconds_per_trajectory = (
        float(np.nansum(condition_times[completed_conditions]) / completed_trajectories)
        if completed_trajectories
        else float("nan")
    )
    full_conditions = 3 * 3 * 6 * 3
    full_trials = 100
    projected = wall_seconds_per_trajectory * full_conditions * full_trials * len(methods)
    print(f"saved final results to {final_path}", flush=True)
    print(f"benchmark wall seconds per trajectory: {wall_seconds_per_trajectory:.3f}", flush=True)
    print(f"benchmark per-process method seconds per trajectory: {cpu_seconds_per_trajectory:.3f}", flush=True)
    print(
        f"projected runtime for 3 d x 3 ratios x 6 etas x 3 betas x 100 trials x {len(methods)} methods: "
        f"{_format_seconds(projected)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

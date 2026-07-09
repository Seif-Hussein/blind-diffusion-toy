"""Reduced finite tilted-posterior/Kalman assimilation experiment."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .algorithms import (
    blind_forced_diffusion,
    covariance_filtered_clean_update,
    finite_kalman_assimilation,
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
    "median_tr_C",
    "median_tr_C_plus",
    "median_grad_norm",
    "median_cov_grad_norm",
    "median_kalman_update_norm",
    "median_assimilation_cond",
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
    "tr_C",
    "tr_C_plus",
    "kalman_update_norm",
    "assimilation_cond",
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
    components: int,
) -> GaussianMixturePrior:
    if name == "ellipse":
        return make_ellipse_gmm(d, n_components=components, rng=rng, sigma_grid=sigma_grid)
    if name == "subspace":
        return make_subspace_gmm(d, n_components=min(8, components), rng=rng, sigma_grid=sigma_grid)
    if name == "full":
        return make_full_gaussian_control(d, rng=rng, sigma_grid=sigma_grid)
    raise ValueError(f"unknown prior: {name}")


def _finite_method_name(alpha: float, lift: str) -> str:
    return f"finite_alpha_{alpha:g}_{lift}"


def _method_names(alphas: list[float], lifts: list[str]) -> list[str]:
    names = ["scheduled_nonblind", "blind_mle", "oracle_scale", "cov_grad", "raw_pnp_tuned"]
    names.extend(_finite_method_name(alpha, lift) for alpha in alphas for lift in lifts)
    return names


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
    if method == "cov_grad":
        return covariance_filtered_clean_update(
            prior,
            A,
            y_obs,
            noise_std,
            schedule,
            eta=eta,
            rng=rng,
            y0=y0,
            sigma_mode="schedule",
            log_full_covariance=True,
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
        ) | {"name": "raw_pnp_tuned"}

    if method.startswith("finite_alpha_"):
        rest = method.removeprefix("finite_alpha_")
        alpha_text, lift = rest.split("_", 1)
        return finite_kalman_assimilation(
            prior,
            A,
            y_obs,
            noise_std,
            schedule,
            alpha=float(alpha_text),
            h=h,
            beta=beta,
            rng=rng,
            y0=y0,
            mode="mle",
            lift=lift,
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

    source_to_idx = {
        "sigma_used": HISTORY_METRIC_NAMES.index("sigma_used"),
        "sigma_hat": HISTORY_METRIC_NAMES.index("sigma_hat"),
        "grad_norm": HISTORY_METRIC_NAMES.index("grad_norm"),
        "cov_grad_norm": HISTORY_METRIC_NAMES.index("cov_grad_norm"),
        "tr_C": HISTORY_METRIC_NAMES.index("tr_C"),
        "tr_C_plus": HISTORY_METRIC_NAMES.index("tr_C_plus"),
        "kalman_update_norm": HISTORY_METRIC_NAMES.index("kalman_update_norm"),
        "assimilation_cond": HISTORY_METRIC_NAMES.index("assimilation_cond"),
    }
    for key, idx in source_to_idx.items():
        vals = np.asarray(result["history"].get(key, []), dtype=float)
        n = min(vals.size, n_steps + 1)
        out[:n, idx] = vals[:n]

    sigma_hat = out[:, HISTORY_METRIC_NAMES.index("sigma_hat")]
    out[:, HISTORY_METRIC_NAMES.index("sigma_min_hit")] = (sigma_hat == sigma_min).astype(np.float32)
    out[:, HISTORY_METRIC_NAMES.index("sigma_max_hit")] = (sigma_hat == sigma_max).astype(np.float32)
    return out


def _nanmedian(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


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
            _nanmedian(history[:, HISTORY_METRIC_NAMES.index("sigma_min_hit")]),
            _nanmedian(history[:, HISTORY_METRIC_NAMES.index("sigma_max_hit")]),
            _nanmedian(history[:, HISTORY_METRIC_NAMES.index("tr_C")]),
            _nanmedian(history[:, HISTORY_METRIC_NAMES.index("tr_C_plus")]),
            _nanmedian(history[:, HISTORY_METRIC_NAMES.index("grad_norm")]),
            _nanmedian(history[:, HISTORY_METRIC_NAMES.index("cov_grad_norm")]),
            _nanmedian(history[:, HISTORY_METRIC_NAMES.index("kalman_update_norm")]),
            _nanmedian(history[:, HISTORY_METRIC_NAMES.index("assimilation_cond")]),
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
    histories = np.full((len(methods), payload["n_steps"] + 1, len(HISTORY_METRIC_NAMES)), np.nan, dtype=np.float32)
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
        histories[mi] = _history_array(
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
        final_metrics[mi] = _final_from_history(histories[mi])

    return {
        "trial_index": payload["trial_index"],
        "final_metrics": final_metrics,
        "histories": histories,
        "method_times": method_times,
    }


def _condition_table(d_values: list[int], etas: list[float], betas: list[float]) -> np.ndarray:
    rows = []
    for d in d_values:
        for eta in etas:
            for beta in betas:
                rows.append((d, eta, beta))
    return np.asarray(rows, dtype=float)


def _save_npz(path: Path, **payload) -> None:
    np.savez_compressed(path, **payload)


def _load_npz(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _format_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    return f"{seconds / 60.0:.1f}m"


def _write_report(
    path: Path,
    final_metrics: np.ndarray,
    conditions: np.ndarray,
    methods: list[str],
) -> None:
    pm = FINAL_METRIC_NAMES.index("posterior_mean_mse")
    meas = FINAL_METRIC_NAMES.index("measurement_mse")
    nlj = FINAL_METRIC_NAMES.index("neg_log_joint")

    lines = ["# Finite Assimilation Reduced Report", ""]
    lines.append("## Aggregate Median Posterior-Mean MSE")
    aggregate = np.nanmedian(final_metrics[..., pm], axis=(0, 1))
    for mi in np.argsort(aggregate):
        lines.append(f"- `{methods[mi]}`: `{aggregate[mi]:.4e}`")
    lines.append("")

    old_idx = methods.index("cov_grad")
    finite_indices = [i for i, m in enumerate(methods) if m.startswith("finite_alpha_")]
    best_finite_idx = min(finite_indices, key=lambda i: aggregate[i])
    ratio = aggregate[best_finite_idx] / max(aggregate[old_idx], 1e-12)
    verdict = "yes" if ratio < 1.0 else "no"
    lines.append("## Main Question")
    lines.append(
        f"- Does finite posterior assimilation beat the infinitesimal covariance-gradient update? "
        f"**{verdict}** in this reduced sweep."
    )
    lines.append(
        f"- Best finite method: `{methods[best_finite_idx]}` with median posterior-mean MSE "
        f"`{aggregate[best_finite_idx]:.4e}`."
    )
    lines.append(f"- Old covariance-gradient median posterior-mean MSE: `{aggregate[old_idx]:.4e}`.")
    lines.append(f"- Best finite / old covariance-gradient ratio: `{ratio:.3f}`.")
    lines.append("")

    lines.append("## By Condition")
    for ci, (d, eta, beta) in enumerate(conditions):
        values = np.nanmedian(final_metrics[ci, :, :, pm], axis=0)
        best = int(np.nanargmin(values))
        old = values[old_idx]
        best_finite = int(min(finite_indices, key=lambda i: values[i]))
        lines.append(
            f"- d={int(d)}, eta={eta:g}, beta={beta:g}: best `{methods[best]}` "
            f"pmse `{values[best]:.4e}`; best finite `{methods[best_finite]}` "
            f"pmse `{values[best_finite]:.4e}`; old cov-grad `{old:.4e}`."
        )
    lines.append("")

    lines.append("## Aggregate Measurement MSE And Negative Log Joint")
    for label, idx in [("measurement_mse", meas), ("neg_log_joint", nlj)]:
        values = np.nanmedian(final_metrics[..., idx], axis=(0, 1))
        lines.append(f"### {label}")
        for mi in np.argsort(values):
            lines.append(f"- `{methods[mi]}`: `{values[mi]:.4e}`")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="toy_blind_splitting/results/finite_assimilation_reduced")
    parser.add_argument("--prior", choices=["ellipse", "subspace", "full"], default="ellipse")
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--d-values", default="12,50")
    parser.add_argument("--measurement-ratio", type=float, default=0.5)
    parser.add_argument("--eta-values", default="1e-3,3e-3,1e-2")
    parser.add_argument("--beta-values", default="0.0")
    parser.add_argument("--alphas", default="0.1,0.3,1.0,3.0,10.0")
    parser.add_argument("--lifts", default="residual,fresh,cov_shrink")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=100)
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
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    data_dir = ensure_dir(out_dir / "data")
    checkpoint_path = data_dir / f"finite_assimilation_{args.prior}_partial.npz"
    final_path = data_dir / f"finite_assimilation_{args.prior}.npz"
    report_path = data_dir / "finite_assimilation_report.md"

    d_values = _parse_int_list(args.d_values)
    eta_values = _parse_float_list(args.eta_values)
    beta_values = _parse_float_list(args.beta_values)
    alphas = _parse_float_list(args.alphas)
    lifts = _parse_str_list(args.lifts)
    methods = _method_names(alphas, lifts)
    conditions = _condition_table(d_values, eta_values, beta_values)
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)

    shape = (conditions.shape[0], args.n_trials, len(methods))
    existing = _load_npz(checkpoint_path) if args.resume else None
    if existing is None:
        final_metrics = np.full(shape + (len(FINAL_METRIC_NAMES),), np.nan, dtype=np.float32)
        histories = np.full(
            shape + (args.n_steps + 1, len(HISTORY_METRIC_NAMES)),
            np.nan,
            dtype=np.float32,
        )
        trajectory_times = np.full(shape, np.nan, dtype=np.float32)
        method_times = np.full((conditions.shape[0], len(methods)), np.nan, dtype=np.float32)
        condition_times = np.full(conditions.shape[0], np.nan, dtype=np.float32)
        completed = np.zeros(conditions.shape[0], dtype=bool)
    else:
        final_metrics = existing["final_metrics"]
        histories = existing["histories"]
        trajectory_times = existing["trajectory_times"]
        method_times = existing["method_times"]
        condition_times = existing["condition_times"]
        completed = existing["completed"].astype(bool)

    sweep_start = time.perf_counter()
    total_trajectories = conditions.shape[0] * args.n_trials * len(methods)

    for ci, (d_float, eta, beta) in enumerate(conditions):
        if completed[ci]:
            continue
        d = int(d_float)
        m = max(1, int(round(args.measurement_ratio * d)))
        condition_start = time.perf_counter()
        prior_rng = np.random.default_rng(args.seed + 1000 * ci + d)
        prior = _make_prior(args.prior, d, prior_rng, sigma_grid, args.curve_components)
        schedule = log_schedule(args.sigma0, args.schedule_sigma_min, args.n_steps)
        prior.precompute_sigma_grid()
        prior.precompute_sigmas(schedule)

        print(
            f"starting condition {ci + 1}/{conditions.shape[0]}: d={d}, "
            f"ratio={args.measurement_ratio:g}, eta={eta:g}, beta={beta:g}, "
            f"trials={args.n_trials}, methods={len(methods)}",
            flush=True,
        )

        payloads = []
        for trial in range(args.n_trials):
            payloads.append(
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
                }
            )

        done_trials = 0
        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                futures = [pool.submit(_run_trial, payload) for payload in payloads]
                for future in as_completed(futures):
                    result = future.result()
                    trial = int(result["trial_index"])
                    final_metrics[ci, trial] = result["final_metrics"]
                    histories[ci, trial] = result["histories"]
                    trajectory_times[ci, trial] = result["method_times"]
                    done_trials += 1
                    print(f"  completed trial {done_trials}/{args.n_trials} for condition {ci + 1}", flush=True)
        else:
            for payload in payloads:
                result = _run_trial(payload)
                trial = int(result["trial_index"])
                final_metrics[ci, trial] = result["final_metrics"]
                histories[ci, trial] = result["histories"]
                trajectory_times[ci, trial] = result["method_times"]
                done_trials += 1
                print(f"  completed trial {done_trials}/{args.n_trials} for condition {ci + 1}", flush=True)

        condition_times[ci] = time.perf_counter() - condition_start
        method_times[ci] = np.nanmean(trajectory_times[ci], axis=0)
        completed[ci] = True
        completed_trajectories = int((ci + 1) * args.n_trials * len(methods))
        elapsed = time.perf_counter() - sweep_start
        avg = elapsed / max(completed_trajectories, 1)
        remaining = total_trajectories - completed_trajectories
        eta_remaining = avg * remaining
        print(
            f"completed condition {ci + 1}/{conditions.shape[0]}; "
            f"condition_time={_format_seconds(condition_times[ci])}; "
            f"avg={avg:.3f}s/trajectory; elapsed={_format_seconds(elapsed)}; "
            f"eta_remaining={_format_seconds(eta_remaining)}",
            flush=True,
        )

        payload = {
            "final_metrics": final_metrics,
            "histories": histories,
            "trajectory_times": trajectory_times,
            "method_times": method_times,
            "condition_times": condition_times,
            "completed": completed,
            "conditions": conditions,
            "method_names": np.asarray(methods),
            "final_metric_names": np.asarray(FINAL_METRIC_NAMES),
            "history_metric_names": np.asarray(HISTORY_METRIC_NAMES),
            "d_values": np.asarray(d_values),
            "measurement_ratio": np.asarray(args.measurement_ratio),
            "eta_values": np.asarray(eta_values),
            "beta_values": np.asarray(beta_values),
            "alphas": np.asarray(alphas),
            "lifts": np.asarray(lifts),
            "n_steps": np.asarray(args.n_steps),
            "n_trials": np.asarray(args.n_trials),
        }
        _save_npz(checkpoint_path, **payload)

    payload = {
        "final_metrics": final_metrics,
        "histories": histories,
        "trajectory_times": trajectory_times,
        "method_times": method_times,
        "condition_times": condition_times,
        "completed": completed,
        "conditions": conditions,
        "method_names": np.asarray(methods),
        "final_metric_names": np.asarray(FINAL_METRIC_NAMES),
        "history_metric_names": np.asarray(HISTORY_METRIC_NAMES),
        "d_values": np.asarray(d_values),
        "measurement_ratio": np.asarray(args.measurement_ratio),
        "eta_values": np.asarray(eta_values),
        "beta_values": np.asarray(beta_values),
        "alphas": np.asarray(alphas),
        "lifts": np.asarray(lifts),
        "n_steps": np.asarray(args.n_steps),
        "n_trials": np.asarray(args.n_trials),
    }
    _save_npz(final_path, **payload)
    _write_report(report_path, final_metrics, conditions, methods)
    print(f"saved final results to {final_path}", flush=True)
    print(f"saved report to {report_path}", flush=True)


if __name__ == "__main__":
    main()

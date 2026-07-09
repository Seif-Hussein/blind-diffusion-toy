"""Light high-dimensional tests for first-order noisy likelihood tilt.

This script avoids full dense covariance matrices and exact inverse-posterior
diagnostics. It uses a closed-form low-rank-plus-isotropic ellipse mixture so
high ambient dimensions can be tested cheaply.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .plotting import ensure_dir


LOG2PI = np.log(2.0 * np.pi)


FINAL_METRIC_NAMES = [
    "mse_true",
    "measurement_mse",
    "neg_log_joint",
    "final_sigma_hat",
    "median_sigma_hat",
    "min_sigma_hat",
    "max_sigma_hat",
]


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _orthonormal_matrix(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q[:, :k]


@dataclass
class FastEllipsePrior:
    d: int
    n_components: int
    sigma_grid: np.ndarray
    U: np.ndarray
    latent_means: np.ndarray
    means: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    eps: float
    tangent_std: float
    normal_std: float

    @classmethod
    def make(
        cls,
        d: int,
        n_components: int,
        sigma_grid: np.ndarray,
        rng: np.random.Generator,
        r1: float = 2.0,
        r2: float = 0.9,
        eps: float = 0.02,
        tangent_std: float = 0.22,
        normal_std: float = 0.035,
    ) -> "FastEllipsePrior":
        U = _orthonormal_matrix(d, 2, rng)
        angles = np.linspace(0.0, 2.0 * np.pi, n_components, endpoint=False)
        latent = np.column_stack([r1 * np.cos(angles), r2 * np.sin(angles)])
        means = latent @ U.T
        tangents = np.empty((n_components, d), dtype=float)
        normals = np.empty((n_components, d), dtype=float)
        for i, theta in enumerate(angles):
            t_lat = np.array([-r1 * np.sin(theta), r2 * np.cos(theta)])
            n_lat = np.array([np.cos(theta) / r1, np.sin(theta) / r2])
            t_lat /= np.linalg.norm(t_lat)
            n_lat /= np.linalg.norm(n_lat)
            tangents[i] = U @ t_lat
            normals[i] = U @ n_lat
        return cls(
            d=d,
            n_components=n_components,
            sigma_grid=np.asarray(sigma_grid, dtype=float),
            U=U,
            latent_means=latent,
            means=means,
            tangents=tangents,
            normals=normals,
            weights=np.ones(n_components) / n_components,
            eps=eps,
            tangent_std=tangent_std,
            normal_std=normal_std,
        )

    @property
    def mean(self) -> np.ndarray:
        return self.weights @ self.means

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        ids = rng.choice(self.n_components, size=n, p=self.weights)
        x = self.means[ids] + self.eps * rng.normal(size=(n, self.d))
        zt = rng.normal(size=n) * self.tangent_std
        zn = rng.normal(size=n) * self.normal_std
        x += zt[:, None] * self.tangents[ids]
        x += zn[:, None] * self.normals[ids]
        return x, ids

    def _component_log_terms(self, y: np.ndarray, sigma: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[None, :]
        lam = self.eps**2 + float(sigma) ** 2
        vt = self.tangent_std**2
        vn = self.normal_std**2
        logdet = (self.d - 2) * np.log(lam) + np.log(lam + vt) + np.log(lam + vn)
        out = np.empty((y.shape[0], self.n_components), dtype=float)
        for i in range(self.n_components):
            diff = y - self.means[i]
            norm2 = np.sum(diff**2, axis=1)
            tp = diff @ self.tangents[i]
            npj = diff @ self.normals[i]
            mahal = (
                norm2 / lam
                - (vt * tp**2) / (lam * (lam + vt))
                - (vn * npj**2) / (lam * (lam + vn))
            )
            out[:, i] = np.log(self.weights[i]) - 0.5 * (self.d * LOG2PI + logdet + mahal)
        return out

    def log_p_sigma(self, y: np.ndarray, sigma: float) -> np.ndarray:
        return logsumexp(self._component_log_terms(y, sigma), axis=1)

    def sigma_mle(self, y: np.ndarray) -> np.ndarray | float:
        y = np.asarray(y, dtype=float)
        single = y.ndim == 1
        if single:
            y = y[None, :]
        scores = np.empty((y.shape[0], self.sigma_grid.size), dtype=float)
        log_prior = -np.log(np.maximum(self.sigma_grid, 1e-300))
        for j, sigma in enumerate(self.sigma_grid):
            scores[:, j] = self.log_p_sigma(y, float(sigma)) + log_prior[j]
        sigmas = self.sigma_grid[np.argmax(scores, axis=1)]
        return float(sigmas[0]) if single else sigmas

    def posterior_weights(self, y: np.ndarray, sigma: float) -> np.ndarray:
        terms = self._component_log_terms(y, sigma)
        terms -= logsumexp(terms, axis=1, keepdims=True)
        return np.exp(terms)

    def component_posterior_means(self, y: np.ndarray, sigma: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[None, :]
        var_a = self.eps**2
        gt = (var_a + self.tangent_std**2) / (var_a + self.tangent_std**2 + sigma**2)
        gn = (var_a + self.normal_std**2) / (var_a + self.normal_std**2 + sigma**2)
        ga = var_a / (var_a + sigma**2)
        mus = np.empty((y.shape[0], self.n_components, self.d), dtype=float)
        for i in range(self.n_components):
            diff = y - self.means[i]
            tp = diff @ self.tangents[i]
            npj = diff @ self.normals[i]
            mus[:, i, :] = (
                self.means[i]
                + ga * diff
                + (gt - ga) * tp[:, None] * self.tangents[i]
                + (gn - ga) * npj[:, None] * self.normals[i]
            )
        return mus

    def denoise(self, y: np.ndarray, sigma: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        single = y.ndim == 1
        if single:
            y = y[None, :]
        w = self.posterior_weights(y, sigma)
        mus = self.component_posterior_means(y, sigma)
        out = np.einsum("nk,nkd->nd", w, mus)
        return out[0] if single else out

    def log_prior_density(self, x: np.ndarray) -> float:
        return float(self.log_p_sigma(np.asarray(x, dtype=float)[None, :], 0.0)[0])


def make_measurement(
    prior: FastEllipsePrior,
    ratio: float,
    noise_std: float,
    rng: np.random.Generator,
) -> dict:
    m = max(1, int(round(ratio * prior.d)))
    A = rng.normal(scale=1.0 / np.sqrt(m), size=(m, prior.d))
    x_true = prior.sample(1, rng)[0][0]
    y_obs = A @ x_true + noise_std * rng.normal(size=m)
    return {"A": A, "x_true": x_true, "y_obs": y_obs, "noise_std": noise_std}


def data_gradient(x: np.ndarray, A: np.ndarray, y_obs: np.ndarray, noise_std: float) -> np.ndarray:
    return A.T @ (A @ x - y_obs) / (noise_std**2)


def neg_log_joint(prior: FastEllipsePrior, x: np.ndarray, A: np.ndarray, y_obs: np.ndarray, noise_std: float) -> float:
    residual = A @ x - y_obs
    data = 0.5 * np.sum(residual**2) / (noise_std**2)
    data += y_obs.size * (np.log(noise_std) + 0.5 * LOG2PI)
    return float(data - prior.log_prior_density(x))


def summarize(
    prior: FastEllipsePrior,
    x: np.ndarray,
    x_true: np.ndarray,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    sigma_history: list[float],
) -> np.ndarray:
    sig = np.asarray([s for s in sigma_history if np.isfinite(s)], dtype=float)
    return np.asarray(
        [
            np.mean((x - x_true) ** 2),
            np.mean((A @ x - y_obs) ** 2),
            neg_log_joint(prior, x, A, y_obs, noise_std),
            sig[-1] if sig.size else np.nan,
            np.median(sig) if sig.size else np.nan,
            np.min(sig) if sig.size else np.nan,
            np.max(sig) if sig.size else np.nan,
        ],
        dtype=np.float32,
    )


def scheduled_likelihood_tilt(
    prior: FastEllipsePrior,
    meas: dict,
    schedule: np.ndarray,
    eta: float,
    h: float,
    rng: np.random.Generator,
    sigma_probe_stride: int,
) -> tuple[np.ndarray, list[float]]:
    Y = prior.mean + schedule[0] * rng.normal(size=prior.d)
    sigma_hist = [float(prior.sigma_mle(Y))]
    for k, sigma in enumerate(schedule):
        x = prior.denoise(Y, float(sigma))
        g = data_gradient(x, meas["A"], meas["y_obs"], meas["noise_std"])
        Y_tilde = Y - eta * float(sigma) ** 2 * g
        x_tilde = prior.denoise(Y_tilde, float(sigma))
        Y = Y_tilde + h * (x_tilde - Y_tilde)
        if (k + 1) % sigma_probe_stride == 0 or k + 1 == schedule.size:
            sigma_hist.append(float(prior.sigma_mle(Y)))
    return prior.denoise(Y, float(schedule[-1])), sigma_hist


def blind_mle_likelihood_tilt(
    prior: FastEllipsePrior,
    meas: dict,
    n_steps: int,
    eta: float,
    h: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[float]]:
    Y = prior.mean + prior.sigma_grid[-1] * rng.normal(size=prior.d)
    sigma_hist = []
    for _ in range(n_steps):
        sigma = float(prior.sigma_mle(Y))
        sigma_hist.append(sigma)
        x = prior.denoise(Y, sigma)
        g = data_gradient(x, meas["A"], meas["y_obs"], meas["noise_std"])
        Y_tilde = Y - eta * sigma**2 * g
        sigma_tilde = float(prior.sigma_mle(Y_tilde))
        x_tilde = prior.denoise(Y_tilde, sigma_tilde)
        Y = Y_tilde + h * (x_tilde - Y_tilde)
    sigma_hist.append(float(prior.sigma_mle(Y)))
    return prior.denoise(Y, sigma_hist[-1]), sigma_hist


def unscaled_noisy_shift(
    prior: FastEllipsePrior,
    meas: dict,
    schedule: np.ndarray,
    eta: float,
    h: float,
    rng: np.random.Generator,
    sigma_probe_stride: int,
) -> tuple[np.ndarray, list[float]]:
    Y = prior.mean + schedule[0] * rng.normal(size=prior.d)
    sigma_hist = [float(prior.sigma_mle(Y))]
    for k, sigma in enumerate(schedule):
        x = prior.denoise(Y, float(sigma))
        g = data_gradient(x, meas["A"], meas["y_obs"], meas["noise_std"])
        Y_tilde = Y - eta * g
        x_tilde = prior.denoise(Y_tilde, float(sigma))
        Y = Y_tilde + h * (x_tilde - Y_tilde)
        if (k + 1) % sigma_probe_stride == 0 or k + 1 == schedule.size:
            sigma_hist.append(float(prior.sigma_mle(Y)))
    return prior.denoise(Y, float(schedule[-1])), sigma_hist


def raw_pnp_tuned(
    prior: FastEllipsePrior,
    meas: dict,
    schedule: np.ndarray,
    eta: float,
    raw_step_cap: float,
    rng: np.random.Generator,
    sigma_probe_stride: int,
) -> tuple[np.ndarray, list[float]]:
    Y0 = prior.mean + schedule[0] * rng.normal(size=prior.d)
    x = prior.denoise(Y0, float(schedule[0]))
    sigma_hist = [float(prior.sigma_mle(x))]
    step = min(eta, raw_step_cap)
    for k, sigma in enumerate(schedule):
        g = data_gradient(x, meas["A"], meas["y_obs"], meas["noise_std"])
        x = prior.denoise(x - step * g, float(sigma))
        if (k + 1) % sigma_probe_stride == 0 or k + 1 == schedule.size:
            sigma_hist.append(float(prior.sigma_mle(x)))
    return x, sigma_hist


def run_method(
    method: str,
    prior: FastEllipsePrior,
    meas: dict,
    schedule: np.ndarray,
    eta: float,
    h: float,
    raw_step_cap: float,
    rng: np.random.Generator,
    sigma_probe_stride: int,
) -> tuple[np.ndarray, list[float]]:
    if method == "scheduled_tilt":
        return scheduled_likelihood_tilt(prior, meas, schedule, eta, h, rng, sigma_probe_stride)
    if method == "blind_mle_tilt":
        return blind_mle_likelihood_tilt(prior, meas, schedule.size, eta, h, rng)
    if method == "unscaled_noisy":
        return unscaled_noisy_shift(prior, meas, schedule, eta, h, rng, sigma_probe_stride)
    if method == "raw_pnp_tuned":
        return raw_pnp_tuned(prior, meas, schedule, eta, raw_step_cap, rng, sigma_probe_stride)
    raise ValueError(f"unknown method: {method}")


def write_report(path: Path, payload: dict) -> None:
    metrics = payload["metrics"]
    d_values = payload["d_values"]
    eta_values = payload["eta_values"]
    methods = [str(x) for x in payload["method_names"]]
    metric_names = [str(x) for x in payload["metric_names"]]
    mse = metric_names.index("mse_true")
    meas = metric_names.index("measurement_mse")
    nlj = metric_names.index("neg_log_joint")

    lines = ["# High-D Likelihood Tilt Report", ""]
    lines.append("## Aggregate Median MSE To Truth")
    aggregate = np.nanmedian(metrics[..., mse], axis=(0, 1, 2))
    for i in np.argsort(aggregate):
        lines.append(f"- `{methods[i]}`: `{aggregate[i]:.4e}`")
    lines.append("")

    lines.append("## By Dimension")
    for di, d in enumerate(d_values):
        values = np.nanmedian(metrics[di, ..., mse], axis=(0, 1))
        lines.append(f"### d={int(d)}")
        for i in np.argsort(values):
            lines.append(
                f"- `{methods[i]}` mse `{values[i]:.4e}`, "
                f"meas `{np.nanmedian(metrics[di, :, :, i, meas]):.4e}`, "
                f"nlj `{np.nanmedian(metrics[di, :, :, i, nlj]):.4e}`"
            )
        lines.append("")

    lines.append("## By Eta")
    for ei, eta in enumerate(eta_values):
        values = np.nanmedian(metrics[:, ei, :, :, mse], axis=(0, 1))
        lines.append(f"### eta={eta:g}")
        for i in np.argsort(values):
            lines.append(f"- `{methods[i]}`: `{values[i]:.4e}`")
        lines.append("")

    if "blind_mle_tilt" in methods and "scheduled_tilt" in methods:
        b = methods.index("blind_mle_tilt")
        s = methods.index("scheduled_tilt")
        ratio = np.nanmedian(metrics[..., b, mse] / np.maximum(metrics[..., s, mse], 1e-12))
        lines.append(f"Blind/scheduled median MSE ratio: `{ratio:.3f}`.")
    if "unscaled_noisy" in methods and "scheduled_tilt" in methods:
        u = methods.index("unscaled_noisy")
        s = methods.index("scheduled_tilt")
        ratio = np.nanmedian(metrics[..., u, mse] / np.maximum(metrics[..., s, mse], 1e-12))
        lines.append(f"Unscaled-noisy/scheduled median MSE ratio: `{ratio:.3f}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="toy_blind_splitting/results/highd_likelihood_tilt")
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--d-values", default="50,100,200,500,1000")
    parser.add_argument("--measurement-ratio", type=float, default=0.5)
    parser.add_argument("--eta-values", default="1e-3,3e-3,1e-2")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--schedule-sigma-min", type=float, default=0.02)
    parser.add_argument("--sigma0", type=float, default=1.2)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--raw-step-cap", type=float, default=3e-4)
    parser.add_argument("--methods", default="scheduled_tilt,blind_mle_tilt,raw_pnp_tuned,unscaled_noisy")
    parser.add_argument("--sigma-probe-stride", type=int, default=10)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    data_dir = ensure_dir(out_dir / "data")
    d_values = _parse_int_list(args.d_values)
    eta_values = _parse_float_list(args.eta_values)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)
    schedule = np.geomspace(args.sigma0, args.schedule_sigma_min, args.n_steps)

    metrics = np.full(
        (len(d_values), len(eta_values), args.n_trials, len(methods), len(FINAL_METRIC_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    sigma_summaries = np.full(
        (len(d_values), len(eta_values), args.n_trials, len(methods), 3),
        np.nan,
        dtype=np.float32,
    )
    condition_times = np.full((len(d_values), len(eta_values)), np.nan, dtype=np.float32)

    total_conditions = len(d_values) * len(eta_values)
    completed = 0
    start_all = time.perf_counter()
    for di, d in enumerate(d_values):
        prior_rng = np.random.default_rng(args.seed + 1000 * di)
        prior = FastEllipsePrior.make(d, args.components, sigma_grid, prior_rng)
        for ei, eta in enumerate(eta_values):
            completed += 1
            start = time.perf_counter()
            print(
                f"condition {completed}/{total_conditions}: d={d}, ratio={args.measurement_ratio:g}, "
                f"eta={eta:g}, trials={args.n_trials}, methods={len(methods)}",
                flush=True,
            )
            for trial in range(args.n_trials):
                trial_seed = args.seed + 100000 * di + 1000 * ei + trial
                meas = make_measurement(prior, args.measurement_ratio, args.noise_std, np.random.default_rng(trial_seed))
                for mi, method in enumerate(methods):
                    xhat, sigmas = run_method(
                        method,
                        prior,
                        meas,
                        schedule,
                        eta,
                        args.h,
                        args.raw_step_cap,
                        np.random.default_rng(trial_seed + 17 + 100 * mi),
                        args.sigma_probe_stride,
                    )
                    metrics[di, ei, trial, mi] = summarize(
                        prior,
                        xhat,
                        meas["x_true"],
                        meas["A"],
                        meas["y_obs"],
                        args.noise_std,
                        sigmas,
                    )
                    sigmas_arr = np.asarray(sigmas, dtype=float)
                    sigma_summaries[di, ei, trial, mi] = [
                        np.median(sigmas_arr),
                        np.min(sigmas_arr),
                        np.max(sigmas_arr),
                    ]
                if (trial + 1) % max(1, args.n_trials // 5) == 0:
                    print(f"  trial {trial + 1}/{args.n_trials}", flush=True)
            condition_times[di, ei] = time.perf_counter() - start
            elapsed = time.perf_counter() - start_all
            avg_condition = elapsed / completed
            eta_remaining = avg_condition * (total_conditions - completed)
            print(
                f"completed condition {completed}/{total_conditions}; "
                f"condition_time={condition_times[di, ei]:.1f}s; "
                f"eta_remaining={eta_remaining / 60.0:.1f}m",
                flush=True,
            )
            np.savez_compressed(
                data_dir / "highd_likelihood_tilt_partial.npz",
                metrics=metrics,
                sigma_summaries=sigma_summaries,
                condition_times=condition_times,
                d_values=np.asarray(d_values),
                eta_values=np.asarray(eta_values),
                method_names=np.asarray(methods),
                metric_names=np.asarray(FINAL_METRIC_NAMES),
                measurement_ratio=np.asarray(args.measurement_ratio),
                n_trials=np.asarray(args.n_trials),
                n_steps=np.asarray(args.n_steps),
            )

    payload = {
        "metrics": metrics,
        "sigma_summaries": sigma_summaries,
        "condition_times": condition_times,
        "d_values": np.asarray(d_values),
        "eta_values": np.asarray(eta_values),
        "method_names": np.asarray(methods),
        "metric_names": np.asarray(FINAL_METRIC_NAMES),
        "measurement_ratio": np.asarray(args.measurement_ratio),
        "n_trials": np.asarray(args.n_trials),
        "n_steps": np.asarray(args.n_steps),
    }
    np.savez_compressed(data_dir / "highd_likelihood_tilt.npz", **payload)
    write_report(data_dir / "highd_likelihood_tilt_report.md", payload)
    print(f"saved results to {data_dir / 'highd_likelihood_tilt.npz'}", flush=True)


if __name__ == "__main__":
    main()


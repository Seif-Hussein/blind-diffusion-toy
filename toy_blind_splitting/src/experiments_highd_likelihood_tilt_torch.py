"""GPU-oriented high-dimensional likelihood-tilt experiments.

This is a PyTorch port of the lightweight high-dimensional experiment. It keeps
the same closed-form Gaussian-mixture prior, but uses low-rank-plus-isotropic
ellipse formulas and batches independent trials on the selected device.

The target is the first-order/noisy-coordinate likelihood-tilt question:

- scheduled noisy likelihood tilt;
- blind MLE noisy likelihood tilt;
- oracle-scale MLE noisy likelihood tilt;
- uncapped and tuned raw clean-space PnP;
- unscaled noisy-coordinate shift.

No full posterior covariance and no Kalman assimilation are computed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOG2PI = math.log(2.0 * math.pi)

FINAL_METRIC_NAMES = [
    "mse_true",
    "measurement_mse",
    "neg_log_joint",
    "final_sigma_hat",
    "median_sigma_hat",
    "min_sigma_hat",
    "max_sigma_hat",
    "sigma_min_hit",
    "sigma_max_hit",
]


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class BatchedMeasurement:
    A: torch.Tensor
    x_true: torch.Tensor
    y_obs: torch.Tensor
    noise_std: float


class FastEllipseTorchPrior:
    """Low-rank-plus-isotropic ellipse Gaussian mixture in PyTorch."""

    def __init__(
        self,
        d: int,
        n_components: int,
        sigma_grid: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        r1: float = 2.0,
        r2: float = 0.9,
        eps: float = 0.02,
        tangent_std: float = 0.22,
        normal_std: float = 0.035,
    ):
        self.d = int(d)
        self.n_components = int(n_components)
        self.device = device
        self.dtype = dtype
        self.sigma_grid = sigma_grid.to(device=device, dtype=dtype)
        self.eps = float(eps)
        self.tangent_std = float(tangent_std)
        self.normal_std = float(normal_std)

        gen = _torch_generator(device, seed)
        q, _ = torch.linalg.qr(torch.randn((d, 2), device=device, dtype=dtype, generator=gen))
        self.U = q[:, :2]
        angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            n_components + 1,
            device=device,
            dtype=dtype,
        )[:-1]
        latent = torch.stack([r1 * torch.cos(angles), r2 * torch.sin(angles)], dim=1)
        self.latent_means = latent
        self.means = latent @ self.U.T

        t_lat = torch.stack([-r1 * torch.sin(angles), r2 * torch.cos(angles)], dim=1)
        n_lat = torch.stack([torch.cos(angles) / r1, torch.sin(angles) / r2], dim=1)
        t_lat = t_lat / torch.linalg.norm(t_lat, dim=1, keepdim=True).clamp_min(1e-30)
        n_lat = n_lat / torch.linalg.norm(n_lat, dim=1, keepdim=True).clamp_min(1e-30)
        self.tangents = t_lat @ self.U.T
        self.normals = n_lat @ self.U.T
        self.weights = torch.full((n_components,), 1.0 / n_components, device=device, dtype=dtype)
        self.log_weights = torch.log(self.weights)
        self._mean = self.weights @ self.means

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    def sample(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.multinomial(self.weights, n, replacement=True, generator=gen)
        x = self.means[ids] + self.eps * torch.randn(
            (n, self.d), device=self.device, dtype=self.dtype, generator=gen
        )
        zt = self.tangent_std * torch.randn((n,), device=self.device, dtype=self.dtype, generator=gen)
        zn = self.normal_std * torch.randn((n,), device=self.device, dtype=self.dtype, generator=gen)
        x = x + zt[:, None] * self.tangents[ids] + zn[:, None] * self.normals[ids]
        return x, ids

    def _component_log_terms_scalar(self, y: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y[None, :]
        sigma_t = torch.as_tensor(sigma, device=self.device, dtype=self.dtype)
        lam = self.eps**2 + sigma_t**2
        vt = self.tangent_std**2
        vn = self.normal_std**2
        logdet = (self.d - 2) * torch.log(lam) + torch.log(lam + vt) + torch.log(lam + vn)
        terms = []
        for i in range(self.n_components):
            diff = y - self.means[i]
            norm2 = torch.sum(diff * diff, dim=1)
            tp = diff @ self.tangents[i]
            npj = diff @ self.normals[i]
            mahal = (
                norm2 / lam
                - (vt * tp.square()) / (lam * (lam + vt))
                - (vn * npj.square()) / (lam * (lam + vn))
            )
            terms.append(self.log_weights[i] - 0.5 * (self.d * LOG2PI + logdet + mahal))
        return torch.stack(terms, dim=1)

    def _sigma_grid_scores(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y[None, :]
        lam = self.eps**2 + self.sigma_grid.square()
        vt = self.tangent_std**2
        vn = self.normal_std**2
        logdet = (self.d - 2) * torch.log(lam) + torch.log(lam + vt) + torch.log(lam + vn)
        scores = torch.full(
            (y.shape[0], self.sigma_grid.numel()),
            -torch.inf,
            device=self.device,
            dtype=self.dtype,
        )
        for i in range(self.n_components):
            diff = y - self.means[i]
            norm2 = torch.sum(diff * diff, dim=1)
            tp2 = (diff @ self.tangents[i]).square()
            np2 = (diff @ self.normals[i]).square()
            mahal = (
                norm2[:, None] / lam[None, :]
                - (vt * tp2[:, None]) / (lam[None, :] * (lam[None, :] + vt))
                - (vn * np2[:, None]) / (lam[None, :] * (lam[None, :] + vn))
            )
            terms = self.log_weights[i] - 0.5 * (self.d * LOG2PI + logdet[None, :] + mahal)
            scores = torch.logaddexp(scores, terms)
        scores = scores - torch.log(self.sigma_grid.clamp_min(1e-30))[None, :]
        return scores

    def sigma_mle_idx(self, y: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self._sigma_grid_scores(y), dim=1)

    def sigma_mle(self, y: torch.Tensor) -> torch.Tensor:
        return self.sigma_grid[self.sigma_mle_idx(y)]

    def posterior_weights(self, y: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        terms = self._component_log_terms_scalar(y, sigma)
        return torch.softmax(terms, dim=1)

    def denoise(self, y: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y[None, :]
        sigma_t = torch.as_tensor(sigma, device=self.device, dtype=self.dtype)
        var_a = self.eps**2
        gt = (var_a + self.tangent_std**2) / (var_a + self.tangent_std**2 + sigma_t**2)
        gn = (var_a + self.normal_std**2) / (var_a + self.normal_std**2 + sigma_t**2)
        ga = var_a / (var_a + sigma_t**2)
        weights = self.posterior_weights(y, sigma_t)
        out = torch.zeros_like(y)
        for i in range(self.n_components):
            diff = y - self.means[i]
            tp = diff @ self.tangents[i]
            npj = diff @ self.normals[i]
            mu_i = (
                self.means[i]
                + ga * diff
                + (gt - ga) * tp[:, None] * self.tangents[i]
                + (gn - ga) * npj[:, None] * self.normals[i]
            )
            out = out + weights[:, i : i + 1] * mu_i
        return out

    def denoise_by_sigma_idx(self, y: torch.Tensor, sigma_idx: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y[None, :]
        out = torch.empty_like(y)
        for idx in torch.unique(sigma_idx):
            mask = sigma_idx == idx
            out[mask] = self.denoise(y[mask], self.sigma_grid[idx])
        return out

    def log_prior_density(self, x: torch.Tensor) -> torch.Tensor:
        terms = self._component_log_terms_scalar(x, torch.zeros((), device=self.device, dtype=self.dtype))
        return torch.logsumexp(terms, dim=1)


def make_measurement_batch(
    prior: FastEllipseTorchPrior,
    batch_size: int,
    ratio: float,
    noise_std: float,
    gen: torch.Generator,
) -> BatchedMeasurement:
    m = max(1, int(round(ratio * prior.d)))
    A = torch.randn(
        (batch_size, m, prior.d),
        device=prior.device,
        dtype=prior.dtype,
        generator=gen,
    ) / math.sqrt(m)
    x_true, _ = prior.sample(batch_size, gen)
    y_obs = torch.bmm(A, x_true[:, :, None]).squeeze(-1)
    y_obs = y_obs + noise_std * torch.randn(
        y_obs.shape, device=prior.device, dtype=prior.dtype, generator=gen
    )
    return BatchedMeasurement(A=A, x_true=x_true, y_obs=y_obs, noise_std=noise_std)


def data_gradient(x: torch.Tensor, meas: BatchedMeasurement) -> torch.Tensor:
    residual = torch.bmm(meas.A, x[:, :, None]).squeeze(-1) - meas.y_obs
    return torch.bmm(meas.A.transpose(1, 2), residual[:, :, None]).squeeze(-1) / (meas.noise_std**2)


def measurement_mse(x: torch.Tensor, meas: BatchedMeasurement) -> torch.Tensor:
    residual = torch.bmm(meas.A, x[:, :, None]).squeeze(-1) - meas.y_obs
    return torch.mean(residual.square(), dim=1)


def negative_log_joint(prior: FastEllipseTorchPrior, x: torch.Tensor, meas: BatchedMeasurement) -> torch.Tensor:
    residual = torch.bmm(meas.A, x[:, :, None]).squeeze(-1) - meas.y_obs
    data = 0.5 * torch.sum(residual.square(), dim=1) / (meas.noise_std**2)
    data = data + meas.y_obs.shape[1] * (math.log(meas.noise_std) + 0.5 * LOG2PI)
    return data - prior.log_prior_density(x)


def sigma_summary(sigmas: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stacked = torch.stack(sigmas, dim=0)
    return stacked[-1], torch.median(stacked, dim=0).values, torch.min(stacked, dim=0).values, torch.max(stacked, dim=0).values


def summarize_batch(
    prior: FastEllipseTorchPrior,
    x: torch.Tensor,
    meas: BatchedMeasurement,
    sigmas: list[torch.Tensor],
) -> torch.Tensor:
    final_sigma, median_sigma, min_sigma, max_sigma = sigma_summary(sigmas)
    sigma_min_hit = (min_sigma <= prior.sigma_grid[0] * 1.000001).to(x.dtype)
    sigma_max_hit = (max_sigma >= prior.sigma_grid[-1] * 0.999999).to(x.dtype)
    return torch.stack(
        [
            torch.mean((x - meas.x_true).square(), dim=1),
            measurement_mse(x, meas),
            negative_log_joint(prior, x, meas),
            final_sigma,
            median_sigma,
            min_sigma,
            max_sigma,
            sigma_min_hit,
            sigma_max_hit,
        ],
        dim=1,
    )


def scheduled_tilt(
    prior: FastEllipseTorchPrior,
    meas: BatchedMeasurement,
    schedule: torch.Tensor,
    eta: float,
    h: float,
    gen: torch.Generator,
    sigma_probe_stride: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    Y = prior.mean[None, :] + schedule[0] * torch.randn(
        meas.x_true.shape, device=prior.device, dtype=prior.dtype, generator=gen
    )
    sigmas = [prior.sigma_mle(Y)]
    for step, sigma in enumerate(schedule):
        x = prior.denoise(Y, sigma)
        g = data_gradient(x, meas)
        Y_tilde = Y - eta * float(sigma.item()) ** 2 * g
        x_tilde = prior.denoise(Y_tilde, sigma)
        Y = Y_tilde + h * (x_tilde - Y_tilde)
        if (step + 1) % sigma_probe_stride == 0 or step + 1 == schedule.numel():
            sigmas.append(prior.sigma_mle(Y))
    return prior.denoise(Y, schedule[-1]), sigmas


def blind_mle_tilt(
    prior: FastEllipseTorchPrior,
    meas: BatchedMeasurement,
    n_steps: int,
    eta: float,
    h: float,
    init_sigma: float | torch.Tensor,
    gen: torch.Generator,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    init_sigma_t = torch.as_tensor(init_sigma, device=prior.device, dtype=prior.dtype)
    Y = prior.mean[None, :] + init_sigma_t * torch.randn(
        meas.x_true.shape, device=prior.device, dtype=prior.dtype, generator=gen
    )
    sigmas = []
    for _ in range(n_steps):
        idx = prior.sigma_mle_idx(Y)
        sigma = prior.sigma_grid[idx]
        sigmas.append(sigma)
        x = prior.denoise_by_sigma_idx(Y, idx)
        g = data_gradient(x, meas)
        Y_tilde = Y - eta * sigma[:, None].square() * g
        idx_tilde = prior.sigma_mle_idx(Y_tilde)
        x_tilde = prior.denoise_by_sigma_idx(Y_tilde, idx_tilde)
        Y = Y_tilde + h * (x_tilde - Y_tilde)
    sigmas.append(prior.sigma_mle(Y))
    return prior.denoise_by_sigma_idx(Y, prior.sigma_mle_idx(Y)), sigmas


def oracle_mle_tilt(
    prior: FastEllipseTorchPrior,
    meas: BatchedMeasurement,
    n_steps: int,
    eta: float,
    h: float,
    init_sigma: float | torch.Tensor,
    gen: torch.Generator,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # For the closed-form MLE blind denoiser, the oracle-scale nonblind update
    # is the same computation with the inferred sigma passed explicitly.
    return blind_mle_tilt(prior, meas, n_steps, eta, h, init_sigma, gen)


def unscaled_noisy(
    prior: FastEllipseTorchPrior,
    meas: BatchedMeasurement,
    schedule: torch.Tensor,
    eta: float,
    h: float,
    gen: torch.Generator,
    sigma_probe_stride: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    Y = prior.mean[None, :] + schedule[0] * torch.randn(
        meas.x_true.shape, device=prior.device, dtype=prior.dtype, generator=gen
    )
    sigmas = [prior.sigma_mle(Y)]
    for step, sigma in enumerate(schedule):
        x = prior.denoise(Y, sigma)
        g = data_gradient(x, meas)
        Y_tilde = Y - eta * g
        x_tilde = prior.denoise(Y_tilde, sigma)
        Y = Y_tilde + h * (x_tilde - Y_tilde)
        if (step + 1) % sigma_probe_stride == 0 or step + 1 == schedule.numel():
            sigmas.append(prior.sigma_mle(Y))
    return prior.denoise(Y, schedule[-1]), sigmas


def raw_pnp_tuned(
    prior: FastEllipseTorchPrior,
    meas: BatchedMeasurement,
    schedule: torch.Tensor,
    eta: float,
    raw_step_cap: float,
    gen: torch.Generator,
    sigma_probe_stride: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    Y0 = prior.mean[None, :] + schedule[0] * torch.randn(
        meas.x_true.shape, device=prior.device, dtype=prior.dtype, generator=gen
    )
    x = prior.denoise(Y0, schedule[0])
    sigmas = [prior.sigma_mle(x)]
    step_size = min(float(eta), float(raw_step_cap))
    for step, sigma in enumerate(schedule):
        g = data_gradient(x, meas)
        x = prior.denoise(x - step_size * g, sigma)
        if (step + 1) % sigma_probe_stride == 0 or step + 1 == schedule.numel():
            sigmas.append(prior.sigma_mle(x))
    return x, sigmas


def raw_pnp_uncapped(
    prior: FastEllipseTorchPrior,
    meas: BatchedMeasurement,
    schedule: torch.Tensor,
    eta: float,
    gen: torch.Generator,
    sigma_probe_stride: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    Y0 = prior.mean[None, :] + schedule[0] * torch.randn(
        meas.x_true.shape, device=prior.device, dtype=prior.dtype, generator=gen
    )
    x = prior.denoise(Y0, schedule[0])
    sigmas = [prior.sigma_mle(x)]
    for step, sigma in enumerate(schedule):
        g = data_gradient(x, meas)
        x = prior.denoise(x - float(eta) * g, sigma)
        if (step + 1) % sigma_probe_stride == 0 or step + 1 == schedule.numel():
            sigmas.append(prior.sigma_mle(x))
    return x, sigmas


def run_method(
    method: str,
    prior: FastEllipseTorchPrior,
    meas: BatchedMeasurement,
    schedule: torch.Tensor,
    eta: float,
    h: float,
    raw_step_cap: float,
    gen: torch.Generator,
    sigma_probe_stride: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if method == "scheduled_tilt":
        return scheduled_tilt(prior, meas, schedule, eta, h, gen, sigma_probe_stride)
    if method == "blind_mle_tilt":
        return blind_mle_tilt(prior, meas, schedule.numel(), eta, h, schedule[0], gen)
    if method == "oracle_mle_tilt":
        return oracle_mle_tilt(prior, meas, schedule.numel(), eta, h, schedule[0], gen)
    if method == "raw_pnp_uncapped":
        return raw_pnp_uncapped(prior, meas, schedule, eta, gen, sigma_probe_stride)
    if method == "raw_pnp_tuned":
        return raw_pnp_tuned(prior, meas, schedule, eta, raw_step_cap, gen, sigma_probe_stride)
    if method == "unscaled_noisy":
        return unscaled_noisy(prior, meas, schedule, eta, h, gen, sigma_probe_stride)
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

    lines = ["# GPU High-D Likelihood Tilt Report", ""]
    lines.append(f"Device: `{payload['device']}`")
    lines.append("")
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
    if "blind_mle_tilt" in methods and "oracle_mle_tilt" in methods:
        b = methods.index("blind_mle_tilt")
        o = methods.index("oracle_mle_tilt")
        ratio = np.nanmedian(metrics[..., b, mse] / np.maximum(metrics[..., o, mse], 1e-12))
        lines.append(f"Blind/oracle-MLE median MSE ratio: `{ratio:.3f}`.")
    if "raw_pnp_uncapped" in methods and "blind_mle_tilt" in methods:
        r = methods.index("raw_pnp_uncapped")
        b = methods.index("blind_mle_tilt")
        ratio = np.nanmedian(metrics[..., b, mse] / np.maximum(metrics[..., r, mse], 1e-12))
        lines.append(f"Blind/raw-PnP-uncapped median MSE ratio: `{ratio:.3f}`.")
    if "raw_pnp_tuned" in methods and "blind_mle_tilt" in methods:
        r = methods.index("raw_pnp_tuned")
        b = methods.index("blind_mle_tilt")
        ratio = np.nanmedian(metrics[..., b, mse] / np.maximum(metrics[..., r, mse], 1e-12))
        lines.append(f"Blind/raw-PnP-tuned median MSE ratio: `{ratio:.3f}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_tables(data_dir: Path, payload: dict) -> None:
    metrics = payload["metrics"]
    d_values = payload["d_values"]
    eta_values = payload["eta_values"]
    methods = [str(x) for x in payload["method_names"]]
    metric_names = [str(x) for x in payload["metric_names"]]

    condition_path = data_dir / "summary_by_condition.csv"
    with condition_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["d", "eta", "method", "n_trials"] + [f"median_{name}" for name in metric_names])
        for di, d in enumerate(d_values):
            for ei, eta in enumerate(eta_values):
                for mi, method in enumerate(methods):
                    values = np.nanmedian(metrics[di, ei, :, mi, :], axis=0)
                    writer.writerow([int(d), float(eta), method, metrics.shape[2], *values.tolist()])

    dimension_path = data_dir / "summary_by_dimension.csv"
    with dimension_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["d", "method", "n_eta", "n_trials"] + [f"median_{name}" for name in metric_names])
        for di, d in enumerate(d_values):
            for mi, method in enumerate(methods):
                values = np.nanmedian(metrics[di, :, :, mi, :], axis=(0, 1))
                writer.writerow([int(d), method, len(eta_values), metrics.shape[2], *values.tolist()])

    eta_path = data_dir / "summary_by_eta.csv"
    with eta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eta", "method", "n_d", "n_trials"] + [f"median_{name}" for name in metric_names])
        for ei, eta in enumerate(eta_values):
            for mi, method in enumerate(methods):
                values = np.nanmedian(metrics[:, ei, :, mi, :], axis=(0, 1))
                writer.writerow([float(eta), method, len(d_values), metrics.shape[2], *values.tolist()])


def write_plots(fig_dir: Path, payload: dict) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = payload["metrics"]
    d_values = payload["d_values"]
    eta_values = payload["eta_values"]
    methods = [str(x) for x in payload["method_names"]]
    metric_names = [str(x) for x in payload["metric_names"]]
    mse = metric_names.index("mse_true")
    meas = metric_names.index("measurement_mse")
    nlj = metric_names.index("neg_log_joint")
    sig_med = metric_names.index("median_sigma_hat")
    sig_min_hit = metric_names.index("sigma_min_hit")
    sig_max_hit = metric_names.index("sigma_max_hit")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mi, method in enumerate(methods):
        values = np.nanmedian(metrics[:, :, :, mi, mse], axis=(1, 2))
        ax.plot(d_values, values, marker="o", label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("median MSE to truth")
    ax.set_title("High-D likelihood tilt: MSE by dimension")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "mse_by_dimension.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mi, method in enumerate(methods):
        values = np.nanmedian(metrics[:, :, :, mi, mse], axis=(0, 2))
        ax.plot(eta_values, values, marker="o", label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eta")
    ax.set_ylabel("median MSE to truth")
    ax.set_title("High-D likelihood tilt: MSE by eta")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "mse_by_eta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mi, method in enumerate(methods):
        values = np.nanmedian(metrics[:, :, :, mi, nlj], axis=(0, 2))
        ax.plot(eta_values, values, marker="o", label=method)
    ax.set_xscale("log")
    ax.set_xlabel("eta")
    ax.set_ylabel("median negative log joint")
    ax.set_title("High-D likelihood tilt: joint objective by eta")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "neg_log_joint_by_eta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mi, method in enumerate(methods):
        values = np.nanmedian(metrics[:, :, :, mi, meas], axis=(1, 2))
        ax.plot(d_values, values, marker="o", label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("median measurement MSE")
    ax.set_title("Measurement consistency by dimension")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "measurement_mse_by_dimension.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mi, method in enumerate(methods):
        values = np.nanmedian(metrics[:, :, :, mi, sig_med], axis=(1, 2))
        ax.plot(d_values, values, marker="o", label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("median inferred sigma")
    ax.set_title("Blind scale diagnostic by dimension")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "median_sigma_by_dimension.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mi, method in enumerate(methods):
        min_hit = np.nanmean(metrics[:, :, :, mi, sig_min_hit], axis=(1, 2))
        max_hit = np.nanmean(metrics[:, :, :, mi, sig_max_hit], axis=(1, 2))
        ax.plot(d_values, min_hit, marker="o", linestyle="-", label=f"{method} min")
        ax.plot(d_values, max_hit, marker="s", linestyle="--", label=f"{method} max")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel("boundary hit rate")
    ax.set_title("Sigma-grid boundary hits")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "sigma_boundary_hits.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for mi, method in enumerate(methods):
        min_hit = np.nanmean(metrics[:, :, :, mi, sig_min_hit], axis=(0, 2))
        max_hit = np.nanmean(metrics[:, :, :, mi, sig_max_hit], axis=(0, 2))
        ax.plot(eta_values, min_hit, marker="o", linestyle="-", label=f"{method} min")
        ax.plot(eta_values, max_hit, marker="s", linestyle="--", label=f"{method} max")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("eta")
    ax.set_ylabel("boundary hit rate")
    ax.set_title("Sigma-grid boundary hits by eta")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "sigma_boundary_hits_by_eta.png", dpi=180)
    plt.close(fig)

    def plot_ratio(numer_method: str, denom_method: str, filename: str, title: str) -> None:
        if numer_method not in methods or denom_method not in methods:
            return
        numer = methods.index(numer_method)
        denom = methods.index(denom_method)
        ratio = np.nanmedian(
            metrics[:, :, :, numer, mse] / np.maximum(metrics[:, :, :, denom, mse], 1e-12),
            axis=(0, 2),
        )
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax.plot(eta_values, ratio, marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("eta")
        ax.set_ylabel("median MSE ratio")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    plot_ratio(
        "blind_mle_tilt",
        "scheduled_tilt",
        "blind_vs_scheduled_mse_ratio_by_eta.png",
        "Blind MLE / scheduled MSE by eta",
    )
    plot_ratio(
        "blind_mle_tilt",
        "oracle_mle_tilt",
        "blind_vs_oracle_mse_ratio_by_eta.png",
        "Blind MLE / oracle MLE MSE by eta",
    )
    plot_ratio(
        "blind_mle_tilt",
        "raw_pnp_uncapped",
        "blind_vs_raw_uncapped_mse_ratio_by_eta.png",
        "Blind MLE / raw PnP uncapped MSE by eta",
    )
    plot_ratio(
        "blind_mle_tilt",
        "raw_pnp_tuned",
        "blind_vs_raw_tuned_mse_ratio_by_eta.png",
        "Blind MLE / raw PnP tuned MSE by eta",
    )


def run_experiment(
    *,
    out: str | Path = "toy_blind_splitting/results/highd_likelihood_tilt_gpu",
    seed: int = 909,
    d_values: list[int] | None = None,
    measurement_ratio: float = 0.5,
    eta_values: list[float] | None = None,
    n_trials: int = 32,
    batch_size: int = 16,
    n_steps: int = 80,
    components: int = 16,
    grid_size: int = 49,
    sigma_min: float = 0.005,
    sigma_max: float = 10.0,
    schedule_sigma_min: float = 0.02,
    sigma0: float = 1.2,
    noise_std: float = 0.08,
    h: float = 0.05,
    raw_step_cap: float = 3e-4,
    methods: list[str] | None = None,
    sigma_probe_stride: int = 10,
    device_name: str = "auto",
    dtype_name: str = "float32",
) -> dict:
    device = _device(device_name)
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    d_values = [500, 1000] if d_values is None else d_values
    eta_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2] if eta_values is None else eta_values
    methods = (
        [
            "scheduled_tilt",
            "blind_mle_tilt",
            "oracle_mle_tilt",
            "raw_pnp_uncapped",
            "raw_pnp_tuned",
            "unscaled_noisy",
        ]
        if methods is None
        else methods
    )

    out = Path(out)
    data_dir = out / "data"
    fig_dir = out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sigma_grid = torch.tensor(np.geomspace(sigma_min, sigma_max, grid_size), device=device, dtype=dtype)
    schedule = torch.tensor(np.geomspace(sigma0, schedule_sigma_min, n_steps), device=device, dtype=dtype)

    metrics = np.full(
        (len(d_values), len(eta_values), n_trials, len(methods), len(FINAL_METRIC_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    condition_times = np.full((len(d_values), len(eta_values)), np.nan, dtype=np.float32)

    total_conditions = len(d_values) * len(eta_values)
    completed = 0
    start_all = time.perf_counter()
    with torch.no_grad():
        for di, d in enumerate(d_values):
            prior = FastEllipseTorchPrior(
                d=d,
                n_components=components,
                sigma_grid=sigma_grid,
                device=device,
                dtype=dtype,
                seed=seed + 1000 * di,
            )
            for ei, eta in enumerate(eta_values):
                completed += 1
                start = time.perf_counter()
                print(
                    f"condition {completed}/{total_conditions}: d={d}, ratio={measurement_ratio:g}, "
                    f"eta={eta:g}, trials={n_trials}, batch={batch_size}, methods={len(methods)}, device={device}",
                    flush=True,
                )
                for start_trial in range(0, n_trials, batch_size):
                    current_batch = min(batch_size, n_trials - start_trial)
                    batch_seed = seed + 100000 * di + 1000 * ei + start_trial
                    meas = make_measurement_batch(
                        prior,
                        current_batch,
                        measurement_ratio,
                        noise_std,
                        _torch_generator(device, batch_seed),
                    )
                    for mi, method in enumerate(methods):
                        xhat, sigmas = run_method(
                            method,
                            prior,
                            meas,
                            schedule,
                            eta,
                            h,
                            raw_step_cap,
                            _torch_generator(device, batch_seed + 17),
                            sigma_probe_stride,
                        )
                        batch_metrics = summarize_batch(prior, xhat, meas, sigmas)
                        metrics[di, ei, start_trial : start_trial + current_batch, mi] = (
                            batch_metrics.detach().cpu().numpy().astype(np.float32)
                        )
                    _sync(device)
                    print(f"  trials {start_trial + current_batch}/{n_trials}", flush=True)
                _sync(device)
                condition_times[di, ei] = time.perf_counter() - start
                elapsed = time.perf_counter() - start_all
                avg_condition = elapsed / completed
                remaining = avg_condition * (total_conditions - completed)
                print(
                    f"completed condition {completed}/{total_conditions}; "
                    f"condition_time={condition_times[di, ei]:.1f}s; eta_remaining={remaining / 60.0:.1f}m",
                    flush=True,
                )
                np.savez_compressed(
                    data_dir / "highd_likelihood_tilt_gpu_partial.npz",
                    metrics=metrics,
                    condition_times=condition_times,
                    d_values=np.asarray(d_values),
                    eta_values=np.asarray(eta_values),
                    method_names=np.asarray(methods),
                    metric_names=np.asarray(FINAL_METRIC_NAMES),
                    measurement_ratio=np.asarray(measurement_ratio),
                    n_trials=np.asarray(n_trials),
                    n_steps=np.asarray(n_steps),
                    batch_size=np.asarray(batch_size),
                    device=np.asarray(str(device)),
                )

    payload = {
        "metrics": metrics,
        "condition_times": condition_times,
        "d_values": np.asarray(d_values),
        "eta_values": np.asarray(eta_values),
        "method_names": np.asarray(methods),
        "metric_names": np.asarray(FINAL_METRIC_NAMES),
        "measurement_ratio": np.asarray(measurement_ratio),
        "n_trials": np.asarray(n_trials),
        "n_steps": np.asarray(n_steps),
        "batch_size": np.asarray(batch_size),
        "device": np.asarray(str(device)),
    }
    np.savez_compressed(data_dir / "highd_likelihood_tilt_gpu.npz", **payload)
    write_report(data_dir / "highd_likelihood_tilt_gpu_report.md", payload)
    write_analysis_tables(data_dir, payload)
    write_plots(fig_dir, payload)
    config = {
        "seed": seed,
        "d_values": d_values,
        "measurement_ratio": measurement_ratio,
        "eta_values": eta_values,
        "n_trials": n_trials,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "components": components,
        "grid_size": grid_size,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "schedule_sigma_min": schedule_sigma_min,
        "sigma0": sigma0,
        "noise_std": noise_std,
        "h": h,
        "raw_step_cap": raw_step_cap,
        "methods": methods,
        "sigma_probe_stride": sigma_probe_stride,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
    }
    (data_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"saved results to {data_dir / 'highd_likelihood_tilt_gpu.npz'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="toy_blind_splitting/results/highd_likelihood_tilt_gpu")
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--d-values", default="500,1000")
    parser.add_argument("--measurement-ratio", type=float, default=0.5)
    parser.add_argument("--eta-values", default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    parser.add_argument("--n-trials", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=49)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=10.0)
    parser.add_argument("--schedule-sigma-min", type=float, default=0.02)
    parser.add_argument("--sigma0", type=float, default=1.2)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--raw-step-cap", type=float, default=3e-4)
    parser.add_argument(
        "--methods",
        default="scheduled_tilt,blind_mle_tilt,oracle_mle_tilt,raw_pnp_uncapped,raw_pnp_tuned,unscaled_noisy",
    )
    parser.add_argument("--sigma-probe-stride", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    run_experiment(
        out=args.out,
        seed=args.seed,
        d_values=_parse_int_list(args.d_values),
        measurement_ratio=args.measurement_ratio,
        eta_values=_parse_float_list(args.eta_values),
        n_trials=args.n_trials,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        components=args.components,
        grid_size=args.grid_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        schedule_sigma_min=args.schedule_sigma_min,
        sigma0=args.sigma0,
        noise_std=args.noise_std,
        h=args.h,
        raw_step_cap=args.raw_step_cap,
        methods=_parse_str_list(args.methods),
        sigma_probe_stride=args.sigma_probe_stride,
        device_name=args.device,
        dtype_name=args.dtype,
    )


if __name__ == "__main__":
    main()

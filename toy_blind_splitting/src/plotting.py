"""Matplotlib plotting helpers for the toy experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .metrics import latent_coordinates
from .priors import GaussianMixturePrior


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_scale_histograms(
    d_values: list[int],
    ratios: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(d_values), figsize=(4.0 * len(d_values), 3.0), squeeze=False)
    for ax, d, vals in zip(axes[0], d_values, ratios):
        ax.hist(vals, bins=30, density=True, color="#4777b3", alpha=0.82)
        ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(f"d={d}")
        ax.set_xlabel("sigma_hat^2 / sigma^2")
        ax.set_ylabel("density")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mean_std_vs_ratio(
    d_values: list[int],
    intrinsic_k: int,
    sigma_ratio_mean: np.ndarray,
    sigma_ratio_std: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    x = np.asarray(d_values, dtype=float) / float(intrinsic_k)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.errorbar(x, sigma_ratio_mean, yerr=sigma_ratio_std, marker="o", color="#245c99", capsize=3)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("d / intrinsic_k")
    ax.set_ylabel("sigma_hat / sigma")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_likelihood_identity(
    etas: np.ndarray,
    rel_error_mean: np.ndarray,
    rel_error_std: np.ndarray,
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.errorbar(etas, rel_error_mean, yerr=rel_error_std, marker="o", color="#7b4a9d", capsize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eta")
    ax.set_ylabel("relative error")
    ax.set_title("Likelihood-tilt identity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tangent_filtering(
    raw_ratio: np.ndarray,
    cov_ratio: np.ndarray,
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.boxplot([raw_ratio, cov_ratio], labels=["raw g", "C g"], showfliers=False)
    ax.set_ylabel("normal fraction")
    ax.set_title("Tangent-normal force filtering")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_trajectories(
    trajectories: dict[str, np.ndarray],
    out_path: str | Path,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, values in trajectories.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            mean = arr
            std = np.zeros_like(arr)
        else:
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
        steps = np.arange(mean.size)
        ax.plot(steps, mean, label=name)
        if arr.ndim > 1:
            ax.fill_between(steps, mean - std, mean + std, alpha=0.12)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_final_boxplots(
    metrics: np.ndarray,
    method_names: list[str],
    metric_names: list[str],
    metric_name: str,
    out_path: str | Path,
    title: str,
) -> None:
    idx = metric_names.index(metric_name)
    vals = metrics[..., idx]
    vals = vals.reshape((-1, vals.shape[-1])) if vals.ndim > 2 else vals
    if vals.shape[0] == len(method_names):
        data = [vals[i] for i in range(vals.shape[0])]
    else:
        data = [vals[:, i] for i in range(vals.shape[1])]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(method_names)), 3.8))
    ax.boxplot(data, labels=method_names, showfliers=False)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_latent_trajectories(
    prior: GaussianMixturePrior,
    trajectories: dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "Latent trajectories",
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    latent_means = prior.metadata.get("latent_means")
    if latent_means is not None and latent_means.shape[1] == 2:
        closed = np.vstack([latent_means, latent_means[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.0, alpha=0.55)
        ax.scatter(latent_means[:, 0], latent_means[:, 1], s=8, color="black", alpha=0.4)
    for name, traj in trajectories.items():
        coords = latent_coordinates(prior, np.asarray(traj, dtype=float))
        if coords.shape[1] < 2:
            continue
        ax.plot(coords[:, 0], coords[:, 1], marker="o", markersize=2, linewidth=1.2, label=name)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("latent 1")
    ax.set_ylabel("latent 2")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


"""Metrics and geometry diagnostics for toy experiments."""

from __future__ import annotations

import numpy as np

from .priors import GaussianMixturePrior, LOG2PI


def mse_per_dim(x: np.ndarray, ref: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return float(np.mean((x - ref) ** 2))


def measurement_mse(x: np.ndarray, y_obs: np.ndarray, A: np.ndarray) -> float:
    residual = A @ x - y_obs
    return float(np.mean(residual**2))


def posterior_mean_mse(x: np.ndarray, posterior_mean: np.ndarray) -> float:
    return mse_per_dim(x, posterior_mean)


def negative_log_joint(
    x: np.ndarray,
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    covariance_floor: float = 1e-6,
) -> float:
    residual = A @ x - y_obs
    data_term = 0.5 * np.sum(residual**2) / (noise_std**2)
    data_norm = y_obs.size * (np.log(noise_std) + 0.5 * LOG2PI)
    prior_term = -float(prior.log_prior_density(x, covariance_floor=covariance_floor))
    return float(data_term + data_norm + prior_term)


def nearest_component_distance(prior: GaussianMixturePrior, x: np.ndarray) -> float:
    dists = np.linalg.norm(prior.means - x[None, :], axis=1)
    return float(np.min(dists))


def nearest_component_index(prior: GaussianMixturePrior, x: np.ndarray) -> int:
    dists = np.linalg.norm(prior.means - x[None, :], axis=1)
    return int(np.argmin(dists))


def local_tangent_normal_ratio(prior: GaussianMixturePrior, x: np.ndarray, vector: np.ndarray) -> float:
    """Return ||normal component|| / ||vector|| at the nearest curve component.

    For non-curve priors this returns NaN.
    """

    tangents = prior.metadata.get("ambient_tangents")
    if tangents is None:
        return float("nan")
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-15:
        return 0.0
    idx = nearest_component_index(prior, x)
    tangent = tangents[idx]
    tangent = tangent / max(np.linalg.norm(tangent), 1e-15)
    tangential = np.dot(vector, tangent) * tangent
    normal = vector - tangential
    return float(np.linalg.norm(normal) / norm)


def latent_coordinates(prior: GaussianMixturePrior, x: np.ndarray) -> np.ndarray:
    u = prior.metadata.get("U")
    if u is None:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x @ u
    return x @ u


def summarize_final_metrics(
    x: np.ndarray,
    x_true: np.ndarray,
    posterior_mean: np.ndarray,
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
) -> dict[str, float]:
    return {
        "mse_true": mse_per_dim(x, x_true),
        "measurement_mse": measurement_mse(x, y_obs, A),
        "posterior_mean_mse": posterior_mean_mse(x, posterior_mean),
        "neg_log_joint": negative_log_joint(x, prior, A, y_obs, noise_std),
        "nearest_component_distance": nearest_component_distance(prior, x),
    }


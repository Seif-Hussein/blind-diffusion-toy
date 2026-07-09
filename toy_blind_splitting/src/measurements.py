"""Linear Gaussian inverse-problem utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from .priors import GaussianMixturePrior, LOG2PI


def make_operator(
    d: int,
    m: int,
    A_type: str,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    if A_type == "random":
        return rng.normal(scale=1.0 / np.sqrt(m), size=(m, d))
    if A_type == "mask":
        if m > d:
            raise ValueError("mask measurement count cannot exceed d")
        idx = rng.choice(d, size=m, replace=False)
        A = np.zeros((m, d), dtype=float)
        A[np.arange(m), idx] = 1.0
        return A
    raise ValueError(f"unknown A_type: {A_type}")


def make_measurement(
    prior: GaussianMixturePrior,
    A_type: str,
    m: int,
    noise_std: float,
    rng: Optional[np.random.Generator] = None,
    x_true: Optional[np.ndarray] = None,
) -> dict:
    rng = np.random.default_rng() if rng is None else rng
    A = make_operator(prior.d, m, A_type, rng)
    if x_true is None:
        x_samples, component_ids = prior.sample(1, rng)
        x_true = x_samples[0]
        component_id = int(component_ids[0])
    else:
        x_true = np.asarray(x_true, dtype=float)
        component_id = -1
    y_obs = A @ x_true + noise_std * rng.normal(size=m)
    return {
        "A": A,
        "x_true": x_true,
        "component_id": component_id,
        "y_obs": y_obs,
        "noise_std": float(noise_std),
        "A_type": A_type,
    }


def data_gradient(x: np.ndarray, y_obs: np.ndarray, A: np.ndarray, noise_std: float) -> np.ndarray:
    return A.T @ (A @ x - y_obs) / (noise_std**2)


def exact_linear_posterior(
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
) -> dict:
    """Exact posterior for a Gaussian-mixture prior and linear Gaussian data."""

    y_obs = np.asarray(y_obs, dtype=float)
    m_obs = y_obs.size
    comp_log = np.empty(prior.K, dtype=float)
    comp_means = np.empty((prior.K, prior.d), dtype=float)
    comp_covs = np.empty((prior.K, prior.d, prior.d), dtype=float)

    eye_m = np.eye(m_obs)
    for i in range(prior.K):
        mean_y = A @ prior.means[i]
        sigma = prior.covariances[i]
        AS = A @ sigma
        S_y = AS @ A.T + (noise_std**2) * eye_m
        c, lower = cho_factor(S_y, lower=True, check_finite=False)
        residual = y_obs - mean_y
        solved_residual = cho_solve((c, lower), residual, check_finite=False)
        solved_AS = cho_solve((c, lower), AS, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(c)))

        comp_log[i] = (
            np.log(max(prior.weights[i], 1e-300))
            - 0.5 * (m_obs * LOG2PI + logdet + residual @ solved_residual)
        )
        comp_means[i] = prior.means[i] + sigma @ A.T @ solved_residual
        comp_covs[i] = sigma - sigma @ A.T @ solved_AS
        comp_covs[i] = 0.5 * (comp_covs[i] + comp_covs[i].T)

    log_norm = logsumexp(comp_log)
    weights = np.exp(comp_log - log_norm)
    mean = weights @ comp_means
    cov = np.zeros((prior.d, prior.d), dtype=float)
    for i in range(prior.K):
        centered = comp_means[i] - mean
        cov += weights[i] * (comp_covs[i] + np.outer(centered, centered))

    return {
        "component_weights": weights,
        "component_means": comp_means,
        "component_covariances": comp_covs,
        "mean": mean,
        "covariance": cov,
        "log_evidence": float(log_norm),
    }


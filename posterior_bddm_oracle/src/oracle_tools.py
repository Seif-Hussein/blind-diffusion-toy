"""Shared utilities for posterior-BDDM oracle diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from toy_blind_splitting.src.measurements import data_gradient
from toy_blind_splitting.src.metrics import local_tangent_normal_ratio
from toy_blind_splitting.src.priors import GaussianMixturePrior


EPS = 1e-15


@dataclass(frozen=True)
class SplitSpec:
    """A finite splitting force configuration."""

    name: str
    kind: str
    pdhg_gamma: float = 1.0
    hqs_tau: float = 1.0


def posterior_as_gmm(
    prior: GaussianMixturePrior,
    posterior: dict,
    sigma_grid: Optional[np.ndarray] = None,
) -> GaussianMixturePrior:
    """Represent the exact inverse posterior as a reusable GMM prior object."""

    metadata = dict(prior.metadata)
    metadata["kind"] = f"posterior_{metadata.get('kind', 'gmm')}"
    metadata["base_kind"] = prior.metadata.get("kind", "gmm")
    return GaussianMixturePrior(
        posterior["component_weights"],
        posterior["component_means"],
        posterior["component_covariances"],
        sigma_grid=prior.sigma_grid if sigma_grid is None else sigma_grid,
        metadata=metadata,
    )


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.maximum(denominator, EPS)


def relative_error(estimate: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(estimate - target, axis=-1) / np.maximum(
        np.linalg.norm(target, axis=-1),
        EPS,
    )


def component_kl(estimated: np.ndarray, target: np.ndarray) -> float:
    estimated = np.asarray(estimated, dtype=float)
    target = np.asarray(target, dtype=float)
    estimated = np.maximum(estimated, EPS)
    target = np.maximum(target, EPS)
    estimated = estimated / np.sum(estimated)
    target = target / np.sum(target)
    return float(np.sum(estimated * (np.log(estimated) - np.log(target))))


def component_weight_estimate(
    posterior_prior: GaussianMixturePrior,
    samples: np.ndarray,
    sigma: float = 1e-6,
) -> np.ndarray:
    responsibilities = posterior_prior.posterior_component_weights(samples, sigma)
    if responsibilities.ndim == 1:
        return responsibilities
    return np.mean(responsibilities, axis=0)


def covariance_error(samples: np.ndarray, target_covariance: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    if samples.shape[0] <= 1:
        return float("nan")
    sample_cov = np.cov(samples, rowvar=False, bias=False)
    return float(
        np.linalg.norm(sample_cov - target_covariance, ord="fro")
        / max(np.linalg.norm(target_covariance, ord="fro"), EPS)
    )


def posterior_mean_error(samples: np.ndarray, target_mean: np.ndarray) -> float:
    sample_mean = np.mean(np.asarray(samples, dtype=float), axis=0)
    return float(np.mean((sample_mean - target_mean) ** 2))


def batch_data_gradient(
    x: np.ndarray,
    y_obs: np.ndarray,
    A: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return data_gradient(x, y_obs, A, noise_std)
    residuals = x @ A.T - y_obs[None, :]
    return residuals @ A / (noise_std**2)


def pdhg_force(
    x: np.ndarray,
    y_obs: np.ndarray,
    A: np.ndarray,
    noise_std: float,
    gamma: float,
    w: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A.T @ w_next`` for the quadratic-likelihood PDHG split."""

    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    x2 = x[None, :] if single else x
    m_obs = A.shape[0]
    if w is None:
        w2 = np.zeros((x2.shape[0], m_obs), dtype=float)
    else:
        w_arr = np.asarray(w, dtype=float)
        w2 = w_arr[None, :] if w_arr.ndim == 1 else w_arr
    ax = x2 @ A.T
    v = ax + w2 / float(gamma)
    z = (float(gamma) * noise_std**2 * v + y_obs[None, :]) / (
        float(gamma) * noise_std**2 + 1.0
    )
    w_next = w2 + float(gamma) * (ax - z)
    force = w_next @ A
    return (force[0], w_next[0]) if single else (force, w_next)


def hqs_force(
    x: np.ndarray,
    y_obs: np.ndarray,
    A: np.ndarray,
    noise_std: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the clean-space HQS/prox-gradient force and prox point.

    The prox solves ``min_z 0.5||z-x||^2 + tau h(Az;y)`` for
    ``h(u;y)=||u-y||^2/(2 noise_std^2)``. The returned force is
    ``(x - prox) / tau``, which tends to the likelihood gradient as
    ``tau -> 0``.
    """

    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    x2 = x[None, :] if single else x
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("hqs tau must be positive")
    residuals = x2 @ A.T - y_obs[None, :]
    system = A @ A.T + (noise_std**2 / tau) * np.eye(A.shape[0])
    q = np.linalg.solve(system, residuals.T).T
    step = q @ A
    prox = x2 - step
    force = step / tau
    return (force[0], prox[0]) if single else (force, prox)


def split_force(
    spec: SplitSpec,
    x: np.ndarray,
    y_obs: np.ndarray,
    A: np.ndarray,
    noise_std: float,
    w: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if spec.kind == "gradient":
        return batch_data_gradient(x, y_obs, A, noise_std), w
    if spec.kind == "pdhg":
        force, w_next = pdhg_force(x, y_obs, A, noise_std, spec.pdhg_gamma, w)
        return force, w_next
    if spec.kind == "hqs":
        force, _prox = hqs_force(x, y_obs, A, noise_std, spec.hqs_tau)
        return force, w
    raise ValueError(f"unknown split kind: {spec.kind}")


def make_split_specs(
    methods: list[str],
    pdhg_gammas: list[float],
    hqs_taus: list[float],
) -> list[SplitSpec]:
    specs: list[SplitSpec] = []
    for method in methods:
        if method == "gradient":
            specs.append(SplitSpec(name="gradient", kind="gradient"))
        elif method == "pdhg":
            for gamma in pdhg_gammas:
                specs.append(
                    SplitSpec(
                        name=f"pdhg_gamma_{gamma:g}",
                        kind="pdhg",
                        pdhg_gamma=float(gamma),
                    )
                )
        elif method == "hqs":
            for tau in hqs_taus:
                specs.append(
                    SplitSpec(
                        name=f"hqs_tau_{tau:g}",
                        kind="hqs",
                        hqs_tau=float(tau),
                    )
                )
        else:
            raise ValueError(f"unknown split method: {method}")
    return specs


def tangent_normal_fraction(
    prior: GaussianMixturePrior,
    points: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    single = points.ndim == 1
    p2 = points[None, :] if single else points
    v2 = vectors[None, :] if vectors.ndim == 1 else vectors
    out = np.asarray(
        [local_tangent_normal_ratio(prior, point, vector) for point, vector in zip(p2, v2)],
        dtype=float,
    )
    return out[0] if single else out


def sigma_diagnostics(
    prior: GaussianMixturePrior,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=float)
    sigmas = np.asarray(prior.sigma_mle(states), dtype=float)
    if sigmas.ndim == 0:
        sigmas = sigmas[None]
    post = prior.sigma_posterior(states)
    weights = post.weights[None, :] if post.weights.ndim == 1 else post.weights
    entropy = -np.sum(weights * np.log(np.maximum(weights, EPS)), axis=1)
    boundary = np.column_stack(
        [
            np.isclose(sigmas, prior.sigma_grid[0]).astype(float),
            np.isclose(sigmas, prior.sigma_grid[-1]).astype(float),
        ]
    )
    return sigmas, entropy, boundary


"""Closed-form Gaussian-mixture priors and blind denoisers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp


LOG2PI = np.log(2.0 * np.pi)


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    return arr, single


def _rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _orthonormal_matrix(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q[:, :k]


@dataclass
class SigmaPosterior:
    """Posterior over the blind diffusion scale grid."""

    sigma_grid: np.ndarray
    weights: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return self.weights @ self.sigma_grid

    @property
    def second_moment(self) -> np.ndarray:
        return self.weights @ (self.sigma_grid**2)


class GaussianMixturePrior:
    """Gaussian-mixture prior with analytical noisy denoisers.

    Parameters
    ----------
    weights:
        Mixture weights, shape ``(K,)``.
    means:
        Component means, shape ``(K, d)``.
    covariances:
        Component covariance matrices, shape ``(K, d, d)``.

    Notes
    -----
    The implementation stores a dense covariance and an eigendecomposition for
    each component. This keeps formulas transparent and is adequate for the toy
    sizes used by the default scripts. Larger sweeps are possible but can become
    expensive for high ``d`` and large ``K``.
    """

    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        sigma_grid: Optional[np.ndarray] = None,
        sigma_prior: str | Callable[[np.ndarray], np.ndarray] = "log_uniform",
        metadata: Optional[dict] = None,
        eig_floor: float = 1e-12,
    ):
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / np.sum(self.weights)
        self.log_weights = np.log(np.maximum(self.weights, 1e-300))

        self.means = np.asarray(means, dtype=float)
        self.covariances = np.asarray(covariances, dtype=float)
        if self.means.ndim != 2:
            raise ValueError("means must have shape (K, d)")
        if self.covariances.ndim != 3:
            raise ValueError("covariances must have shape (K, d, d)")
        if self.covariances.shape[:2] != (self.means.shape[0], self.means.shape[1]):
            raise ValueError("covariances must have shape (K, d, d)")
        if self.covariances.shape[2] != self.means.shape[1]:
            raise ValueError("covariances must have shape (K, d, d)")
        if self.weights.shape != (self.means.shape[0],):
            raise ValueError("weights must have shape (K,)")

        self.K, self.d = self.means.shape
        self.metadata = {} if metadata is None else dict(metadata)
        self.sigma_prior = sigma_prior
        self.sigma_grid = (
            np.geomspace(1e-3, 3.0, 81) if sigma_grid is None else np.asarray(sigma_grid, dtype=float)
        )

        eigvals = []
        eigvecs = []
        for cov in self.covariances:
            sym_cov = 0.5 * (cov + cov.T)
            vals, vecs = eigh(sym_cov)
            eigvals.append(np.maximum(vals, eig_floor))
            eigvecs.append(vecs)
        self.eigvals = np.asarray(eigvals)
        self.eigvecs = np.asarray(eigvecs)
        self._lambda_cache: dict[float, list[np.ndarray]] = {}
        self._scalar_sigma_cache: dict[float, dict[str, np.ndarray]] = {}
        self._sigma_grid_cache: Optional[dict[str, np.ndarray]] = None
        self._sigma_grid_cache_key: Optional[tuple[bytes, str]] = None

    @property
    def mean(self) -> np.ndarray:
        return self.weights @ self.means

    def with_sigma_grid(
        self,
        sigma_min: float,
        sigma_max: float,
        n_grid: int,
        sigma_prior: str | Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "GaussianMixturePrior":
        self.sigma_grid = np.geomspace(float(sigma_min), float(sigma_max), int(n_grid))
        self._sigma_grid_cache = None
        self._sigma_grid_cache_key = None
        if sigma_prior is not None:
            self.sigma_prior = sigma_prior
        return self

    def precompute_sigma_grid(self) -> None:
        """Precompute all cached eigenspace quantities for ``self.sigma_grid``."""

        self._ensure_sigma_grid_cache()

    def precompute_sigmas(self, sigmas: np.ndarray) -> None:
        """Precompute scalar denoiser/covariance-action quantities for sigmas."""

        for sigma in np.asarray(sigmas, dtype=float):
            self._scalar_cache(float(sigma))

    def _scalar_cache(self, sigma: float) -> dict[str, np.ndarray]:
        key = float(np.round(float(sigma), 14))
        cached = self._scalar_sigma_cache.get(key)
        if cached is not None:
            return cached
        sigma2 = key**2
        denom = self.eigvals + sigma2
        cached = {
            "sigma": np.asarray(key),
            "inv_denom": 1.0 / denom,
            "logdet": np.sum(np.log(denom), axis=1),
            "gain": self.eigvals / denom,
            "lambda": self.eigvals * sigma2 / denom,
        }
        self._scalar_sigma_cache[key] = cached
        return cached

    def _ensure_sigma_grid_cache(self) -> dict[str, np.ndarray]:
        grid = np.asarray(self.sigma_grid, dtype=float)
        key = (grid.tobytes(), str(self.sigma_prior))
        if self._sigma_grid_cache is not None and self._sigma_grid_cache_key == key:
            return self._sigma_grid_cache
        sigma2 = grid[:, None, None] ** 2
        eigvals = self.eigvals[None, :, :]
        denom = eigvals + sigma2
        self._sigma_grid_cache = {
            "sigma_grid": grid.copy(),
            "log_prior": self._sigma_log_prior(grid),
            "inv_denom": 1.0 / denom,
            "logdet": np.sum(np.log(denom), axis=2),
            "gain": eigvals / denom,
            "lambda": eigvals * sigma2 / denom,
        }
        self._sigma_grid_cache_key = key
        return self._sigma_grid_cache

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray]:
        rng = _rng(rng)
        ids = rng.choice(self.K, size=n, p=self.weights)
        x = np.empty((n, self.d), dtype=float)
        for i in range(self.K):
            mask = ids == i
            count = int(np.sum(mask))
            if count == 0:
                continue
            z = rng.normal(size=(count, self.d))
            vals = np.sqrt(self.eigvals[i])
            x[mask] = self.means[i] + (z * vals) @ self.eigvecs[i].T
        return x, ids

    def _sigma_log_prior(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=float)
        if callable(self.sigma_prior):
            return np.asarray(self.sigma_prior(sigma), dtype=float)
        if self.sigma_prior == "uniform":
            return np.zeros_like(sigma, dtype=float)
        if self.sigma_prior == "log_uniform":
            return -np.log(np.maximum(sigma, 1e-300))
        raise ValueError(f"unknown sigma prior: {self.sigma_prior}")

    def _log_component_terms(self, y: np.ndarray, sigma: float) -> np.ndarray:
        y, _ = _as_2d(y)
        cache = self._scalar_cache(float(sigma))
        out = np.empty((y.shape[0], self.K), dtype=float)
        for i in range(self.K):
            diff = y - self.means[i]
            proj = diff @ self.eigvecs[i]
            mahal = np.sum((proj**2) * cache["inv_denom"][i], axis=1)
            out[:, i] = self.log_weights[i] - 0.5 * (
                self.d * LOG2PI + cache["logdet"][i] + mahal
            )
        return out

    def _sigma_grid_scores(self, y: np.ndarray) -> np.ndarray:
        y, _ = _as_2d(y)
        cache = self._ensure_sigma_grid_cache()
        scores = np.full((y.shape[0], cache["sigma_grid"].size), -np.inf, dtype=float)
        const = self.d * LOG2PI
        for i in range(self.K):
            diff = y - self.means[i]
            proj2 = (diff @ self.eigvecs[i]) ** 2
            mahal = proj2 @ cache["inv_denom"][:, i, :].T
            terms = self.log_weights[i] - 0.5 * (
                const + cache["logdet"][:, i][None, :] + mahal
            )
            scores = np.logaddexp(scores, terms)
        scores += cache["log_prior"][None, :]
        return scores

    def log_p_sigma(self, y: np.ndarray, sigma: float) -> np.ndarray | float:
        y, single = _as_2d(y)
        val = logsumexp(self._log_component_terms(y, sigma), axis=1)
        return float(val[0]) if single else val

    def log_prior_density(self, x: np.ndarray, covariance_floor: float = 0.0) -> np.ndarray | float:
        sigma = float(np.sqrt(max(covariance_floor, 0.0)))
        return self.log_p_sigma(x, sigma)

    def posterior_component_weights(self, y: np.ndarray, sigma: float) -> np.ndarray:
        y, single = _as_2d(y)
        terms = self._log_component_terms(y, sigma)
        log_norm = logsumexp(terms, axis=1, keepdims=True)
        weights = np.exp(terms - log_norm)
        return weights[0] if single else weights

    def component_posterior_means(self, y: np.ndarray, sigma: float) -> np.ndarray:
        y, single = _as_2d(y)
        cache = self._scalar_cache(float(sigma))
        mus = np.empty((y.shape[0], self.K, self.d), dtype=float)
        for i in range(self.K):
            diff = y - self.means[i]
            proj = diff @ self.eigvecs[i]
            mus[:, i, :] = self.means[i] + (proj * cache["gain"][i]) @ self.eigvecs[i].T
        return mus[0] if single else mus

    def denoise(self, y: np.ndarray, sigma: float) -> np.ndarray:
        y, single = _as_2d(y)
        weights = self.posterior_component_weights(y, sigma)
        mus = self.component_posterior_means(y, sigma)
        xhat = np.einsum("nk,nkd->nd", weights, mus)
        return xhat[0] if single else xhat

    def _component_posterior_covariances(self, sigma: float) -> list[np.ndarray]:
        key = float(np.round(float(sigma), 14))
        if key in self._lambda_cache:
            return self._lambda_cache[key]
        cache = self._scalar_cache(float(sigma))
        covs = []
        for i in range(self.K):
            vecs = self.eigvecs[i]
            covs.append((vecs * cache["lambda"][i]) @ vecs.T)
        self._lambda_cache[key] = covs
        return covs

    def posterior_covariance(self, y: np.ndarray, sigma: float) -> np.ndarray:
        y, single = _as_2d(y)
        weights = self.posterior_component_weights(y, sigma)
        mus = self.component_posterior_means(y, sigma)
        xhat = np.einsum("nk,nkd->nd", weights, mus)
        lambdas = self._component_posterior_covariances(sigma)
        cov = np.zeros((y.shape[0], self.d, self.d), dtype=float)
        for n in range(y.shape[0]):
            for i in range(self.K):
                centered = mus[n, i] - xhat[n]
                cov[n] += weights[n, i] * (lambdas[i] + np.outer(centered, centered))
        return cov[0] if single else cov

    def posterior_covariance_action(
        self,
        y: np.ndarray,
        sigma: float,
        vector: np.ndarray,
    ) -> np.ndarray:
        """Compute ``C_sigma(y) @ vector`` without forming ``C_sigma(y)``."""

        y, single = _as_2d(y)
        vector = np.asarray(vector, dtype=float)
        if vector.ndim == 1:
            vector = vector[None, :]
        if vector.shape != y.shape:
            raise ValueError("vector must have the same batch shape as y")

        cache = self._scalar_cache(float(sigma))
        weights = self.posterior_component_weights(y, sigma)
        mus = self.component_posterior_means(y, sigma)
        xhat = np.einsum("nk,nkd->nd", weights, mus)
        out = np.zeros_like(y)
        for i in range(self.K):
            vecs = self.eigvecs[i]
            lambda_v = (vector @ vecs * cache["lambda"][i]) @ vecs.T
            centered = mus[:, i, :] - xhat
            rank_v = centered * np.sum(centered * vector, axis=1, keepdims=True)
            out += weights[:, i:i + 1] * (lambda_v + rank_v)
        return out[0] if single else out

    def sigma_mle(self, y: np.ndarray) -> np.ndarray | float:
        y, single = _as_2d(y)
        scores = self._sigma_grid_scores(y)
        idx = np.argmax(scores, axis=1)
        sigmas = self.sigma_grid[idx]
        return float(sigmas[0]) if single else sigmas

    def sigma_posterior(self, y: np.ndarray) -> SigmaPosterior:
        y, single = _as_2d(y)
        scores = self._sigma_grid_scores(y)
        scores -= logsumexp(scores, axis=1, keepdims=True)
        weights = np.exp(scores)
        if single:
            weights = weights[0]
        return SigmaPosterior(self.sigma_grid.copy(), weights)

    def blind_denoise_mle(self, y: np.ndarray) -> np.ndarray:
        y, single = _as_2d(y)
        sigmas = np.asarray(self.sigma_mle(y), dtype=float)
        if sigmas.ndim == 0:
            sigmas = sigmas[None]
        out = np.empty_like(y)
        for n, sigma in enumerate(sigmas):
            out[n] = self.denoise(y[n], float(sigma))
        return out[0] if single else out

    def blind_denoise_bayes(self, y: np.ndarray) -> np.ndarray:
        y, single = _as_2d(y)
        post = self.sigma_posterior(y)
        weights = post.weights[None, :] if post.weights.ndim == 1 else post.weights
        out = np.zeros_like(y)
        for j, sigma in enumerate(post.sigma_grid):
            out += weights[:, j:j + 1] * self.denoise(y, float(sigma))
        return out[0] if single else out


def make_subspace_gmm(
    d: int,
    intrinsic_k: int = 2,
    n_components: int = 8,
    radius: float = 2.0,
    tangent_std: float = 0.25,
    normal_std: float = 0.08,
    ambient_jitter: float = 0.03,
    rng: Optional[np.random.Generator] = None,
    sigma_grid: Optional[np.ndarray] = None,
) -> GaussianMixturePrior:
    rng = _rng(rng)
    if intrinsic_k != 2:
        latent = rng.normal(size=(n_components, intrinsic_k))
        latent = radius * latent / np.maximum(np.linalg.norm(latent, axis=1, keepdims=True), 1e-12)
    else:
        angles = np.linspace(0.0, 2.0 * np.pi, n_components, endpoint=False)
        latent = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    u = _orthonormal_matrix(d, intrinsic_k, rng)
    means = latent @ u.T

    covs = np.empty((n_components, d, d), dtype=float)
    eye = np.eye(d)
    for i in range(n_components):
        q, _ = np.linalg.qr(rng.normal(size=(intrinsic_k, intrinsic_k)))
        latent_vars = np.linspace(tangent_std**2, normal_std**2, intrinsic_k)
        s = q @ np.diag(latent_vars) @ q.T
        covs[i] = u @ s @ u.T + (ambient_jitter**2) * eye
    metadata = {
        "kind": "subspace",
        "intrinsic_k": intrinsic_k,
        "U": u,
        "latent_means": latent,
    }
    weights = np.ones(n_components) / n_components
    return GaussianMixturePrior(weights, means, covs, sigma_grid=sigma_grid, metadata=metadata)


def make_ellipse_gmm(
    d: int,
    n_components: int = 64,
    r1: float = 2.0,
    r2: float = 0.9,
    tangent_std: float = 0.22,
    normal_std: float = 0.035,
    ambient_jitter: float = 0.02,
    rng: Optional[np.random.Generator] = None,
    sigma_grid: Optional[np.ndarray] = None,
) -> GaussianMixturePrior:
    rng = _rng(rng)
    u = _orthonormal_matrix(d, 2, rng)
    angles = np.linspace(0.0, 2.0 * np.pi, n_components, endpoint=False)
    latent = np.column_stack([r1 * np.cos(angles), r2 * np.sin(angles)])
    means = latent @ u.T
    covs = np.empty((n_components, d, d), dtype=float)
    tangents = np.empty((n_components, d), dtype=float)
    normals = np.empty((n_components, d), dtype=float)
    eye = np.eye(d)
    for i, theta in enumerate(angles):
        t_lat = np.array([-r1 * np.sin(theta), r2 * np.cos(theta)])
        n_lat = np.array([np.cos(theta) / r1, np.sin(theta) / r2])
        t_lat = t_lat / np.linalg.norm(t_lat)
        n_lat = n_lat / np.linalg.norm(n_lat)
        t = u @ t_lat
        n = u @ n_lat
        tangents[i] = t / np.linalg.norm(t)
        normals[i] = n / np.linalg.norm(n)
        covs[i] = (
            tangent_std**2 * np.outer(tangents[i], tangents[i])
            + normal_std**2 * np.outer(normals[i], normals[i])
            + ambient_jitter**2 * eye
        )
    metadata = {
        "kind": "ellipse",
        "intrinsic_k": 2,
        "U": u,
        "latent_means": latent,
        "angles": angles,
        "r1": r1,
        "r2": r2,
        "ambient_tangents": tangents,
        "ambient_normals": normals,
    }
    weights = np.ones(n_components) / n_components
    return GaussianMixturePrior(weights, means, covs, sigma_grid=sigma_grid, metadata=metadata)


def make_full_gaussian_control(
    d: int,
    variance: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    sigma_grid: Optional[np.ndarray] = None,
) -> GaussianMixturePrior:
    _ = _rng(rng)
    weights = np.ones(1)
    means = np.zeros((1, d), dtype=float)
    covs = variance * np.eye(d, dtype=float)[None, :, :]
    metadata = {"kind": "full_gaussian", "intrinsic_k": d}
    return GaussianMixturePrior(weights, means, covs, sigma_grid=sigma_grid, metadata=metadata)

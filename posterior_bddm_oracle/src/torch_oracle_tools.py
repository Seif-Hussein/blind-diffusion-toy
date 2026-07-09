"""Torch/CUDA utilities for posterior-BDDM oracle correction tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


LOG2PI = math.log(2.0 * math.pi)
EPS = 1e-30


def make_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


@dataclass
class TorchGMM:
    weights: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    sigma_grid: torch.Tensor
    metadata: dict

    def __post_init__(self) -> None:
        self.weights = self.weights / torch.sum(self.weights)
        self.log_weights = torch.log(self.weights.clamp_min(EPS))
        self.K, self.d = self.means.shape
        cov = 0.5 * (self.covariances + self.covariances.transpose(-1, -2))
        vals, vecs = torch.linalg.eigh(cov)
        self.eigvals = vals.clamp_min(1e-12)
        self.eigvecs = vecs
        self._mean = self.weights @ self.means

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.means.dtype

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    def sample(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.multinomial(self.weights, int(n), replacement=True, generator=gen)
        z = torch.randn((int(n), self.d), device=self.device, dtype=self.dtype, generator=gen)
        vals = torch.sqrt(self.eigvals[ids])
        vecs = self.eigvecs[ids]
        x = self.means[ids] + torch.bmm((z * vals).unsqueeze(1), vecs.transpose(1, 2)).squeeze(1)
        return x, ids

    def _component_log_terms(self, y: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y[None, :]
        sigma_t = torch.as_tensor(sigma, device=self.device, dtype=self.dtype)
        denom = self.eigvals + sigma_t.square()
        logdet = torch.sum(torch.log(denom), dim=1)
        terms = []
        for i in range(self.K):
            diff = y - self.means[i]
            proj = diff @ self.eigvecs[i]
            mahal = torch.sum(proj.square() / denom[i], dim=1)
            terms.append(self.log_weights[i] - 0.5 * (self.d * LOG2PI + logdet[i] + mahal))
        return torch.stack(terms, dim=1)

    def sigma_grid_scores(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y[None, :]
        grid = self.sigma_grid
        scores = torch.full((y.shape[0], grid.numel()), -torch.inf, device=self.device, dtype=self.dtype)
        denom = self.eigvals[None, :, :] + grid[:, None, None].square()
        inv_denom = 1.0 / denom
        logdet = torch.sum(torch.log(denom), dim=2)
        for i in range(self.K):
            diff = y - self.means[i]
            proj2 = (diff @ self.eigvecs[i]).square()
            mahal = proj2 @ inv_denom[:, i, :].T
            terms = self.log_weights[i] - 0.5 * (
                self.d * LOG2PI + logdet[:, i][None, :] + mahal
            )
            scores = torch.logaddexp(scores, terms)
        scores = scores - torch.log(grid.clamp_min(EPS))[None, :]
        return scores

    def sigma_mle(self, y: torch.Tensor) -> torch.Tensor:
        idx = torch.argmax(self.sigma_grid_scores(y), dim=1)
        return self.sigma_grid[idx]

    def sigma_entropy(self, y: torch.Tensor) -> torch.Tensor:
        scores = self.sigma_grid_scores(y)
        log_probs = scores - torch.logsumexp(scores, dim=1, keepdim=True)
        probs = torch.exp(log_probs)
        return -torch.sum(probs * log_probs, dim=1)

    def posterior_component_weights(self, y: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._component_log_terms(y, sigma), dim=1)

    def component_posterior_means(self, y: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y[None, :]
        sigma_t = torch.as_tensor(sigma, device=self.device, dtype=self.dtype)
        gain = self.eigvals / (self.eigvals + sigma_t.square())
        mus = []
        for i in range(self.K):
            diff = y - self.means[i]
            proj = diff @ self.eigvecs[i]
            mus.append(self.means[i] + (proj * gain[i]) @ self.eigvecs[i].T)
        return torch.stack(mus, dim=1)

    def denoise(self, y: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        single = y.ndim == 1
        y2 = y[None, :] if single else y
        weights = self.posterior_component_weights(y2, sigma)
        mus = self.component_posterior_means(y2, sigma)
        out = torch.einsum("nk,nkd->nd", weights, mus)
        return out[0] if single else out


def make_ellipse_gmm_torch(
    d: int,
    n_components: int,
    sigma_grid: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    r1: float = 2.0,
    r2: float = 0.9,
    tangent_std: float = 0.22,
    normal_std: float = 0.035,
    ambient_jitter: float = 0.02,
) -> TorchGMM:
    gen = make_generator(device, seed)
    q, _ = torch.linalg.qr(torch.randn((d, 2), device=device, dtype=dtype, generator=gen))
    U = q[:, :2]
    angles = torch.linspace(0.0, 2.0 * math.pi, n_components + 1, device=device, dtype=dtype)[:-1]
    latent = torch.stack([r1 * torch.cos(angles), r2 * torch.sin(angles)], dim=1)
    means = latent @ U.T

    eye = torch.eye(d, device=device, dtype=dtype)
    covs = torch.empty((n_components, d, d), device=device, dtype=dtype)
    tangents = torch.empty((n_components, d), device=device, dtype=dtype)
    normals = torch.empty((n_components, d), device=device, dtype=dtype)
    for i, theta in enumerate(angles):
        t_lat = torch.stack([-r1 * torch.sin(theta), r2 * torch.cos(theta)])
        n_lat = torch.stack([torch.cos(theta) / r1, torch.sin(theta) / r2])
        t_lat = t_lat / torch.linalg.norm(t_lat).clamp_min(EPS)
        n_lat = n_lat / torch.linalg.norm(n_lat).clamp_min(EPS)
        tangents[i] = U @ t_lat
        normals[i] = U @ n_lat
        covs[i] = (
            tangent_std**2 * torch.outer(tangents[i], tangents[i])
            + normal_std**2 * torch.outer(normals[i], normals[i])
            + ambient_jitter**2 * eye
        )
    weights = torch.full((n_components,), 1.0 / n_components, device=device, dtype=dtype)
    return TorchGMM(
        weights=weights,
        means=means,
        covariances=covs,
        sigma_grid=sigma_grid,
        metadata={
            "kind": "ellipse",
            "intrinsic_k": 2,
            "U": U,
            "latent_means": latent,
            "tangents": tangents,
            "normals": normals,
        },
    )


def make_full_gaussian_torch(
    d: int,
    variance: float,
    sigma_grid: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> TorchGMM:
    weights = torch.ones((1,), device=device, dtype=dtype)
    means = torch.zeros((1, d), device=device, dtype=dtype)
    covs = variance * torch.eye(d, device=device, dtype=dtype)[None, :, :]
    return TorchGMM(weights, means, covs, sigma_grid, {"kind": "full", "intrinsic_k": d})


def make_operator(d: int, m: int, kind: str, gen: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kind == "random":
        return torch.randn((m, d), device=device, dtype=dtype, generator=gen) / math.sqrt(m)
    if kind == "mask":
        if m > d:
            raise ValueError("mask measurement count cannot exceed d")
        idx = torch.randperm(d, device=device, generator=gen)[:m]
        A = torch.zeros((m, d), device=device, dtype=dtype)
        A[torch.arange(m, device=device), idx] = 1.0
        return A
    raise ValueError(f"unknown operator kind: {kind}")


def make_measurement(
    prior: TorchGMM,
    A: torch.Tensor,
    noise_std: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, _ids = prior.sample(1, gen)
    y_obs = A @ x[0] + float(noise_std) * torch.randn(
        (A.shape[0],),
        device=prior.device,
        dtype=prior.dtype,
        generator=gen,
    )
    return x[0], y_obs


def exact_linear_posterior_gmm(
    prior: TorchGMM,
    A: torch.Tensor,
    y_obs: torch.Tensor,
    noise_std: float,
) -> tuple[TorchGMM, dict[str, torch.Tensor]]:
    m_obs = A.shape[0]
    eye_m = torch.eye(m_obs, device=prior.device, dtype=prior.dtype)
    comp_log = torch.empty((prior.K,), device=prior.device, dtype=prior.dtype)
    comp_means = torch.empty_like(prior.means)
    comp_covs = torch.empty_like(prior.covariances)

    for i in range(prior.K):
        cov = prior.covariances[i]
        mean_y = A @ prior.means[i]
        AS = A @ cov
        S_y = AS @ A.T + float(noise_std) ** 2 * eye_m
        S_y = 0.5 * (S_y + S_y.T)
        chol = torch.linalg.cholesky(S_y)
        residual = y_obs - mean_y
        solved_residual = torch.cholesky_solve(residual[:, None], chol).squeeze(1)
        solved_AS = torch.cholesky_solve(AS, chol)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(chol)))
        comp_log[i] = prior.log_weights[i] - 0.5 * (
            m_obs * LOG2PI + logdet + residual @ solved_residual
        )
        comp_means[i] = prior.means[i] + AS.T @ solved_residual
        post_cov = cov - AS.T @ solved_AS
        comp_covs[i] = 0.5 * (post_cov + post_cov.T)

    weights = torch.softmax(comp_log, dim=0)
    mean = weights @ comp_means
    cov = torch.zeros((prior.d, prior.d), device=prior.device, dtype=prior.dtype)
    for i in range(prior.K):
        centered = comp_means[i] - mean
        cov = cov + weights[i] * (comp_covs[i] + torch.outer(centered, centered))

    posterior = TorchGMM(
        weights=weights,
        means=comp_means,
        covariances=comp_covs,
        sigma_grid=prior.sigma_grid,
        metadata={**prior.metadata, "kind": f"posterior_{prior.metadata.get('kind', 'gmm')}"},
    )
    return posterior, {
        "component_weights": weights,
        "component_means": comp_means,
        "component_covariances": comp_covs,
        "mean": mean,
        "covariance": cov,
    }


def data_gradient(x: torch.Tensor, y_obs: torch.Tensor, A: torch.Tensor, noise_std: float) -> torch.Tensor:
    residual = x @ A.T - y_obs[None, :]
    return residual @ A / (float(noise_std) ** 2)


def pdhg_force(
    x: torch.Tensor,
    y_obs: torch.Tensor,
    A: torch.Tensor,
    noise_std: float,
    gamma: float,
    w: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ax = x @ A.T
    if w is None:
        w = torch.zeros_like(ax)
    v = ax + w / float(gamma)
    z = (float(gamma) * float(noise_std) ** 2 * v + y_obs[None, :]) / (
        float(gamma) * float(noise_std) ** 2 + 1.0
    )
    w_next = w + float(gamma) * (ax - z)
    return w_next @ A, w_next


def hqs_force(
    x: torch.Tensor,
    y_obs: torch.Tensor,
    A: torch.Tensor,
    noise_std: float,
    tau: float,
) -> torch.Tensor:
    residual = x @ A.T - y_obs[None, :]
    system = A @ A.T + (float(noise_std) ** 2 / float(tau)) * torch.eye(
        A.shape[0],
        device=A.device,
        dtype=A.dtype,
    )
    q = torch.linalg.solve(system, residual.T).T
    return (q @ A) / float(tau)


def split_force(
    method: str,
    x: torch.Tensor,
    y_obs: torch.Tensor,
    A: torch.Tensor,
    noise_std: float,
    pdhg_gamma: float,
    hqs_tau: float,
) -> torch.Tensor:
    if method == "gradient":
        return data_gradient(x, y_obs, A, noise_std)
    if method == "pdhg":
        force, _w = pdhg_force(x, y_obs, A, noise_std, pdhg_gamma)
        return force
    if method == "hqs":
        return hqs_force(x, y_obs, A, noise_std, hqs_tau)
    raise ValueError(f"unknown split method: {method}")


def relative_error(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(estimate - target, dim=1) / torch.linalg.norm(target, dim=1).clamp_min(1e-15)


def cosine_similarity(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(estimate * target, dim=1)
    denom = torch.linalg.norm(estimate, dim=1) * torch.linalg.norm(target, dim=1)
    return numerator / denom.clamp_min(1e-15)


def tangent_normal_fraction(prior: TorchGMM, points: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    tangents = prior.metadata.get("tangents")
    if tangents is None:
        return torch.full((points.shape[0],), torch.nan, device=prior.device, dtype=prior.dtype)
    distances = torch.cdist(points, prior.means)
    idx = torch.argmin(distances, dim=1)
    tangent = tangents[idx]
    tangent = tangent / torch.linalg.norm(tangent, dim=1, keepdim=True).clamp_min(1e-15)
    projection = torch.sum(vectors * tangent, dim=1, keepdim=True) * tangent
    normal = vectors - projection
    return torch.linalg.norm(normal, dim=1) / torch.linalg.norm(vectors, dim=1).clamp_min(1e-15)


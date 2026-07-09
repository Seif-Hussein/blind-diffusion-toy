"""CUDA GMM prior factories.

The implementation reuses the already validated dense-eigenspace Torch GMM
tools in `posterior_bddm_oracle`. This module provides the requested package API
and the extra low-dimensional two-component prior factory.
"""

from __future__ import annotations

import torch
import numpy as np

from posterior_bddm_oracle.src.torch_oracle_tools import (
    TorchGMM as GaussianMixturePrior,
    make_ellipse_gmm_torch,
    make_full_gaussian_torch,
    make_generator,
)


def make_sigma_grid(cfg: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma_min = float(cfg.get("min", 0.01))
    sigma_max = float(cfg.get("max", 3.0))
    n = int(cfg.get("n", 128))
    if cfg.get("spacing", "log") == "linear":
        return torch.linspace(sigma_min, sigma_max, n, device=device, dtype=dtype)
    return torch.tensor(np.geomspace(sigma_min, sigma_max, n), device=device, dtype=dtype)


def make_lowdim_two_component_gmm(
    d: int,
    sigma_grid: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    intrinsic_dim: int = 2,
    separation: float = 1.5,
    diag_noise: float = 1e-4,
    latent_std: float = 0.15,
) -> GaussianMixturePrior:
    gen = make_generator(device, seed)
    q, _ = torch.linalg.qr(torch.randn((d, intrinsic_dim), device=device, dtype=dtype, generator=gen))
    U = q[:, :intrinsic_dim]
    latent = torch.zeros((2, intrinsic_dim), device=device, dtype=dtype)
    latent[0, 0] = -float(separation)
    latent[1, 0] = float(separation)
    means = latent @ U.T
    eye = torch.eye(d, device=device, dtype=dtype)
    latent_cov = (float(latent_std) ** 2) * (U @ U.T)
    covs = latent_cov[None, :, :].repeat(2, 1, 1) + float(diag_noise) ** 2 * eye[None, :, :]
    weights = torch.full((2,), 0.5, device=device, dtype=dtype)
    return GaussianMixturePrior(
        weights=weights,
        means=means,
        covariances=covs,
        sigma_grid=sigma_grid,
        metadata={"kind": "lowdim_two_component", "intrinsic_k": intrinsic_dim, "U": U},
    )


def make_prior(cfg: dict, sigma_grid: torch.Tensor, device: torch.device, dtype: torch.dtype, seed: int):
    kind = cfg.get("type", "ellipse")
    d = int(cfg.get("ambient_dim", 50))
    if kind == "ellipse":
        return make_ellipse_gmm_torch(
            d=d,
            n_components=int(cfg.get("n_components", 64)),
            sigma_grid=sigma_grid,
            device=device,
            dtype=dtype,
            seed=seed,
            tangent_std=float(cfg.get("tangent_std", 0.10)),
            normal_std=float(cfg.get("normal_std", 0.02)),
            ambient_jitter=float(cfg.get("diag_noise", 1e-4)),
        )
    if kind == "lowdim_gmm":
        return make_lowdim_two_component_gmm(
            d=d,
            sigma_grid=sigma_grid,
            device=device,
            dtype=dtype,
            seed=seed,
            intrinsic_dim=int(cfg.get("intrinsic_dim", 2)),
            separation=float(cfg.get("separation", 1.5)),
            diag_noise=float(cfg.get("diag_noise", 1e-4)),
            latent_std=float(cfg.get("latent_std", 0.15)),
        )
    if kind == "full":
        return make_full_gaussian_torch(
            d=d,
            variance=float(cfg.get("variance", 1.0)),
            sigma_grid=sigma_grid,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"unknown prior type: {kind}")

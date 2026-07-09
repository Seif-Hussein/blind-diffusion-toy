"""Closed-form BDDM scale and denoising helpers."""

from __future__ import annotations

import torch


def blind_sigma_mle(prior, Y: torch.Tensor) -> torch.Tensor:
    return prior.sigma_mle(Y)


def blind_denoise_mle(prior, Y: torch.Tensor) -> torch.Tensor:
    sigma_hat = prior.sigma_mle(Y)
    out = torch.empty_like(Y)
    for sigma in torch.unique(sigma_hat):
        mask = sigma_hat == sigma
        out[mask] = prior.denoise(Y[mask], sigma)
    return out


def bddm_sigma_from_residual(Y: torch.Tensor, fY: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((fY - Y).square(), dim=1).clamp_min(0.0))

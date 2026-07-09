"""Diagnostics for correction alignment and split-induced perturbations."""

from __future__ import annotations

import torch

from posterior_bddm_oracle.src.torch_oracle_tools import cosine_similarity, relative_error


def correction_metrics(c_split, c_star):
    rel = relative_error(c_split, c_star)
    cos = cosine_similarity(c_split, c_star)
    norm_ratio = torch.linalg.norm(c_split, dim=1) / torch.linalg.norm(c_star, dim=1).clamp_min(1e-15)
    return rel, cos, norm_ratio


def perturbation_statistics(b: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = torch.mean(b, dim=0)
    centered = b - mean[None, :]
    cov = centered.T @ centered / max(1, b.shape[0] - 1)
    trace = torch.trace(cov).clamp_min(1e-30)
    eig_max = torch.linalg.eigvalsh(cov).max().clamp_min(0.0)
    bias_ratio = mean.square().sum() / trace
    anisotropy_ratio = eig_max / (trace / b.shape[1])
    return {
        "mean": mean,
        "cov": cov,
        "trace": trace,
        "bias_ratio": bias_ratio,
        "anisotropy_ratio": anisotropy_ratio,
    }

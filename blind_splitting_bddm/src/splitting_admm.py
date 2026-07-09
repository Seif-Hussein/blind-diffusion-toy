"""Small-d ADMM split helper for linear Gaussian measurements."""

from __future__ import annotations

import torch


def admm_z_update(x, u, y_obs, A, noise_std: float, mu: float):
    d = A.shape[1]
    AtA = A.T @ A / float(noise_std) ** 2
    rhs = A.T @ y_obs / float(noise_std) ** 2 + float(mu) * (x - u)
    system = AtA + float(mu) * torch.eye(d, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(system, rhs.T).T

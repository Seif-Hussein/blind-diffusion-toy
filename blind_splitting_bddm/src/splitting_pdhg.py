"""PDHG dual update and dual-law diagnostics for quadratic data terms."""

from __future__ import annotations

import torch


def pdhg_dual_update(x, y_obs, A, noise_std: float, gamma: float, w=None):
    ax = x @ A.T
    if w is None:
        w = torch.zeros_like(ax)
    h = 1.0 / float(noise_std) ** 2
    z = (h * y_obs[None, :] + float(gamma) * ax + w) / (h + float(gamma))
    w_next = w + float(gamma) * (ax - z)
    return w_next @ A, w_next, z


def quadratic_dual_memory_scalar(noise_std: float, gamma: float) -> float:
    h = 1.0 / float(noise_std) ** 2
    return h / (h + float(gamma))


def dual_tracking_residual(delta_next, delta_prev, m_now, m_prev, A, noise_std: float, gamma: float):
    h = 1.0 / float(noise_std) ** 2
    M = quadratic_dual_memory_scalar(noise_std, gamma)
    expected = M * delta_prev - M * h * ((m_now - m_prev) @ A.T)
    return delta_next - expected

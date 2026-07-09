"""Proximal-gradient/HQS-style split corrections."""

from __future__ import annotations

from posterior_bddm_oracle.src.torch_oracle_tools import data_gradient


def pg_force(x, y_obs, A, noise_std):
    return data_gradient(x, y_obs, A, noise_std)
